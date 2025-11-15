"""Extract embeddings for abstracts and summary sections using parallel workers."""
import json
import gzip
import os
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed


# Load environment variables
load_dotenv(Path.home() / '.env')


class EmbeddingExtractor:
    """Extract embeddings using OpenAI text-embedding-3-large with parallel workers."""
    
    def __init__(self, api_key: Optional[str] = None, max_workers: int = 20):
        """Initialize the extractor."""
        if api_key is None:
            api_key = os.getenv('OPENAI')
        
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI in ~/.env")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "text-embedding-3-large"
        self.embedding_dim = 3072
        self.max_workers = max_workers
    
    def get_embedding(self, text: str, retries: int = 3) -> Optional[np.ndarray]:
        """Get embedding for a single text with retry logic."""
        for attempt in range(retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                    encoding_format="float"
                )
                
                embedding = np.array(response.data[0].embedding, dtype=np.float32)
                return embedding
                
            except Exception as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)
                else:
                    print(f"Failed after {retries} attempts: {e}")
                    return None
        
        return None
    
    def extract_abstract_embeddings(
        self,
        abstracts_file: Path,
        output_file: Path
    ):
        """Extract embeddings for all abstracts using parallel workers."""
        print(f"\n{'='*60}")
        print(f"EXTRACTING ABSTRACT EMBEDDINGS")
        print(f"{'='*60}")
        print(f"Model: {self.model}")
        print(f"Workers: {self.max_workers}")
        
        # Load abstracts
        abstracts = []
        with open(abstracts_file, 'r') as f:
            for line in f:
                abstracts.append(json.loads(line))
        
        print(f"Loaded {len(abstracts):,} abstracts")
        print(f"Starting extraction...\n")
        
        # Process in parallel
        results = []
        
        def process_one(item: Dict) -> Optional[Dict]:
            """Process a single abstract."""
            arxiv_id = item['arxiv_id']
            abstract = item['abstract']
            
            embedding = self.get_embedding(abstract)
            
            if embedding is not None:
                return {
                    'arxiv_id': arxiv_id,
                    'embedding': embedding
                }
            return None
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_one, item): i for i, item in enumerate(abstracts)}
            
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    results.append(result)
                
                if i % 100 == 0:
                    print(f"Progress: {i}/{len(abstracts)} ({i/len(abstracts)*100:.1f}%)")
        
        # Save embeddings
        print(f"\nSaving {len(results):,} embeddings...")
        
        # Convert to arrays
        arxiv_ids = [r['arxiv_id'] for r in results]
        embeddings = np.array([r['embedding'] for r in results])
        
        # Save
        np.savez(output_file, embeddings=embeddings, arxiv_ids=arxiv_ids)
        
        print(f"✅ Saved to {output_file}")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Success rate: {len(results)/len(abstracts)*100:.1f}%")
        
        return arxiv_ids
    
    def extract_summary_embeddings(
        self,
        arxiv_ids: List[str],
        summaries_file: Path,
        output_file: Path
    ):
        """Extract embeddings for each section of summaries using parallel workers."""
        print(f"\n{'='*60}")
        print(f"EXTRACTING SUMMARY SECTION EMBEDDINGS")
        print(f"{'='*60}")
        print(f"Model: {self.model}")
        print(f"Workers: {self.max_workers}")
        
        # Load summaries for sampled papers
        print(f"Loading summaries for {len(arxiv_ids):,} papers...")
        arxiv_id_set = set(arxiv_ids)
        summaries = {}
        
        with gzip.open(summaries_file, 'rt') as f:
            for line in f:
                data = json.loads(line)
                arxiv_id = data['arxiv_id']
                if arxiv_id in arxiv_id_set:
                    summaries[arxiv_id] = data['summary']
                
                if len(summaries) >= len(arxiv_ids):
                    break
        
        print(f"Loaded {len(summaries):,} summaries")
        print(f"Starting extraction...\n")
        
        # Extract embeddings for each section
        section_names = ['background', 'motivation', 'methodology', 'results', 'interpretation', 'implication']
        
        # Process papers in parallel
        results = []
        
        def process_one(arxiv_id: str) -> Optional[Dict]:
            """Process a single paper's summary."""
            if arxiv_id not in summaries:
                return None
            
            summary = summaries[arxiv_id]
            
            embeddings = {}
            for section in section_names:
                if section in summary:
                    text = summary[section]
                    embedding = self.get_embedding(text)
                    if embedding is not None:
                        embeddings[section] = embedding
            
            # Only keep if we got all sections
            if len(embeddings) == len(section_names):
                return {
                    'arxiv_id': arxiv_id,
                    'section_embeddings': embeddings
                }
            return None
        
        arxiv_ids_list = list(summaries.keys())
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_one, arxiv_id): i for i, arxiv_id in enumerate(arxiv_ids_list)}
            
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    results.append(result)
                
                if i % 50 == 0:
                    print(f"Progress: {i}/{len(arxiv_ids_list)} ({i/len(arxiv_ids_list)*100:.1f}%)")
        
        # Save embeddings
        print(f"\nSaving {len(results):,} section embeddings...")
        
        # Convert to structured arrays
        valid_ids = [r['arxiv_id'] for r in results]
        section_data = {section: [] for section in section_names}
        
        for result in results:
            for section in section_names:
                section_data[section].append(result['section_embeddings'][section])
        
        # Convert to numpy arrays
        for section in section_names:
            section_data[section] = np.array(section_data[section])
        
        # Save with arxiv_ids
        section_data['arxiv_ids'] = np.array(valid_ids)
        np.savez(output_file, **section_data)
        
        print(f"✅ Saved to {output_file}")
        print(f"   Papers with complete sections: {len(valid_ids):,}")
        print(f"   Success rate: {len(valid_ids)/len(arxiv_ids_list)*100:.1f}%")
        for section in section_names:
            print(f"   {section}: {section_data[section].shape}")
        
        return valid_ids


def main():
    """Main function."""
    print("\n" + "="*60)
    print("EMBEDDING EXTRACTION FOR ABSTRACT vs. SUMMARY ANALYSIS")
    print("="*60)
    
    # Initialize extractor with more workers for faster processing
    extractor = EmbeddingExtractor(max_workers=20)
    
    # Extract abstract embeddings
    abstracts_file = Path('abstracts_sample.jsonl')
    abstract_embeddings_file = Path('abstract_embeddings.npz')
    
    if abstract_embeddings_file.exists():
        print(f"\n{'='*60}")
        print(f"ABSTRACT EMBEDDINGS ALREADY EXIST - SKIPPING")
        print(f"{'='*60}")
        # Load existing IDs
        abstract_data = np.load(abstract_embeddings_file, allow_pickle=True)
        arxiv_ids = list(abstract_data['arxiv_ids'])
        print(f"Loaded {len(arxiv_ids):,} arxiv IDs from existing file")
    elif abstracts_file.exists():
        arxiv_ids = extractor.extract_abstract_embeddings(
            abstracts_file,
            abstract_embeddings_file
        )
    else:
        print(f"Error: {abstracts_file} not found. Run download_abstracts.py first.")
        return
    
    # Extract summary section embeddings
    summaries_file = Path('../papers_summaries.jsonl.gz')
    summary_embeddings_file = Path('section_embeddings.npz')
    
    if summary_embeddings_file.exists():
        print(f"\n{'='*60}")
        print(f"SECTION EMBEDDINGS ALREADY EXIST - SKIPPING")
        print(f"{'='*60}")
        # Load existing IDs
        section_data = np.load(summary_embeddings_file, allow_pickle=True)
        valid_ids = list(section_data['arxiv_ids'])
        print(f"Loaded {len(valid_ids):,} papers with complete sections")
    elif summaries_file.exists():
        valid_ids = extractor.extract_summary_embeddings(
            arxiv_ids,
            summaries_file,
            summary_embeddings_file
        )
    else:
        print(f"Error: {summaries_file} not found.")
        return
    
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Abstract embeddings: {abstract_embeddings_file}")
    print(f"✅ Section embeddings: {summary_embeddings_file}")
    print(f"✅ Papers with both: {len(valid_ids):,}")
    print(f"\nReady for analysis! Run: jupyter notebook embedding_analysis.ipynb")


if __name__ == '__main__':
    main()

