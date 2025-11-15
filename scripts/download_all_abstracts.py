"""Download abstracts for ALL papers using arXiv API.

This script downloads abstracts for all 408,590 papers in the dataset.
Rate limit: 3 seconds between batches (100 papers per batch).
Estimated time: ~3-4 hours for complete download.
"""
import json
import gzip
import time
import requests
from pathlib import Path
from typing import List, Dict
import xml.etree.ElementTree as ET


def load_all_arxiv_ids(papers_file: str = '../papers_index_mapping.csv.gz') -> List[str]:
    """Load ALL arxiv IDs from papers_index_mapping.csv.gz."""
    print(f"Loading all paper IDs from {papers_file}...")
    
    all_ids = []
    with gzip.open(papers_file, 'rt') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                arxiv_id = parts[1].strip()
                all_ids.append(arxiv_id)
    
    print(f"✓ Loaded {len(all_ids):,} paper IDs")
    return all_ids


def fetch_abstract_batch(arxiv_ids: List[str]) -> Dict[str, str]:
    """Fetch abstracts for a batch of arXiv IDs (max 100 per request)."""
    base_url = "http://export.arxiv.org/api/query"
    
    # Convert old format IDs (astro-ph-YYMMNNN) to new format for API
    # API expects: astro-ph/YYMMNNN instead of astro-ph-YYMMNNN
    formatted_ids = []
    for arxiv_id in arxiv_ids:
        if arxiv_id.startswith('astro-ph-'):
            # Convert astro-ph-0006237 -> astro-ph/0006237
            formatted_ids.append(arxiv_id.replace('astro-ph-', 'astro-ph/', 1))
        else:
            formatted_ids.append(arxiv_id)
    
    # Build query with ID list
    id_list = ",".join(formatted_ids)
    params = {
        'id_list': id_list,
        'max_results': len(arxiv_ids)
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        
        # Extract abstracts
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        abstracts = {}
        
        for entry in root.findall('atom:entry', ns):
            # Get arXiv ID
            id_elem = entry.find('atom:id', ns)
            if id_elem is not None:
                full_id = id_elem.text
                # Extract just the arxiv ID (e.g., "0704.0007" from full URL)
                arxiv_id = full_id.split('/abs/')[-1]
                
                # Convert back to our format (astro-ph/YYMMNNN -> astro-ph-YYMMNNN)
                if '/' in arxiv_id:
                    arxiv_id = arxiv_id.replace('/', '-')
                
                # Strip version number (e.g., v1, v2)
                if 'v' in arxiv_id:
                    arxiv_id = arxiv_id.split('v')[0]
                
                # Get abstract
                abstract_elem = entry.find('atom:summary', ns)
                if abstract_elem is not None:
                    abstract = abstract_elem.text.strip()
                    abstracts[arxiv_id] = abstract
        
        return abstracts
        
    except Exception as e:
        print(f"Error fetching batch: {e}")
        return {}


def download_all_abstracts(
    arxiv_ids: List[str], 
    output_file: Path, 
    batch_size: int = 100,
    checkpoint_interval: int = 1000
):
    """Download abstracts for all arxiv IDs and save to JSONL with checkpointing.
    
    Args:
        arxiv_ids: List of arXiv IDs
        output_file: Output JSONL file
        batch_size: Number of papers per API request (max 100)
        checkpoint_interval: Save checkpoint every N abstracts
    """
    print(f"\nDownloading abstracts for {len(arxiv_ids):,} papers...")
    print(f"Batch size: {batch_size}")
    print(f"Checkpoint interval: {checkpoint_interval:,} abstracts")
    
    # Check if partial download exists
    existing_abstracts = {}
    if output_file.exists():
        print(f"\n⚠️  Found existing file: {output_file}")
        with open(output_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                existing_abstracts[item['arxiv_id']] = item['abstract']
        print(f"   Already have {len(existing_abstracts):,} abstracts")
        
        # Filter out already downloaded
        arxiv_ids = [aid for aid in arxiv_ids if aid not in existing_abstracts]
        print(f"   Remaining: {len(arxiv_ids):,} to download")
        
        if len(arxiv_ids) == 0:
            print("✅ All abstracts already downloaded!")
            return
    
    # Split into batches
    batches = [arxiv_ids[i:i+batch_size] for i in range(0, len(arxiv_ids), batch_size)]
    print(f"Total batches: {len(batches)}")
    
    # Estimated time
    estimated_seconds = len(batches) * 3
    estimated_hours = estimated_seconds / 3600
    print(f"Estimated time: ~{estimated_hours:.1f} hours")
    
    all_abstracts = list(existing_abstracts.items())
    checkpoint_counter = 0
    
    # Process batches sequentially (arXiv API rate limit: ~1 request per 3 seconds)
    start_time = time.time()
    for i, batch in enumerate(batches, 1):
        # Calculate progress metrics
        progress_pct = (i / len(batches)) * 100
        elapsed = time.time() - start_time
        rate = i / elapsed if elapsed > 0 else 0
        remaining_batches = len(batches) - i
        eta_seconds = remaining_batches / rate if rate > 0 else 0
        eta_minutes = eta_seconds / 60
        eta_hours = eta_minutes / 60
        
        # Format ETA nicely
        if eta_hours >= 1:
            eta_str = f"{eta_hours:.1f}h"
        else:
            eta_str = f"{eta_minutes:.0f}m"
        
        # Print progress
        print(f"[{progress_pct:5.1f}%] Batch {i:,}/{len(batches):,} | Total: {len(all_abstracts):,} | ETA: {eta_str}", 
              end='', flush=True)
        
        abstracts_dict = fetch_abstract_batch(batch)
        
        # Add to results
        for arxiv_id, abstract in abstracts_dict.items():
            all_abstracts.append((arxiv_id, abstract))
            checkpoint_counter += 1
        
        print(f" ✓ +{len(abstracts_dict)}")
        
        # Save checkpoint
        if checkpoint_counter >= checkpoint_interval or i == len(batches):
            elapsed_min = elapsed / 60
            print(f"   💾 Checkpoint: {len(all_abstracts):,} abstracts | Elapsed: {elapsed_min:.1f}m | Speed: {rate*60:.1f} batches/min")
            with open(output_file, 'w') as f:
                for arxiv_id, abstract in all_abstracts:
                    f.write(json.dumps({'arxiv_id': arxiv_id, 'abstract': abstract}) + '\n')
            checkpoint_counter = 0
        
        # Respect API rate limit (3 seconds between requests)
        if i < len(batches):
            time.sleep(3)
    
    print(f"\n✅ Downloaded {len(all_abstracts):,} abstracts")
    print(f"   Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"   Coverage: {len(all_abstracts)/len(arxiv_ids)*100:.1f}%")


def main():
    """Main function."""
    # Load all arxiv IDs
    arxiv_ids = load_all_arxiv_ids()
    
    # Download abstracts
    output_file = Path('../abstracts_all.jsonl')
    print(f"\nOutput file: {output_file}")
    
    download_all_abstracts(arxiv_ids, output_file)
    
    print(f"\n✅ Complete! Abstracts saved to: {output_file}")
    print("\nNext steps:")
    print("  1. Verify coverage: wc -l abstracts_all.jsonl")
    print("  2. Compress: gzip abstracts_all.jsonl")
    print("  3. Move to main directory")


if __name__ == '__main__':
    main()

