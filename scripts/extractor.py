"""Extract key concepts from organized summaries using Azure OpenAI.

Configuration:
    Set these environment variables in ~/.env:
    - GPT_Cloud_Bank: Azure OpenAI API key
    - ENDPOINT_URL: Azure OpenAI endpoint (default: https://astromlab.openai.azure.com/)
    - DEPLOYMENT_NAME: Azure deployment name (default: gpt-4o)
    
    Or modify the Config class below with your settings.
"""
import json
import os
import time
from pathlib import Path
from typing import List, Optional
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path.home() / ".env")


class Config:
    """Configuration for concept extraction."""
    AZURE_API_KEY = os.getenv("GPT_Cloud_Bank")
    AZURE_ENDPOINT = os.getenv("ENDPOINT_URL", "https://astromlab.openai.azure.com/")
    AZURE_DEPLOYMENT = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
    AZURE_API_VERSION = "2024-08-01-preview"
    MAX_RETRIES = 10
    RETRY_DELAY = 30
    
    # Default paths
    PROJECT_ROOT = Path(__file__).parent.parent
    ORGANIZED_DIR = PROJECT_ROOT / "outputs" / "organized"
    CONCEPTS_DIR = PROJECT_ROOT / "outputs" / "concepts"


class ConceptExtractor:
    """Extract key concepts and classifications from papers."""
    
    def __init__(self):
        """Initialize the extractor."""
        self.config = Config
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=Config.AZURE_ENDPOINT,
            api_key=Config.AZURE_API_KEY,
            api_version=Config.AZURE_API_VERSION
        )
    
    def extract_concepts(
        self,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        skip_existing: bool = True,
        max_workers: int = 5
    ) -> List[str]:
        """
        Extract concepts from all organized summaries.
        
        Args:
            input_dir: Directory containing organized JSON files
            output_dir: Directory to save extracted concepts
            skip_existing: Skip already processed files
            max_workers: Number of parallel workers
        
        Returns:
            List of successfully processed arxiv IDs
        """
        if input_dir is None:
            input_dir = self._find_latest_organized_dir()
            if input_dir is None:
                print("Error: No organized summary directories found")
                return []
        else:
            input_dir = Path(input_dir)
        
        if output_dir is None:
            date_str = input_dir.name
            output_dir = Config.CONCEPTS_DIR / date_str
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find organized summaries to process
        organized_files = []
        for f in sorted(input_dir.glob("*_organized.json")):
            concept_file = output_dir / f.name.replace('_organized.json', '_concepts.json')
            if skip_existing and concept_file.exists():
                continue
            organized_files.append(f)
        
        if not organized_files:
            print("No new organized summaries to process")
            return []
        
        print(f"Extracting concepts from {len(organized_files)} papers with {max_workers} workers...")
        
        # Process in parallel
        successful = []
        
        def process_organized(organized_file: Path) -> tuple:
            """Extract concepts from a single organized summary."""
            try:
                concepts = self._extract_single(organized_file)
                
                if not concepts:
                    return (False, None)
                
                # Save concepts
                concept_file = output_dir / organized_file.name.replace('_organized.json', '_concepts.json')
                with open(concept_file, 'w', encoding='utf-8') as f:
                    json.dump(concepts, f, indent=2, ensure_ascii=False)
                
                return (True, organized_file.stem.replace('_organized', ''))
                
            except Exception as e:
                print(f"Error extracting concepts from {organized_file.name}: {e}")
                return (False, None)
        
        # Submit all tasks
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_organized, f): f for f in organized_files}
            
            for future in as_completed(futures):
                success, arxiv_id = future.result()
                if success and arxiv_id:
                    successful.append(arxiv_id)
        
        print(f"✓ Extracted concepts from {len(successful)}/{len(organized_files)} papers")
        return successful
    
    def _extract_single(self, organized_file: Path) -> Optional[List[dict]]:
        """Extract concepts from a single organized summary."""
        with open(organized_file, 'r', encoding='utf-8') as f:
            organized = json.load(f)
        
        # Convert organized dict to text
        paper_content = "\n\n".join([
            f"{k.replace('_', ' ').title()}: {v}"
            for k, v in organized.items() if v
        ])
        
        system_prompt = """
You are an AI specializing in astrophysics, tasked with extracting key concepts from journal articles and providing technical descriptions for each. These concepts will be used to construct a knowledge graph, so focus on identifying the most relevant and informative concepts. Emphasize key innovations of the papers rather than references in the introduction. Extract both scientific concepts in astronomy and technological concepts, including techniques in machine learning, statistics, and numerical simulations.

Consider the following guidelines when extracting concepts:

**Relevance**: Ensure the concepts are directly related to the core findings and innovations of the paper.

**Clarity**: Extract clear and specific concepts that can be easily understood and categorized. Aim to limit each concept to three to four words.

**Classification**: Identify the appropriate class for each concept. Classes can include:

- Galaxy Physics (e.g., "Galaxy Formation", "Spiral Galaxies", "Dwarf Galaxies", "Intergalactic Medium", "Galactic Nuclei")

- Cosmology & Nongalactic Physics (e.g., "Dark Matter", "Cosmic Microwave Background", "Large-Scale Structure", "Cosmic Inflation", "Cosmological Parameters")

- Earth & Planetary Science (e.g., "Exoplanet Detection", "Planetary Atmospheres", "Astrobiology", "Planetary Formation", "Solar System Evolution")

- High Energy Astrophysics (e.g., "Black Hole Physics", "Neutron Stars", "Gamma-Ray Bursts", "Supernovae", "High-Energy Cosmic Rays")

- Solar & Stellar Physics (e.g., "Stellar Evolution", "Solar Flares", "Star Formation", "Stellar Atmospheres", "Helioseismology")

- Statistics & AI (e.g., "Machine Learning Algorithms", "Bayesian Inference", "Neural Networks", "Statistical Analysis", "Data Mining")

- Numerical Simulation (e.g., "N-body Simulations", "Hydrodynamic Simulations", "Radiative Transfer", "Simulation Codes", "Computational Astrophysics")

- Instrumental Design (e.g., "Telescope Design", "Spectrographs", "Detector Technology", "Observational Techniques", "Space Telescopes")

Note that the examples above are just purely for reference, unless they fit as concepts in the paper, do not use them. Aim to extract about 10 key concepts for each paper.

For each concept, also provide a concise technical description (~100 words) explaining its general principles and significance in astronomy. While you may reference the paper's context, focus on broadly defining and explaining each concept.

**Description Guidelines:**
1. Ensure each description is technically precise and suitable for an astronomy expert audience
2. Do not use backslashes or special characters in your descriptions as they can cause JSON parsing errors
3. Use only verbal descriptions - no mathematical equations or formulas
4. Focus on clear, conceptual explanations using words
5. Keep concept names exactly as provided - do not add plurals, capitalization, or prepositions
6. Don't output any RA and Dec information in the description

Provide your response as a JSON array of concept objects. Begin your output with the JSON structure immediately, without any preceding text. Strictly adhere to the specified output format.
"""
        
        user_prompt = f"""
Output format (JSON array):
[
    {{
        "concept": "[Concept]",
        "class": "[Class]",
        "description": "[Technical description (~100 words)]"
    }}
]

Summary: {paper_content}

Extract approximately 10 key concepts. Output only valid JSON array, no other text.
"""
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=Config.AZURE_DEPLOYMENT
                )
                
                # Parse and validate JSON
                content = response.choices[0].message.content.strip()
                
                # Remove markdown code blocks if present
                if content.startswith('```'):
                    content = content.split('\n', 1)[1]
                    if content.endswith('```'):
                        content = content.rsplit('\n', 1)[0]
                
                concepts = json.loads(content)
                
                # Validate structure
                if isinstance(concepts, list) and all('concept' in c and 'class' in c and 'description' in c for c in concepts):
                    return concepts
                else:
                    raise ValueError("Invalid concept structure")
                
            except json.JSONDecodeError as e:
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(Config.RETRY_DELAY)
                    continue
                else:
                    print(f"JSON parsing failed for {organized_file.name}: {e}")
                    return None
            except Exception as e:
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(Config.RETRY_DELAY)
                    continue
                else:
                    raise
        
        return None
    
    def _find_latest_organized_dir(self) -> Optional[Path]:
        """Find the most recent organized directory."""
        organized_dirs = sorted(
            [d for d in Config.ORGANIZED_DIR.iterdir() if d.is_dir()],
            reverse=True
        )
        if organized_dirs:
            return organized_dirs[0]
        return None

