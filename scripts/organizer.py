"""Organize summaries into structured JSON format using Azure OpenAI.

Configuration:
    Set these environment variables in ~/.env:
    - GPT_Cloud_Bank: Azure OpenAI API key
    - ENDPOINT_URL: Azure OpenAI endpoint (default: https://astromlab.openai.azure.com/)
    - DEPLOYMENT_NAME: Azure deployment name (default: gpt-4o)
    
    Or modify the Config class below with your settings.
"""
import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path.home() / ".env")


class Config:
    """Configuration for summary organization."""
    AZURE_API_KEY = os.getenv("GPT_Cloud_Bank")
    AZURE_ENDPOINT = os.getenv("ENDPOINT_URL", "https://astromlab.openai.azure.com/")
    AZURE_DEPLOYMENT = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
    AZURE_API_VERSION = "2024-08-01-preview"
    MAX_RETRIES = 10
    RETRY_DELAY = 30
    
    # Default paths
    PROJECT_ROOT = Path(__file__).parent.parent
    SUMMARIES_DIR = PROJECT_ROOT / "outputs" / "summaries"
    ORGANIZED_DIR = PROJECT_ROOT / "outputs" / "organized"


class SummaryOrganizer:
    """Reorganize summaries into structured JSON format."""
    
    def __init__(self):
        """Initialize the organizer."""
        self.config = Config
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=Config.AZURE_ENDPOINT,
            api_key=Config.AZURE_API_KEY,
            api_version=Config.AZURE_API_VERSION
        )
    
    def organize_summaries(
        self,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        skip_existing: bool = True,
        max_workers: int = 5
    ) -> List[str]:
        """
        Organize all summaries into structured JSON format.
        
        Args:
            input_dir: Directory containing summary files
            output_dir: Directory to save organized summaries
            skip_existing: Skip already organized summaries
            max_workers: Number of parallel workers
        
        Returns:
            List of successfully processed arxiv IDs
        """
        if input_dir is None:
            input_dir = self._find_latest_summary_dir()
            if input_dir is None:
                print("Error: No summary directories found")
                return []
        else:
            input_dir = Path(input_dir)
        
        if output_dir is None:
            date_str = input_dir.name
            output_dir = Config.ORGANIZED_DIR / date_str
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find summaries to process
        summary_files = []
        for f in sorted(input_dir.glob("*_summary.md")):
            organized_file = output_dir / f.name.replace('_summary.md', '_organized.json')
            if skip_existing and organized_file.exists():
                continue
            summary_files.append(f)
        
        if not summary_files:
            print("No new summaries to organize")
            return []
        
        print(f"Organizing {len(summary_files)} summaries with {max_workers} workers...")
        
        # Process in parallel
        successful = []
        
        def process_summary(summary_file: Path) -> tuple:
            """Process a single summary."""
            try:
                organized = self._organize_single_summary(summary_file)
                
                if not organized:
                    return (False, None)
                
                # Save as JSON
                organized_file = output_dir / summary_file.name.replace('_summary.md', '_organized.json')
                with open(organized_file, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(organized, indent=2, ensure_ascii=False))
                
                return (True, summary_file.stem.replace('_summary', ''))
                
            except Exception as e:
                print(f"Error organizing {summary_file.name}: {e}")
                return (False, None)
        
        # Submit all tasks
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_summary, f): f for f in summary_files}
            
            for future in as_completed(futures):
                success, arxiv_id = future.result()
                if success and arxiv_id:
                    successful.append(arxiv_id)
        
        print(f"✓ Organized {len(successful)}/{len(summary_files)} summaries")
        return successful
    
    def _organize_single_summary(self, summary_file: Path) -> Optional[dict]:
        """Organize a single summary into structured format."""
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = f.read()
        
        system_prompt = """
You are an AI specializing in astrophysics, tasked with reorganizing astrophysics paper summaries. 
Adhere to these guidelines:

1. Reorganize the summary strictly into the following key areas and nothing else:
   - Title and Author
   - Background
   - Motivation
   - Methodology
   - Results
   - Interpretation
   - Implication

2. Output the reorganized summary as a valid JSON object with these exact keys:
   {
     "title_and_author": "",
     "background": "",
     "motivation": "",
     "methodology": "",
     "results": "",
     "interpretation": "",
     "implication": ""
   }

3. **Writing style:**
   - Use THIRD PERSON only: "this study", "the authors", "the paper"
   - NEVER use first person: no "we", "our", "I"
   - Remove ALL references to specific sections, figures, tables, or appendices (e.g., "Section 3", "Figure 2", "Table 1")
   - Write in continuous narrative form, avoiding bullet points or lists

4. **Logical flow:**
   - Background → Motivation should connect naturally (Background sets context, Motivation explains why this work matters)
   - Motivation → Methodology should flow logically (Motivation identifies gap, Methodology describes approach)
   - Methodology → Results should be connected (Methods used lead to Results obtained)
   - Results → Interpretation → Implication should build on each other progressively

5. Ensure as much as possible information from the original summary is included.
6. Do not add any new information beyond what is already in the summary.
7. Retain any LaTeX formulas present in the original summary.
8. Keep technical jargon intact, as it's meant for astrophysics researchers.
9. Ensure the title and first author (with "et al." if applicable) are under the "title_and_author" section.
10. If there is no information for a particular section, leave the value as an empty string but keep the key.
11. Integrate all related information into the main sections without creating any subsections.
12. Each section must appear exactly once. Do not duplicate any section.
13. Omit the "References" section entirely.
14. Ensure the output is valid JSON that can be parsed. Be careful with escape characters - use proper JSON escaping for quotes, backslashes, and other special characters.
15. Format the JSON output with line breaks between each key-value pair for better readability.
"""
        
        user_prompt = f"Please reorganize the following astrophysics paper summary strictly into the key areas specified in the guidelines, outputting as valid JSON. Here's the summary:\n\n{summary}"
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=Config.AZURE_DEPLOYMENT
                )
                
                # Clean and validate JSON
                content = response.choices[0].message.content
                content = re.sub(r'```json\s*', '', content)
                content = re.sub(r'```\s*$', '', content)
                content = content.strip()
                
                # Validate JSON
                parsed = json.loads(content)
                return parsed
                
            except json.JSONDecodeError as e:
                if attempt < Config.MAX_RETRIES - 1:
                    # Exponential backoff: 2, 4, 8, 16 seconds
                    delay = Config.RETRY_DELAY * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    print(f"JSON validation failed for {summary_file.name}: {e}")
                    return None
            except Exception as e:
                if attempt < Config.MAX_RETRIES - 1:
                    # Exponential backoff with jitter for rate limits
                    delay = Config.RETRY_DELAY * (2 ** attempt) + (time.time() % 1)
                    time.sleep(delay)
                    continue
                else:
                    raise
        
        return None
    
    def _find_latest_summary_dir(self) -> Optional[Path]:
        """Find the most recent summary directory."""
        summary_dirs = sorted(
            [d for d in Config.SUMMARIES_DIR.iterdir() if d.is_dir()],
            reverse=True
        )
        if summary_dirs:
            return summary_dirs[0]
        return None

