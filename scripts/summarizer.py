"""Summarize papers using Azure OpenAI.

Configuration:
    Set these environment variables in ~/.env:
    - GPT_Cloud_Bank: Azure OpenAI API key
    - ENDPOINT_URL: Azure OpenAI endpoint (default: https://astromlab.openai.azure.com/)
    - DEPLOYMENT_NAME: Azure deployment name (default: gpt-4o)
    
    Or modify the Config class below with your settings.
"""
import re
import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Optional
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path.home() / ".env")


def setup_logger(name: str, level: int = logging.ERROR):
    """Set up minimal logger (errors only by default)."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger


class Config:
    """Configuration for paper summarization."""
    AZURE_API_KEY = os.getenv("GPT_Cloud_Bank")
    AZURE_ENDPOINT = os.getenv("ENDPOINT_URL", "https://astromlab.openai.azure.com/")
    AZURE_DEPLOYMENT = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
    AZURE_API_VERSION = "2024-08-01-preview"
    
    # Summarization settings
    CHUNK_SIZE = 10000
    GPT5_TEMPERATURE = 1.0
    GPT5_MAX_TOKENS = 16384
    MAX_RETRIES = 10
    RETRY_DELAY = 30
    
    # Default paths
    PROJECT_ROOT = Path(__file__).parent.parent
    MARKDOWN_DIR = PROJECT_ROOT / "outputs" / "markdown"
    SUMMARIES_DIR = PROJECT_ROOT / "outputs" / "summaries"


class PaperSplitter:
    """Split papers into chunks for summarization."""
    
    @staticmethod
    def merge_nearby_small_chunks(strings: List[str], limit: int) -> List[str]:
        """Merge consecutive small chunks up to the character limit."""
        if not strings:
            return []
        
        merged_strings = []
        current_string = strings[0]
        
        for i in range(1, len(strings)):
            if len(current_string) + len(strings[i]) <= limit:
                current_string += " " + strings[i]
            else:
                merged_strings.append(current_string)
                current_string = strings[i]
        
        merged_strings.append(current_string)
        return merged_strings
    
    def split_paper(self, paper_content: str, num_char_limit: int) -> List[str]:
        """
        Split paper content using LaTeX section markers.
        
        Args:
            paper_content: Full paper markdown content
            num_char_limit: Maximum characters per chunk
        
        Returns:
            List of text chunks
        """
        # Extract title and author sections
        title_pattern = r'(\\title\{[^}]+\})'
        author_pattern = r'\\author\{[\s\S]*?\n\}'
        
        title_match = re.search(title_pattern, paper_content)
        author_match = re.search(author_pattern, paper_content)
        
        header = ""
        if title_match:
            header += title_match.group(0) + "\n\n"
        if author_match:
            header += author_match.group(0) + "\n\n"
        
        # Remove title and author from content for section processing
        if header:
            if title_match:
                paper_content = paper_content.replace(title_match.group(0), "", 1)
            if author_match:
                paper_content = paper_content.replace(author_match.group(0), "", 1)
        
        # Split by sections using regex
        pattern = r'(\\section\*?\{[^}]+\})'
        parts = re.split(pattern, paper_content)
        
        # Combine section headings with their content
        sections = []
        if header:
            sections.append(header.strip())
        if parts[0].strip():
            sections.append(parts[0].strip())
        
        for i in range(1, len(parts), 2):
            heading = parts[i].strip()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            chunk = heading + "\n" + content
            sections.append(chunk)
        
        # Further split any section that exceeds num_char_limit
        chunk_list = []
        for section in sections:
            if len(section) <= num_char_limit:
                chunk_list.append(section)
            else:
                # Split large sections into smaller chunks
                for j in range(0, len(section), num_char_limit):
                    chunk_list.append(section[j:j + num_char_limit])
        
        # Merge small nearby chunks
        merged_list = self.merge_nearby_small_chunks(chunk_list, num_char_limit)
        return merged_list


class PaperSummarizer:
    """Summarize papers using Azure OpenAI GPT-5."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the summarizer."""
        self.config = Config
        self.verbose = verbose
        self.logger = setup_logger("summarizer", level=logging.ERROR)
        self.splitter = PaperSplitter()
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=Config.AZURE_ENDPOINT,
            api_key=Config.AZURE_API_KEY,
            api_version=Config.AZURE_API_VERSION
        )
    
    def summarize_papers(
        self,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        skip_existing: bool = True,
        max_workers: int = 5
    ) -> List[str]:
        """
        Summarize all papers in the input directory with parallel processing.
        
        Args:
            input_dir: Directory containing markdown files (defaults to latest)
            output_dir: Directory to save summaries (defaults to config)
            skip_existing: Whether to skip papers that already have summaries
            max_workers: Number of parallel workers for summarization
        
        Returns:
            List of successfully processed arxiv IDs
        """
        if input_dir is None:
            # Find the latest date directory
            input_dir = self._find_latest_markdown_dir()
            if input_dir is None:
                self.logger.error("No markdown directories found")
                return []
        else:
            input_dir = Path(input_dir)
        
        if output_dir is None:
            # Use same date structure for summaries
            date_str = input_dir.name
            output_dir = Config.SUMMARIES_DIR / date_str
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Input directory: {input_dir}")
        self.logger.info(f"Output directory: {output_dir}")
        
        # Find markdown files to process
        md_files = []
        for f in sorted(input_dir.glob("*.md")):
            if f.name == "processed_papers.json":
                continue
            
            summary_file = output_dir / f.name.replace('.md', '_summary.md')
            if skip_existing and summary_file.exists():
                self.logger.info(f"Skipping existing summary: {f.name}")
                continue
            
            md_files.append(f)
        
        self.logger.info(f"Found {len(md_files)} papers to summarize")
        self.logger.info(f"Using {max_workers} parallel workers")
        
        # Process papers in parallel
        successful = []
        
        def process_paper(md_file: Path, idx: int) -> tuple:
            """Process a single paper and return (success, arxiv_id)."""
            try:
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"Processing [{idx}/{len(md_files)}]: {md_file.name}")
                self.logger.info(f"{'='*80}")
                
                summary = self._summarize_single_paper(md_file)
                
                # Validate summary
                if not summary or not summary.strip():
                    self.logger.warning(f"Empty result for {md_file.name}, skipping...")
                    return (False, None)
                
                # Save summary
                summary_file = output_dir / md_file.name.replace('.md', '_summary.md')
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(summary)
                
                self.logger.info(f"Successfully wrote summary to {summary_file.name}")
                return (True, md_file.stem)
                
            except Exception as e:
                self.logger.error(f"Error processing {md_file.name}: {e}")
                return (False, None)
        
        # Submit all tasks to thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_paper, md_file, idx): md_file 
                for idx, md_file in enumerate(md_files, start=1)
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                success, arxiv_id = future.result()
                if success and arxiv_id:
                    successful.append(arxiv_id)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Successfully summarized {len(successful)}/{len(md_files)} papers")
        self.logger.info(f"{'='*80}")
        
        return successful
    
    def _summarize_single_paper(self, md_file: Path) -> str:
        """Summarize a single paper."""
        with open(md_file, 'r', encoding='utf-8') as f:
            paper_content = f.read()
        
        # Extract title and author sections
        title_pattern = r'(\\title\{[^}]+\})'
        author_pattern = r'\\author\{[\s\S]*?\n\}'
        
        title_match = re.search(title_pattern, paper_content)
        author_match = re.search(author_pattern, paper_content)
        
        header = ""
        if title_match:
            header += title_match.group(0) + "\n\n"
        if author_match:
            header += author_match.group(0) + "\n\n"
        
        # Remove title and author from content
        if title_match:
            paper_content = paper_content.replace(title_match.group(0), "", 1)
        if author_match:
            paper_content = paper_content.replace(author_match.group(0), "", 1)
        
        # Split paper into chunks
        chunk_list = self.splitter.split_paper(
            paper_content,
            num_char_limit=Config.CHUNK_SIZE
        )
        
        self.logger.info(f"Processing {len(chunk_list)} chunks for paper {md_file.name}")
        
        # Compress each chunk
        compressed_chunk_list = []
        context = ''
        total_char_limit = 10000
        
        for i, chunk in enumerate(chunk_list):
            self.logger.info(f"\nProcessing chunk {i+1}/{len(chunk_list)}")
            
            # Calculate proportional character limit
            num_chunk_char_limit = int(
                len(chunk) * (total_char_limit / sum([len(c) for c in chunk_list]))
            )
            
            try:
                compressed = self._compress_chunk(chunk, num_chunk_char_limit, context)
                compressed_chunk_list.append(compressed)
                context = '\n\n'.join(compressed_chunk_list)
            except Exception as e:
                self.logger.error(f"Error compressing chunk {i+1}: {e}")
                raise
        
        return header + context
    
    def _compress_chunk(self, chunk: str, num_chunk_char_limit: int, context: str) -> str:
        """Compress a single chunk using Azure OpenAI GPT-5."""
        system_prompt = f"""
You are an AI specializing in astrophysics, tasked with condensing astrophysics journal texts. Adhere to these guidelines:
1. Retain LaTeX code for formulas, remove other LaTeX symbols.
2. Exclude acknowledgments and appendices at the end of the paper.
3. Emphasize the paper's motivations, novel technical details, key theories, and concepts.
4. Highlight innovative results and their links to other works.
5. Integrate information from figures' captions, omit figures.
6. Retain or condense section and subsection titles, avoiding redundancy. Use '\\\\section{{...}}', '\\\\subsection{{...}}' and so forth to clearly delineate different sections.
7. Clarify or maintain technical jargon at the level that is clear for astrophysics researchers.
8. Convey the author's perspective and interpretation of results.
Consider context from previous parts when summarizing individual sections. Exclude references at the end. Current context: {context}
"""
        
        user_prompt = (
            f"Condense the following text into a maximum of {num_chunk_char_limit} "
            f"characters, avoiding repetition of the provided context. "
            f"Exclude references at the end. Paragraph: {chunk}"
        )
        
        # Retry logic
        for attempt in range(Config.MAX_RETRIES):
            try:
                completion = self.client.chat.completions.create(
                    model=Config.AZURE_DEPLOYMENT,
                    messages=[
                        {"role": "developer", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=Config.GPT5_MAX_TOKENS,
                    temperature=Config.GPT5_TEMPERATURE,
                    stop=None
                )
                return completion.choices[0].message.content
            except Exception as e:
                if attempt < Config.MAX_RETRIES - 1:
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {Config.RETRY_DELAY}s..."
                    )
                    time.sleep(Config.RETRY_DELAY)
                else:
                    raise Exception(f"Failed after {Config.MAX_RETRIES} attempts: {e}")
    
    def _find_latest_markdown_dir(self) -> Optional[Path]:
        """Find the most recent markdown directory."""
        markdown_dirs = sorted(
            [d for d in Config.MARKDOWN_DIR.iterdir() if d.is_dir()],
            reverse=True
        )
        if markdown_dirs:
            return markdown_dirs[0]
        return None


