# Scripts

Data processing and extraction scripts for the astrophysics papers dataset.

## Overview

This directory contains scripts for:
- **Abstract download**: Downloading abstracts from arXiv API
- **Citation extraction**: Extracting citation data from NASA ADS
- **Keyword extraction**: Extracting and curating ADS keywords
- **Paper processing pipeline**: Summarizing papers, organizing summaries, and extracting concepts

All scripts support automatic checkpointing and resume capability.

## Files

### Abstract Download

**`download_all_abstracts.py`** - Downloads abstracts for all 408,590 papers from arXiv API.

**Requirements**:
- No API key needed (uses public arXiv API)
- Python packages: requests

**Usage**:
```bash
python download_all_abstracts.py
```

**Output**:
- `../abstracts_all.jsonl` - JSONL file with format `{"arxiv_id": "...", "abstract": "..."}`

**Post-processing**:
```bash
gzip abstracts_all.jsonl  # Compress
gunzip -c abstracts_all.jsonl.gz | wc -l  # Verify count
```

### Citation Download

**`download_all_citations.py`** - Downloads citation data from NASA ADS for all papers.

**Requirements**:
- ADS API key in `~/.env` as `ADS=your_key_here`
- Network access

**Usage**:
```bash
python download_all_citations.py
```

**Output**:
- `citations_full_data.jsonl` - Citation data for all papers
- `citations_full_progress.json` - Progress tracking

### ADS Keywords Extraction

**`extract_all_ads_keywords.py`** - Extracts keywords from NASA ADS for all 408,590 papers using parallel processing.

**Requirements**:
- ADS API key in `~/.env` as `ADS=your_key_here`
- Python packages: pandas, requests, python-dotenv

**Usage**:
```bash
python extract_all_ads_keywords.py
```

**Output**:
- `../ads_keywords/ads_keywords_all.jsonl` - Raw keywords for all papers
- `../ads_keywords/ads_keywords_progress.json` - Progress tracking (removed after completion)

**`curate_ads_keywords.py`** - Curates raw ADS keywords by removing overly common, rare, and non-scientific keywords.

**Usage**:
```bash
python curate_ads_keywords.py
```

**Curation steps**:
1. Remove arXiv class keywords (e.g., "Astrophysics - Cosmology")
2. Normalize to lowercase
3. Remove overly common and rare keywords
4. Remove duplicates

**Input**: `../ads_keywords/ads_keywords_all.jsonl`

**Output**: `../ads_keywords/ads_keywords_curated.csv`

**Results**: 1,269,903 curated associations, 6,909 unique keywords, 73% coverage (298,657 papers)

### Paper Processing Pipeline

These three scripts form the core pipeline for processing papers from markdown to structured concept extraction:

#### 1. Summarizer

**`summarizer.py`** - Summarizes papers from markdown files using Azure OpenAI.

**Purpose**: Takes full paper markdown (from OCR tools like Nougat or Mathpix) and generates condensed summaries that preserve key scientific content, motivations, methods, and results.

**Key Features**:
- Splits papers into sections using LaTeX markers (`\section`, `\subsection`)
- Preserves LaTeX formulas and technical content
- Uses proportional compression (allocates summary space based on section length)
- Parallel processing with configurable workers
- Automatic retry logic with exponential backoff

**Requirements**:
- Azure OpenAI API credentials (endpoint, key, deployment name)
- Python packages: `openai`, `python-dotenv`
- Environment variables in `~/.env`:
  ```
  GPT_Cloud_Bank=your_azure_api_key
  ENDPOINT_URL=your_azure_endpoint
  DEPLOYMENT_NAME=your_deployment_name
  ```

**Input**: Markdown files with LaTeX-style section markers

**Output**: Compressed summary files (`*_summary.md`) preserving key scientific content

**Configuration**: Requires Azure OpenAI setup with appropriate models (GPT-4o or similar)

#### 2. Organizer

**`organizer.py`** - Reorganizes summaries into structured JSON format.

**Purpose**: Takes the compressed summaries and restructures them into a standardized JSON format with specific fields for background, motivation, methodology, results, interpretation, and implication.

**Key Features**:
- Converts narrative summaries into structured JSON with 7 key fields
- Ensures third-person perspective and removes section references
- Validates JSON output and handles escape characters properly
- Parallel processing with retry logic
- Creates logical flow between sections

**Requirements**:
- Azure OpenAI API credentials
- Python packages: openai, json, re

**Input**: Summary markdown files (`*_summary.md`)

**Output**: Structured JSON files (`*_organized.json`) with fields:
  - `title_and_author`
  - `background`
  - `motivation`
  - `methodology`
  - `results`
  - `interpretation`
  - `implication`

**Configuration**: Requires Azure OpenAI setup

#### 3. Extractor

**`extractor.py`** - Extracts key concepts and classifications from organized summaries.

**Purpose**: Analyzes the structured paper summaries to identify approximately 10 key concepts per paper, with technical descriptions and domain classifications.

**Key Features**:
- Extracts ~10 concepts per paper covering both scientific and technical content
- Assigns concepts to 8 hierarchical domains:
  - Cosmology & Nongalactic Physics
  - Galaxy Physics
  - High Energy Astrophysics
  - Solar & Stellar Physics
  - Earth & Planetary Science
  - Statistics & AI
  - Numerical Simulation
  - Instrumental Design
- Generates ~100-word technical descriptions for each concept
- Validates JSON structure and handles parsing errors
- Parallel processing with retry logic

**Requirements**:
- Azure OpenAI API credentials
- Python packages: openai, json

**Input**: Organized JSON files (`*_organized.json`)

**Output**: Concept JSON files (`*_concepts.json`) with arrays of:
```json
[
  {
    "concept": "Concept Name",
    "class": "Domain Classification",
    "description": "Technical description (~100 words)"
  }
]
```

**Configuration**: Requires Azure OpenAI setup

#### Pipeline Flow

The complete pipeline operates in sequence:

1. **Paper → Markdown**: Use OCR tools (Nougat, Mathpix) to convert PDFs to markdown
2. **Markdown → Summary**: Run `summarizer.py` to compress papers into key content
3. **Summary → Organized JSON**: Run `organizer.py` to structure summaries
4. **Organized JSON → Concepts**: Run `extractor.py` to extract key concepts
5. **Concepts → Clustering**: Apply clustering to merge similar concepts into 9,999 unique classes

**Note**: These scripts require Azure OpenAI API access and are provided for reproducibility. The dataset already includes the final outputs (organized summaries and extracted concepts).

**Security Note**: The scripts handle API credentials securely. API keys are loaded from environment variables and are never logged or printed. Exception messages from the OpenAI library do not expose API keys.

## Getting ADS API Key

1. Create an account at https://ui.adsabs.harvard.edu/
2. Go to Account → Settings → API Token
3. Copy your token to `~/.env` as `ADS=your_api_key_here`

All scripts are designed to be idempotent (safe to re-run) and automatically save progress for resumable execution.
