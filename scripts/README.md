# Scripts

Data processing and extraction scripts for the astrophysics papers dataset.

## Overview

This directory contains scripts for:
- **Abstract download**: Downloading abstracts from arXiv API
- **Citation extraction**: Extracting citation data from NASA ADS
- **Keyword extraction**: Extracting and curating ADS keywords

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

## Getting ADS API Key

1. Create an account at https://ui.adsabs.harvard.edu/
2. Go to Account → Settings → API Token
3. Copy your token to `~/.env` as `ADS=your_api_key_here`

All scripts are designed to be idempotent (safe to re-run) and automatically save progress for resumable execution.
