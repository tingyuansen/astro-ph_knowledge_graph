# Astro-ph Knowledge Graph Dataset

This repository contains a knowledge graph representation of astrophysics papers from arXiv, with concept extraction, descriptions, embeddings, and a citation network.

## Dataset Overview

The dataset consists of **408,590 astrophysics papers (astro-ph)** from arXiv spanning 1992 to July 2025. Each paper has been analyzed to extract approximately 10 key concepts, which have been clustered into 9,999 unique concept classes with detailed descriptions and semantic embeddings. The citation network includes **21.3M reference relationships** and **16.8M citation relationships**, linking papers within our dataset and to external sources. The dataset also includes **traditional ADS keywords** (73% coverage) for comparison with our systematically extracted concepts.

### Key Visualizations

**Concept Extraction: Distribution and Organization**

Our dataset systematically extracts ~10 concepts per paper, clustering them into 9,999 unique concept classes. The left plot shows most papers have 8-12 concepts, while the right plot shows the number of papers that the concept appears in.

<p align="center">
  <img src="figures/concepts_per_paper_distribution.png" width="45%" />
  <img src="figures/concept_frequency_distribution.png" width="45%" />
</p>

**Top Concepts and Hierarchical Organization**

The most frequent concepts span diverse astrophysics subfields. Concepts are organized into 8 hierarchical domains (Cosmology, Galaxy Physics, High Energy, Solar/Stellar, Earth/Planetary, Numerical, Instrumental, Statistics & AI), enabling both broad-level browsing and fine-grained discovery.

<p align="center">
  <img src="figures/top_20_concepts.png" width="45%" />
  <img src="figures/concepts_by_category.png" width="45%" />
</p>

**Extracted Concepts vs. Traditional Keywords: Superior Distribution and Coverage**

Our systematic concept extraction provides a **more balanced frequency distribution** of concepts (shown in the right plot above, note the log scale). Unlike traditional ADS keywords that cluster at extremes—either overly general (appearing in thousands of papers) or overly specific (appearing in only a handful)—our concepts exhibit more uniform coverage across the frequency range. This reflects our clustering approach: concepts are designed to avoid both over-general and over-specific categories, making them more suitable for quantitative analysis and discovery.

Additionally, we achieve **100% coverage** (every paper has concepts) versus 73% for ADS keywords, with 9,999 concepts providing finer granularity than the 6,909 ADS keywords.

<p align="center">
  <img src="figures/keyword_frequency_distribution.png" width="60%" />
</p>
<p align="center">
  <em>Frequency distribution comparison: ADS keywords cluster at extremes (bimodal) while our extracted concepts show more balanced coverage across the frequency range (note: x-axis is log scale).</em>
</p>

**Citation Network and Temporal Coverage**

The citation network links papers through 21.3M references and 16.8M citations, with most papers having 10-50 references. The dataset spans 1992-2025, with steady growth reflecting the expansion of astrophysics research and arXiv adoption.

<p align="center">
  <img src="figures/citation_network_statistics.png" width="45%" />
  <img src="figures/arxiv_format_and_temporal.png" width="45%" />
</p>

📊 **For detailed analysis and interactive exploration**, see [`dataset_statistics.ipynb`](dataset_statistics.ipynb)

## Getting the Data

**Note**: This repository uses **Git LFS (Large File Storage)** for large files.

To clone this repository with all data files:
```bash
# Make sure you have Git LFS installed
git lfs install

# Clone the repository
git clone https://github.com/tingyuansen/astro-ph_knowledge_graph.git
cd astro-ph_knowledge_graph

# LFS files will be automatically downloaded
```

If you already cloned without LFS:
```bash
git lfs install
git lfs pull
```

**Note**: Gzipped CSV files can be read directly by pandas without decompression:
```python
import pandas as pd
df = pd.read_csv('file.csv.gz')
```

## Files

### Core Index Files

#### 1. `papers_index_mapping.csv.gz`
**Description**: Master mapping of arXiv IDs to integer indices. This is the foundation index used across all other files.

**Columns**:
- `paper_idx`: Integer index (0-408,589)
- `arxiv_id`: arXiv identifier (e.g., "0704.0007", "astro-ph-0612345")

**Date Range**: 1992 through July 2025  
**Note**: All other files use `paper_idx` to reference papers

#### 2. `papers_years.npy`
**Description**: Publication year for each arXiv paper, indexed by `paper_idx` for efficient access. Years are extracted from arXiv IDs and correctly handle both old and new ID formats.

**Format**: NumPy array of shape `(408590,)` with `dtype=int64`  
**Index**: `papers_years[i]` gives the year for paper with `paper_idx=i`  
**Year Range**: 1992-2025

**Example Usage**:
```python
import numpy as np
import pandas as pd

# Load years array
years = np.load('papers_years.npy')

# Get year for a specific paper
paper_idx = 12345
year = years[paper_idx]  # Direct array access - O(1)

# Count papers by year
unique_years, counts = np.unique(years, return_counts=True)
papers_by_year = pd.Series(counts, index=unique_years)
```

**Advantages**: Direct indexing without joins, memory efficient, fast access

#### 3. `arxiv_to_bibcode_mapping.csv.gz`
**Description**: Maps arXiv IDs to NASA ADS bibcodes, providing the connection between arXiv and the broader astronomical literature.

**Columns**:
- `paper_idx`: Integer index (matches papers_index_mapping)
- `arxiv_id`: arXiv identifier
- `primary_bibcode`: Primary ADS bibcode (e.g., "2007PhRvD..76d4016C")
- `all_bibcodes`: Semicolon-separated list of all bibcodes (arXiv + journal versions)

**Coverage**: 100% of papers have ADS bibcodes

**Usage**:
```python
import pandas as pd

# Load mapping
bibcodes = pd.read_csv('arxiv_to_bibcode_mapping.csv.gz', dtype={'arxiv_id': str})

# Get bibcode for a paper
paper_bibcode = bibcodes[bibcodes['arxiv_id'] == '0704.0007']['primary_bibcode'].iloc[0]
print(paper_bibcode)  # "2007PhRvD..76d4016C"
```

### Concept-Related Files

#### 4. `concepts_vocabulary.csv.gz`
**Description**: A vocabulary of all 9,999 unique concepts extracted across all papers.

**Columns**:
- `label`: Unique integer identifier for each concept (0-9,998)
- `class`: Research category (e.g., "Cosmology & Nongalactic Physics")
- `concept`: Name of the concept (e.g., "21 cm Brightness Temperature")
- `description`: Detailed description of the concept explaining its significance and applications

#### 5. `papers_concepts_mapping.csv.gz`
**Description**: Maps each arXiv paper to its associated concepts (approximately 10 concepts per paper).

**Columns**:
- `arxiv_id`: arXiv identifier
- `label`: Concept label matching the `label` in `concepts_vocabulary.csv`

**Usage**:
```python
import pandas as pd

# Load mappings
papers = pd.read_csv('papers_concepts_mapping.csv.gz', dtype={'arxiv_id': str})
concepts = pd.read_csv('concepts_vocabulary.csv.gz')

# Get concepts for a specific paper
arxiv_id = '0704.0007'
paper_concepts = papers[papers['arxiv_id'] == arxiv_id]
paper_concepts_full = paper_concepts.merge(concepts, on='label')
print(paper_concepts_full[['concept', 'description']])
```

#### 6. `concepts_embeddings.npz`
**Description**: Semantic embeddings for all 9,999 concepts using OpenAI's text-embedding-3-large model.

**Format**: Compressed NumPy archive  
**Shape**: (9,999, 3,072) - one row per concept, ordered by label 0-9,998  
**Data Type**: float32

**Usage**:
```python
import numpy as np

# Load embeddings
embeddings = np.load('concepts_embeddings.npz')['embeddings']

# Get embedding for concept 7572
concept_embedding = embeddings[7572]

# Compute cosine similarity
from numpy.linalg import norm
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

similarity = cosine_similarity(embeddings[0], embeddings[1])
```

### Citation Network Files

The citation network was extracted from NASA's Astrophysics Data System (ADS) API with complete coverage of all 408,590 papers. The network includes references (papers cited) and citations (papers citing), using a unified integer indexing system.

#### 7. `citations_indexed.jsonl.gz` ⭐ **Main Citation File**
**Description**: Complete citation network with integer indices for fast graph operations. Each line is a JSON object representing one paper's citation data.

**Format**: JSON Lines (one JSON object per line), gzip-compressed

**Fields**:
- `paper_idx`: Integer index (0-408,589, matches papers_index_mapping)
- `bibcode`: Primary ADS bibcode
- `all_bibcodes`: List of all ADS bibcodes (arXiv + journal versions)
- `references`: List of integer indices of papers/works this paper cites
- `citations`: List of integer indices of papers/works citing this paper
- `num_references`: Count of references
- `num_citations`: Count of citations

**Coverage**: 100% of papers

**Key Features**:
- Uses integer indices for memory efficiency and fast graph operations
- Indices 0-408,589: arXiv papers in our dataset
- Indices 408,590+: External papers (journals, books, conference papers)
- Internal network: 59.4% of references and 80.6% of citations are to papers within our dataset
- Total relationships: 21.3M references, 16.8M citations

**Usage**:
```python
import gzip
import json
import pandas as pd

# Load citation data
citations = []
with gzip.open('citations_indexed.jsonl.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        citations.append(json.loads(line))

# Get citations for a specific paper
paper_idx = 100
paper_data = citations[paper_idx]
print(f"Paper {paper_idx}:")
print(f"  Bibcode: {paper_data['bibcode']}")
print(f"  References: {len(paper_data['references'])}")
print(f"  Citations: {len(paper_data['citations'])}")

# Build directed citation graph (paper → references)
edges = []
for paper in citations:
    paper_idx = paper['paper_idx']
    for ref_idx in paper['references']:
        edges.append((paper_idx, ref_idx))

print(f"Total edges: {len(edges):,}")
```

#### 8. `identifier_mapping_all.csv.gz`
**Description**: Complete mapping of all identifiers (arXiv IDs + external bibcodes) to integer indices.

**Columns**:
- `identifier_idx`: Integer index
- `identifier`: arXiv ID or bibcode
- `type`: 'arxiv' or 'external_bibcode'

**Index Ranges**:
- 0-408,589: arXiv papers (matches papers_index_mapping)
- 408,590-1,668,203: External bibcodes (non-arXiv papers)

#### 9. `identifier_mapping_arxiv.csv.gz`
**Description**: Subset of identifier mapping for arXiv papers only.

**Use**: Convert paper indices 0-408,589 to arXiv IDs

#### 10. `identifier_mapping_external.csv.gz`
**Description**: Subset of identifier mapping for external bibcodes only.

**Use**: Convert reference/citation indices 408,590+ to bibcodes

**Complete Example: Link Citations to Concepts**:
```python
import gzip
import json
import pandas as pd

# Load all mappings
papers_idx = pd.read_csv('papers_index_mapping.csv.gz', dtype={'arxiv_id': str})
concepts_map = pd.read_csv('papers_concepts_mapping.csv.gz', dtype={'arxiv_id': str})
concepts = pd.read_csv('concepts_vocabulary.csv.gz')
identifier_map = pd.read_csv('identifier_mapping_all.csv.gz', dtype={'identifier': str})

# Load citations
with gzip.open('citations_indexed.jsonl.gz', 'rt') as f:
    citations = [json.loads(line) for line in f]

# Example: Find papers that cite paper 100 and are in our dataset
paper_100_cites = citations[100]['citations']
internal_cites = [c for c in paper_100_cites if c < 408590]  # Only papers in our dataset

print(f"Paper 100 has {len(internal_cites)} citations from papers in our dataset")

# Convert indices to arXiv IDs
citing_arxiv_ids = papers_idx[papers_idx['paper_idx'].isin(internal_cites)]['arxiv_id'].tolist()

# Get concepts for citing papers
citing_concepts = concepts_map[concepts_map['arxiv_id'].isin(citing_arxiv_ids)]
citing_concepts_full = citing_concepts.merge(concepts, on='label')

# Analyze: What concepts do citing papers have?
top_concepts = citing_concepts_full['concept'].value_counts().head(10)
print("\nTop concepts in citing papers:")
print(top_concepts)
```

### Paper Summaries

#### 11. `papers_summaries.jsonl.gz` ⚠️ **Not Included in Public Release**

**Note**: To ensure licensing is respected and to take a conservative approach regarding the distribution of AI-generated summaries of copyrighted scientific work, we have chosen not to publicly release the structured summaries dataset. While tools like Google NotebookLM make paper summarization routine, at this scale (400K+ papers), we prefer to err on the side of caution.

**Description**: Structured summaries for all papers were generated internally from the full text of papers using GPT-4o, GPT-4o-mini, o1-mini, and DeepSeek-v3. These summaries include fields for title & authors, background, motivation, methodology, main results, and other structured information.

**For researchers needing summaries**: If you need access to the structured summaries for your research, please contact ting.74@osu.edu with a brief description of your use case. See our paper for the prompts and methodology used.

**Note**: The `abstracts_all.jsonl.gz` file provides original arXiv abstracts with 100% coverage, which can serve as a starting point for summarization tasks, though our summaries were generated from full paper text for more detailed content extraction.

#### 12. `abstracts_all.jsonl.gz`
**Description**: Original abstracts for all papers downloaded from arXiv API.

**Format**: JSON Lines (one JSON object per line), gzip-compressed

**Fields**:
- `arxiv_id`: arXiv identifier
- `abstract`: Full abstract text from arXiv

**Coverage**: 408,590 abstracts (100% coverage)  
**Date Range**: Papers from 1992 through July 2025

**Usage**:
```python
import gzip
import json

# Read abstracts
abstracts = {}
with gzip.open('abstracts_all.jsonl.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        record = json.loads(line)
        abstracts[record['arxiv_id']] = record['abstract']

# Access a specific abstract
abstract = abstracts['0704.0007']
print(abstract)
```

**Note**: Embedding analysis comparing concepts vs. summaries is available in the `embedding_analysis/` directory

### ADS Keywords Files

Traditional ADS (NASA Astrophysics Data System) keywords provide a comparison baseline for our systematically extracted concepts. These manually-assigned keywords demonstrate the difference between author-provided metadata and systematic concept extraction.

#### 13. `ads_keywords_curated.csv.gz`
**Description**: Curated ADS keywords with cleaning and filtering applied.

**Columns**:
- `arxiv_id`: arXiv identifier (stored as string)
- `keyword`: Individual keyword (one row per keyword assignment)

**Size**: 1,269,903 keyword assignments across 298,658 papers  
**Unique Keywords**: 6,909  
**Coverage**: 73% of papers have ADS keywords

**Curation Process**:
1. Removed arXiv classification keywords (e.g., "astro-ph", "astro-ph.CO")
2. Normalized to lowercase
3. Filtered overly common keywords (>100K papers)
4. Filtered overly rare keywords (<5 papers)

**Statistics**:
- Median keywords per paper: 4
- Mean keywords per paper: 4.3
- Most common: "galaxies" (90,438 papers), "cosmology" (75,234 papers)

**Usage**:
```python
import pandas as pd

# Load curated ADS keywords
ads_keywords = pd.read_csv('ads_keywords_curated.csv.gz', dtype={'arxiv_id': str})

# How many keywords does a paper have?
paper_keywords = ads_keywords[ads_keywords['arxiv_id'] == '0704.0007']
print(f"Keywords: {paper_keywords['keyword'].tolist()}")

# Which papers have a specific keyword?
cosmology_papers = ads_keywords[ads_keywords['keyword'] == 'cosmology']
print(f"Papers with 'cosmology': {len(cosmology_papers)}")

# Keyword frequency distribution
keyword_counts = ads_keywords.groupby('keyword').size().sort_values(ascending=False)
print(keyword_counts.head(10))
```


## Data Generation Process

This dataset was created through a multi-stage pipeline:

1. **Paper Collection** (1992-2025)
   - Downloaded 408,590 astro-ph papers from arXiv
   - Covers all astronomy & astrophysics categories
   
2. **Concept Extraction**
   - Used GPT-4o and o1 models to extract ~10 key concepts per paper
   - Generated detailed descriptions for each concept
   - Clustered into 9,999 unique concept classes
   
3. **Embeddings Generation**
   - Generated semantic embeddings using OpenAI's text-embedding-3-large
   - 3,072-dimensional vectors for similarity search
   
4. **Citation Network Construction**
   - Downloaded complete citation data from NASA ADS API
   - Matched arXiv papers to ADS bibcodes
   - Created unified integer index (0-408,589 for arXiv papers, 408,590+ for external references)
   - Prioritized arXiv IDs in the network, keeping bibcodes for non-arXiv papers
   - Deduplication: Merged multiple bibcodes (preprint + journal) for same paper
   - Used integer indices for efficient graph operations
   
5. **Abstracts Download**
   - Downloaded all abstracts from arXiv API (public, no key required)
   - 408,590 abstracts (100% coverage, 99.95% have substantial content)
   - Rate-limited download with automatic checkpointing and resume capability

6. **Paper Summaries**
   - Generated structured summaries using GPT-4o, GPT-4o-mini, o1-mini, and DeepSeek-v3
   - Extracted key information: background, motivation, methodology, results
   - **Note**: Summaries not included in public release (see Paper Summaries section for rationale)

7. **ADS Keywords Extraction** (for comparison)
   - Extracted traditional ADS keywords from NASA ADS API
   - Curated and filtered keywords (removed overly common/rare, normalized)
   - 298,657 papers (73%) have ADS keywords
   - Provides baseline comparison for our extracted concepts
   
8. **Validation & Alignment**
   - Verified 100% coverage across all core files (concepts, citations, summaries, abstracts)
   - Ensured consistent index alignment between citation network, concepts, and summaries
   - All indices 0-408,589 consistently reference the same papers across all files

**Last Updated**: January 2025


## Quick Start

```python
import pandas as pd
import numpy as np
import gzip
import json

# Load the concept vocabulary
concepts = pd.read_csv('concepts_vocabulary.csv.gz')

# Load paper-concept mappings (⚠️ IMPORTANT: Always read arxiv_id as string!)
papers = pd.read_csv('papers_concepts_mapping.csv.gz', dtype={'arxiv_id': str})

# Load paper index mapping
papers_idx = pd.read_csv('papers_index_mapping.csv.gz', dtype={'arxiv_id': str})

# Load concept embeddings
embeddings = np.load('concepts_embeddings.npz')['embeddings']

# Load citation network
citations = []
with gzip.open('citations_indexed.jsonl.gz', 'rt') as f:
    for line in f:
        citations.append(json.loads(line))

# Find all concepts for a specific paper
arxiv_id = '0704.0007'
paper_idx = papers_idx[papers_idx['arxiv_id'] == arxiv_id]['paper_idx'].iloc[0]
paper_concepts = papers[papers['arxiv_id'] == arxiv_id]
paper_concepts_full = paper_concepts.merge(concepts, on='label')
print(paper_concepts_full[['concept', 'class', 'description']])

# Get citation data
paper_citation_data = citations[paper_idx]
print(f"\nReferences: {paper_citation_data['num_references']}")
print(f"Citations: {paper_citation_data['num_citations']}")
print(f"ADS Bibcode: {paper_citation_data['bibcode']}")
```

## Dataset Statistics

### Papers
- **Total papers**: 408,590
- **Date range**: 1992 - July 2025
- **Coverage**: 100% have concepts, citations, summaries, and abstracts

### Concepts
- **Unique concepts**: 9,999
- **Paper-concept associations**: ~3.8 million
- **Average concepts per paper**: ~10
- **Embedding dimensions**: 3,072 (text-embedding-3-large)

### Abstracts
- **Total abstracts**: 408,590 (100% coverage)
- **Quality**: 99.95% have substantial content (≥100 chars)
- **Source**: arXiv API (original abstracts)

### ADS Keywords (for comparison)
- **Papers with keywords**: 298,657 (73% coverage)
- **Unique keywords**: 6,909
- **Keyword assignments**: 1,269,903
- **Average keywords per paper**: 4.3

### Citation Network
- **Total unique identifiers**: 1,668,204
  - arXiv papers: 408,590 (indices 0-408,589)
  - External papers: 1,259,614 (indices 408,590+)
- **Total reference relationships**: 21,327,849
  - Internal (within dataset): 12,660,842 (59.4%)
  - External (to other papers): 8,667,007 (40.6%)
- **Total citation relationships**: 16,782,507
  - Internal (within dataset): 13,528,769 (80.6%)
  - External (from other papers): 3,253,738 (19.4%)
- **Average references per paper**: 52.2
- **Average citations per paper**: 41.1

## Index Alignment

All files use a consistent indexing system:

```
Index 0-408,589 (arXiv papers in our dataset):
  ├─ papers_index_mapping.csv.gz       paper_idx
  ├─ papers_concepts_mapping.csv.gz    paper_idx  
  ├─ citations_indexed.jsonl.gz        paper_idx
  ├─ identifier_mapping_arxiv.csv.gz   identifier_idx
  └─ arxiv_to_bibcode_mapping.csv.gz   paper_idx

Index 408,590-1,668,203 (External references):
  ├─ identifier_mapping_external.csv.gz
  └─ Referenced in: citations_indexed.jsonl.gz (references/citations fields)
```

This alignment enables integration across concepts, citations, and paper metadata.

## Scripts

The [`scripts/`](scripts/) directory contains utility scripts for data processing:

- **`download_all_citations.py`**: Downloads complete citation data from ADS API
- **`extract_all_ads_keywords.py`**: Extracts ADS keywords for all papers (requires ADS API key)
- **`curate_ads_keywords.py`**: Processes and curates raw ADS keywords
- See [`scripts/README.md`](scripts/README.md) for detailed usage instructions

## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{ting2025astromlabknowledgegraph,
  title={AstroMLab 5: Structured Summaries and Concept Extraction for 400,000 Astrophysics Papers},
  author={Ting, Yuan-Sen and Accomazzi, Alberto and Ghosal, Tirthankar and Nguyen, Tuan Dung and Pan, Rui and Sun, Zechang and de Haan, Tijmen},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

**Note**: The arXiv ID will be updated once the paper is posted to arXiv.

## Notes

- **Always read `arxiv_id` as string**: Use `dtype={'arxiv_id': str}` to prevent pandas from converting IDs to floats
- **Index alignment**: All paper_idx values 0-408,589 consistently reference the same papers across all files
- **File formats**: All large CSV files are gzip-compressed but can be read directly by pandas
- **Citation indices**: Indices < 408,590 reference papers in our dataset; indices ≥ 408,590 reference external papers
- **Bibcode mapping**: Use `arxiv_to_bibcode_mapping.csv.gz` to convert between arXiv IDs and ADS bibcodes
- **ADS Keywords**: 73% coverage (298,657 papers); see `dataset_statistics.ipynb` for comparison with our extracted concepts and temporal analysis
- **Git LFS**: Large files are stored with Git LFS - ensure it's installed before cloning

## License

This dataset aggregates data from multiple sources:
- arXiv papers: arXiv.org non-exclusive license
- Citation data: NASA Astrophysics Data System (ADS)
- Concept extraction: Generated using AI models (included in this release)
- Paper summaries: Generated using AI models (**not included in public release**)

Please respect the original licenses and terms of use of the underlying data sources.

### Conservative Approach to Summaries

Out of an abundance of caution regarding intellectual property and licensing, we have chosen not to publicly release the AI-generated structured summaries. While summarization of scientific papers is now routine (e.g., Google NotebookLM), distributing summaries of 400K+ copyrighted works at scale requires careful consideration. We believe this conservative approach respects the original authors and publishers while still enabling the research community to benefit from our concept extraction, embeddings, and citation network.


