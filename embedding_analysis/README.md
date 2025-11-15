# Embedding Analysis

Embedding-based analysis demonstrating that concepts and summaries serve distinct, complementary functions in the knowledge graph.

## Overview

This analysis uses a sample of 10,000 papers to compare semantic similarities between:
- **Concepts**: AI-generated topical labels for each paper
- **Summary sections**: Six structured sections (background, motivation, methodology, results, interpretation, implication)

**Key Finding**: Concepts are dispersed (median similarity ~0.35), enabling multi-faceted discovery across the literature, while summaries cluster (median similarity ~0.88), providing coherent narratives for comprehension. This validates a two-stage workflow: (1) Discovery using dispersed concepts, (2) Comprehension using clustered summaries.

## Files

**Embeddings** (OpenAI text-embedding-3-large, 3072-dim):
- `abstract_embeddings.npz` - 10,000 paper abstracts
- `section_embeddings.npz` - 6 sections × 10,000 papers (background, motivation, methodology, results, interpretation, implication)

**Analysis**:
- `embedding_analysis.ipynb` - Generates figures and computes statistics
- `extract_all_embeddings.py` - Parallel extraction script for OpenAI API

## Quick Start

Run the analysis notebook:

```bash
cd embedding_analysis/
jupyter notebook embedding_analysis.ipynb
```

**Output**:
- UMAP visualizations showing concepts dispersed vs sections clustered
- Distribution plots of cumulative similarity distributions
- Statistics including median similarities and percentiles

## Data Source

- **Abstracts**: Sampled from `../abstracts_all.jsonl.gz`
- **Summaries**: Sampled from `../papers_summaries.jsonl.gz`
- **Concepts**: From `../papers_concepts_mapping.csv.gz` and `../concepts_vocabulary.csv.gz`

This analysis uses a 10,000 paper sample for computational efficiency. The full dataset (408,590 papers) is available in the parent directory.
