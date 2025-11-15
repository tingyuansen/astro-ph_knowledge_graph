#!/usr/bin/env python3
"""
Calculate 2025 reference co-occurrence matrix and concept ordering.

This script:
1. Calculates full-dataset co-occurrence matrix
2. Uses predefined concept classes as Level 1 domains
3. Performs spectral clustering WITHIN each class for Level 2 ordering
4. Saves hierarchical ordering and domain boundaries for use in other scripts

Output:
- concept_ordering_2025.npy - Ordered concept indices
- class_boundaries_2025.npy - Domain boundary positions
- domain_labels_2025.npy - Domain label for each concept
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering
import time

print("="*80)
print("CALCULATING 2025 REFERENCE ORDERING")
print("="*80)

start_time = time.time()

# Load data
print("\n1. Loading data...")
papers_concepts = pd.read_csv('../papers_concepts_mapping.csv.gz', dtype={'arxiv_id': str, 'label': int})
concepts = pd.read_csv('../concepts_vocabulary.csv.gz')
n_concepts = papers_concepts['label'].max() + 1
print(f"   ✓ {len(papers_concepts):,} paper-concept assignments")
print(f"   ✓ {n_concepts:,} concepts")

# Build co-occurrence matrix
print("\n2. Building co-occurrence matrix...")
unique_papers = papers_concepts['arxiv_id'].unique()
paper_to_idx = {paper: idx for idx, paper in enumerate(unique_papers)}
papers_concepts['paper_idx'] = papers_concepts['arxiv_id'].map(paper_to_idx)

X = csr_matrix(
    (np.ones(len(papers_concepts), dtype=np.int32),
     (papers_concepts['paper_idx'].values, papers_concepts['label'].values)),
    shape=(len(unique_papers), n_concepts),
    dtype=np.int32
)

cooccurrence = X.T @ X
print(f"   ✓ Co-occurrence matrix: {cooccurrence.shape}")

# Normalize
concept_counts = np.array(cooccurrence.diagonal(), dtype=np.float64).flatten()
coo = cooccurrence.tocoo()
normalized_values = np.zeros(len(coo.data), dtype=np.float32)

for idx, (i, j, count_ij) in enumerate(zip(coo.row, coo.col, coo.data)):
    if concept_counts[i] > 0 and concept_counts[j] > 0:
        normalized_values[idx] = count_ij / np.sqrt(concept_counts[i] * concept_counts[j])

normalized = csr_matrix(
    (normalized_values, (coo.row, coo.col)),
    shape=(n_concepts, n_concepts),
    dtype=np.float32
)

# Make symmetric
normalized_sym = normalized + normalized.T
print(f"   ✓ Normalized and symmetrized")

# Level 1: Use predefined concept classes (natural ordering)
print("\n3. Level 1: Using predefined concept classes...")
label_to_class = dict(zip(concepts['label'], concepts['class']))

# Natural ordering: cosmic → galactic → stellar → planetary → methodologies
natural_order = [
    'Cosmology & Nongalactic Physics',
    'Galaxy Physics',
    'High Energy Astrophysics',
    'Solar & Stellar Physics',
    'Earth & Planetary Science',
    'Numerical Simulation',
    'Instrumental Design',
    'Statistics & AI'
]
class_names = natural_order
class_to_id = {class_name: i for i, class_name in enumerate(class_names)}

labels_l1 = np.full(n_concepts, -1, dtype=int)
for label in range(n_concepts):
    if label in label_to_class:
        labels_l1[label] = class_to_id[label_to_class[label]]

print(f"   ✓ {len(class_names)} predefined classes (natural ordering):")
for class_id, class_name in enumerate(class_names):
    count = (labels_l1 == class_id).sum()
    print(f"      {class_id}. {class_name}: {count} concepts")

# Level 2: Spectral clustering within each class
print("\n4. Level 2: Spectral clustering within each class...")
THRESHOLD = 0.01  # Only cluster concepts with normalized co-occurrence > 0.01
GROUPING_FACTOR = 200
labels_l2 = np.full(n_concepts, -1, dtype=int)
cluster_counter = 0

for class_id, class_name in enumerate(class_names):
    class_concepts = [label for label, cls in label_to_class.items() if cls == class_name]
    n_in_class = len(class_concepts)
    
    if n_in_class < 30:
        # Too small, single cluster
        for label in class_concepts:
            labels_l2[label] = cluster_counter
        cluster_counter += 1
        print(f"   ✓ {class_name}: {n_in_class} concepts, 1 subcluster")
    else:
        # Spectral clustering within class
        n_subclusters = max(2, min(15, n_in_class // GROUPING_FACTOR))
        class_adj = normalized_sym[class_concepts, :][:, class_concepts].toarray()
        class_adj = (class_adj + class_adj.T) / 2
        np.fill_diagonal(class_adj, 0)
        
        # Apply threshold for clustering (only cluster well-connected concepts)
        class_adj[class_adj < THRESHOLD] = 0
        
        if class_adj.sum() > 0:
            clustering = SpectralClustering(
                n_clusters=n_subclusters,
                affinity='precomputed',
                assign_labels='discretize',
                random_state=42
            )
            class_labels_l2 = clustering.fit_predict(class_adj)
            for subcluster in range(n_subclusters):
                mask = class_labels_l2 == subcluster
                for label in np.array(class_concepts)[mask]:
                    labels_l2[label] = cluster_counter
                cluster_counter += 1
        else:
            for label in class_concepts:
                labels_l2[label] = cluster_counter
            cluster_counter += 1
        
        print(f"   ✓ {class_name}: {n_in_class} concepts, {n_subclusters} subclusters")

# Reorder matrix
print("\n5. Computing final ordering...")
sort_key = np.zeros((n_concepts, 2), dtype=int)
sort_key[:, 0] = labels_l1.copy()
sort_key[:, 1] = labels_l2.copy()
disconnected_mask = labels_l1 == -1
sort_key[disconnected_mask, 0] = 999999
sort_key[disconnected_mask, 1] = 999999

sort_idx = np.lexsort((sort_key[:, 1], sort_key[:, 0]))

# Filter to connected concepts only
labels_l1_sorted = labels_l1[sort_idx]
connected_in_sorted = labels_l1_sorted != -1
sort_idx_2025 = sort_idx[connected_in_sorted]
print(f"   ✓ Total ordered: {len(sort_idx_2025):,} connected concepts")

# Calculate class boundaries (Level 1 - domains)
labels_l1_reordered = labels_l1[sort_idx_2025]
class_boundaries = []
current_class = labels_l1_reordered[0]
for i, cls in enumerate(labels_l1_reordered):
    if cls != current_class:
        class_boundaries.append(i)
        current_class = cls
class_boundaries = np.array(class_boundaries)
print(f"   ✓ {len(class_boundaries)} domain boundaries")

# Calculate subcluster boundaries (Level 2)
labels_l2_reordered = labels_l2[sort_idx_2025]
subcluster_boundaries = []
current_subcluster = labels_l2_reordered[0]
for i, sub in enumerate(labels_l2_reordered):
    if sub != current_subcluster:
        subcluster_boundaries.append(i)
        current_subcluster = sub
subcluster_boundaries = np.array(subcluster_boundaries)
print(f"   ✓ {len(subcluster_boundaries)} subcluster boundaries")

# Save
print("\n5. Saving results...")
np.save('concept_ordering_2025.npy', sort_idx_2025)
np.save('class_boundaries_2025.npy', class_boundaries)
np.save('domain_labels_2025.npy', labels_l1)
np.save('subcluster_boundaries_2025.npy', subcluster_boundaries)
np.save('subcluster_labels_2025.npy', labels_l2)
print(f"   ✓ Saved concept_ordering_2025.npy (9,999 concepts)")
print(f"   ✓ Saved class_boundaries_2025.npy (7 domain boundaries)")
print(f"   ✓ Saved domain_labels_2025.npy")
print(f"   ✓ Saved subcluster_boundaries_2025.npy")
print(f"   ✓ Saved subcluster_labels_2025.npy")

print(f"\n⏱️  Total: {time.time() - start_time:.1f}s")
print("="*80)
print("✅ 2025 reference ordering calculated and saved!")
print("="*80)
