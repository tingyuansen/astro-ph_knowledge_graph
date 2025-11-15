# Concept Co-occurrence Analysis

Analysis of how astrophysics concepts appear together in papers using hierarchical clustering and sparse matrix operations.

## Overview

This analysis reveals the structure and evolution of concept relationships across the astrophysics literature:
- **Hierarchical clustering**: Organizes 9,999 concepts into 8 fixed domains and 47 fixed subclusters (from 2025 reference)
- **Temporal evolution**: Tracks co-occurrence patterns across 40,000-paper windows using 47×47 grid (2,209 subcluster pairs)
- **Sparse matrix operations**: Efficient computation of co-occurrence for 10,000 × 10,000 concept pairs

## Files

**Analysis scripts**:
- `calculate_cooccurrence.py` - Calculate 2025 reference ordering using hierarchical clustering
- `generate_temporal_evolution_scattered.py` - Generate detailed temporal evolution video (3×3 binning)
- `generate_temporal_evolution_grid.py` - Generate aggregated temporal evolution video (47×47 blocks)

**Reference ordering** (from 2025):
- `concept_ordering_2025.npy` - Ordered concept indices (9,999 concepts)
- `class_boundaries_2025.npy` - Domain boundaries (7 boundaries for 8 domains)
- `domain_labels_2025.npy` - Domain labels for each concept
- `subcluster_boundaries_2025.npy` - Subcluster boundaries (46 boundaries for 47 subclusters)
- `subcluster_labels_2025.npy` - Subcluster labels for each concept

**Scattered version** (detailed):
- `concept_cooccurrence_evolution_40k_papers_scattered.mp4` - Video with 3×3 binning (10 frames)
- `temporal_frames_scattered/` - Individual frame PNGs

**Grid version** (aggregated):
- `concept_cooccurrence_evolution_40k_papers_grid.mp4` - Video with 47×47 grid (2,209 blocks, 10 frames)
- `temporal_frames_grid/` - Individual frame PNGs

## Quick Start

**Step 1**: Calculate 2025 reference ordering

```bash
python calculate_cooccurrence.py
```

**Step 2**: Generate temporal evolution video (choose one or both versions)

Scattered version (detailed, 3×3 binning):
```bash
python generate_temporal_evolution_scattered.py
```

Grid version (aggregated, 47×47 grid = 2,209 blocks with dynamic transparency):
```bash
python generate_temporal_evolution_grid.py
```

## Mathematical Details

### 1. Co-occurrence Matrix Calculation

Given paper-concept matrix **X** ∈ {0,1}^(n×m) where:
- n = number of papers
- m = 10,000 concepts
- X[i,j] = 1 if paper i contains concept j, else 0

**Raw co-occurrence matrix**:
```
C = X^T @ X
```

where C[i,j] = number of papers containing both concepts i and j.

**Implementation**: Uses scipy.sparse.csr_matrix for efficient storage (sparsity ~99.9%) and sparse matrix multiplication.

### 2. Normalization (Ochiai Coefficient)

**Formula**:
```python
normalized[i,j] = C[i,j] / sqrt(C[i,i] × C[j,j])
```

where:
- C[i,i] = total papers containing concept i
- C[j,j] = total papers containing concept j
- C[i,j] = papers containing both concepts

**Properties**:
- Range: [0, 1]
- 1.0 = perfect co-occurrence (always appear together)
- 0.0 = never co-occur
- Symmetric: normalized[i,j] = normalized[j,i]
- Invariant to concept frequency (geometric mean normalization)

**Numerical handling**:
```python
diagonal = np.sqrt(co_matrix.diagonal())
normalized = co_matrix / np.outer(diagonal, diagonal)
normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
```

Ensures division by zero → 0.0 (concepts that never appear).

## Hierarchical Clustering

### Level 1: Predefined Classes (8 domains)

Natural ordering from concept vocabulary:

1. **Cosmology & Nongalactic Physics** (2,192 concepts)
2. **Galaxy Physics** (1,267 concepts)
3. **High Energy Astrophysics** (1,606 concepts)
4. **Solar & Stellar Physics** (930 concepts)
5. **Earth & Planetary Science** (639 concepts)
6. **Numerical Simulation** (1,050 concepts)
7. **Instrumental Design** (1,295 concepts)
8. **Statistics & AI** (1,020 concepts)

### Level 2: Spectral Clustering Within Each Class

For each domain independently:

1. **Extract submatrix** of normalized co-occurrence scores for concepts in that domain

2. **Apply threshold = 0.01**:
   ```python
   affinity_matrix = normalized_scores.copy()
   affinity_matrix[affinity_matrix < 0.01] = 0.0
   ```
   Filters weak connections to reveal strong community structure.

3. **Determine number of subclusters**:
   ```python
   n_clusters = max(2, min(15, n_concepts // 200))
   ```
   - Minimum 2 subclusters per domain
   - Maximum 15 subclusters per domain
   - Target ~200 concepts per subcluster (GROUPING_FACTOR)

4. **Spectral clustering**:
   ```python
   from sklearn.cluster import SpectralClustering
   clustering = SpectralClustering(
       n_clusters=n_clusters,
       affinity='precomputed',
       assign_labels='kmeans'
   )
   labels = clustering.fit_predict(affinity_matrix)
   ```
   - Uses thresholded co-occurrence as precomputed affinity
   - Solves eigenvalue problem to find communities
   - K-means on eigenvectors for final assignment

5. **Sort within subclusters** by concept frequency (total paper count):
   ```python
   concept_popularity = np.array(co_matrix.sum(axis=1)).flatten()
   # Sort concepts within each subcluster by descending popularity
   ```

**Result**: 47 fixed subclusters calculated from 2025 reference data, saved and reused for all temporal frames.

## Temporal Evolution

**Method**: Fixed-size sliding windows of 40,000 **unique papers** each (chronologically ordered).

**Why?** Removes field growth bias — later years don't appear "stronger" just because there are more papers. Fixed window size ensures comparable statistics across time.

### Window Construction

1. **Sort papers** by (year, month):
   ```python
   papers_sorted = papers_concepts.sort_values(['year', 'month'])
   unique_papers = papers_sorted['paper_id'].unique()
   ```

2. **Create windows**:
   ```python
   n_papers_per_window = 40000
   n_windows = len(unique_papers) // n_papers_per_window
   
   for window_idx in range(n_windows):
       start_idx = window_idx * n_papers_per_window
       end_idx = start_idx + n_papers_per_window
       window_papers = unique_papers[start_idx:end_idx]
   ```

3. **Extract window data**:
   ```python
   window_data = papers_concepts[papers_concepts['paper_id'].isin(window_papers)]
   ```

**Important**: Uses unique papers (not rows), as each paper can have multiple concepts.

### Common Pipeline (both versions)

1. Filter to 40,000 unique papers for this window
2. Build sparse matrix X (40,000 papers × 10,000 concepts)
3. Calculate C = X^T @ X (sparse matrix multiplication)
4. Normalize using Ochiai coefficient
5. Apply 2025 reference ordering (**fixed** 8 domains + 47 subclusters for consistent comparison)

### Scattered Version (Detailed)

6. **Bin to 3×3 blocks** (~3,333×3,333 heatmap)
7. Draw hierarchical boxes (8 domain boundaries + 47×47 subcluster grid)
8. Visualize with domain colors using sqrt(score) for intensity

Shows fine-grained co-occurrence patterns with detailed texture.

### Grid Version (Aggregated)

6. **Aggregate into 47×47 grid (2,209 subcluster-pair blocks)**
   - Each block represents co-occurrence between two subclusters
   - **Color**: 10th percentile of each block (captures signal)
   - **Alpha**: Based on spread (30th - 10th percentile) for consistency measure
   - Small spread = solid (consistent signal), large spread = transparent (variable signal)
7. Draw hierarchical boxes (8 domain boundaries visible)

Shows clean 47×47 block structure with dynamic transparency revealing pattern reliability.

**Key insight**: Fixed hierarchical structure allows direct comparison of co-occurrence patterns across time periods!

## Visualization Mathematics

### Scattered Version: Binning and Color Mapping

**Step 1: Reorder matrix** using 2025 reference ordering (9,999 concepts):
```python
display_matrix = normalized_scores[concept_ordering_2025, :][:, concept_ordering_2025]
```

**Step 2: Bin to 3×3 blocks** to reduce from 9,999×9,999 to ~3,333×3,333:
```python
block_size = 3
n_blocks = len(concept_ordering_2025) // block_size
binned_matrix = np.zeros((n_blocks, n_blocks))

for i in range(n_blocks):
    for j in range(n_blocks):
        block = display_matrix[i*3:(i+1)*3, j*3:(j+1)*3]
        non_zero = block[block > 0]
        if len(non_zero) > 0:
            binned_matrix[i, j] = np.mean(non_zero)
```
Takes **mean of non-zero values only** within each 3×3 block to avoid dilution from sparse regions.

**Step 3: Map scores to RGB colors**:

Initialize background as **white** (no co-occurrence):
```python
domain_matrix = np.ones((n_blocks, n_blocks, 4))
domain_matrix[:, :, :3] = 1.0  # WHITE background
```

For each binned cell (i, j) **with score > 0**:
1. Identify domain membership:
   ```python
   domain_i = domain_labels_2025[concept_ordering_2025[i * block_size + block_size // 2]]
   domain_j = domain_labels_2025[concept_ordering_2025[j * block_size + block_size // 2]]
   ```

2. Apply **square root scaling** for perceptual uniformity:
   ```python
   intensity = sqrt(binned_matrix[i, j])
   ```
   Compresses dynamic range, emphasizes weak signals.

3. **Apply color scaling**:
   ```python
   intensity = sqrt(binned_matrix[i, j])
   
   if domain_i == domain_j:
       # Within-domain: use domain-specific color
       base_color = domain_color[domain_i]
   else:
       # Cross-domain: use grey
       base_color = [0.5, 0.5, 0.5]
   
   # Same scaling formula for both cases
   color = base_color * (0.5 + 0.5 * intensity)
   ```
   
   **Within-domain examples** (colored):
   - Cosmology [0.8, 0.0, 0.0]: scales from [0.4, 0.0, 0.0] to [0.8, 0.0, 0.0]
   - Statistics/AI [0.0, 0.7, 0.7]: scales from [0.0, 0.35, 0.35] to [0.0, 0.7, 0.7]
   
   **Cross-domain** (grey):
   - Base color [0.5, 0.5, 0.5] scales from [0.25, 0.25, 0.25] to [0.5, 0.5, 0.5]
   - Grey values distinguish from colored within-domain blocks
   
   **No co-occurrence** (score = 0):
   - Remains white background [1.0, 1.0, 1.0]
   - Common in sparse cross-domain regions

**Step 4: Render** with Gaussian interpolation for smoothness.

### Grid Version: Percentile-Based Aggregation

**Much more complex** - aggregates detailed 3×3 binned RGB image into 47×47 subcluster grid.

**Step 1-3**: Same as scattered version (reorder, bin to 3×3, map to RGB colors).

Result: `domain_matrix` with shape (~3,333, ~3,333, 4) containing RGBA values.

**Step 4: Aggregate into 47×47 grid by subcluster boundaries**:

For each subcluster pair (i, j) in 0..46:
1. **Identify pixel region** in binned space:
   ```python
   subcluster_boundaries_binned = subcluster_boundaries_2025 // block_size
   all_boundaries = [0] + list(subcluster_boundaries_binned) + [n_blocks]
   
   y_start = all_boundaries[i]
   y_end = all_boundaries[i + 1]
   x_start = all_boundaries[j]
   x_end = all_boundaries[j + 1]
   
   box_region_rgb = domain_matrix[y_start:y_end, x_start:x_end, :3]
   ```
   Each box contains multiple pixels from the binned 3×3 image.

2. **Calculate 10th percentile for color** (captures signal, robust to outliers):
   ```python
   p10_color = np.percentile(box_region_rgb.reshape(-1, 3), 10, axis=0)
   ```
   - Shape: (3,) representing [R, G, B]
   - Lower RGB = darker (stronger signal in our color scheme)
   - 10th percentile selects darker pixels = higher co-occurrence

3. **Calculate 30th percentile for spread estimation**:
   ```python
   p30_color = np.percentile(box_region_rgb.reshape(-1, 3), 30, axis=0)
   spread = np.mean(p30_color - p10_color)
   ```
   - Measures variability within box
   - Small spread = consistent signal
   - Large spread = heterogeneous patterns

4. **Calculate dynamic alpha** (transparency) based on spread:
   ```python
   alpha = 1.0 - np.clip(spread * 3.0, 0, 1)
   alpha = np.clip(alpha, 0.3, 1.0)
   ```
   - spread ≈ 0 → alpha = 1.0 (fully opaque, consistent signal)
   - spread ≈ 0.33 → alpha = 0.3 (transparent, variable signal)
   - Minimum alpha = 0.3 to keep boxes visible

5. **Fill entire box** with aggregated color:
   ```python
   domain_matrix[y_start:y_end, x_start:x_end, :3] = p10_color
   domain_matrix[y_start:y_end, x_start:x_end, 3] = alpha
   ```

**Step 5: Render** with Gaussian interpolation.

**Key difference from scattered**: Each of 2,209 boxes (47×47 grid) shows **aggregated statistics** rather than raw binned pixels, with **dynamic transparency** revealing pattern consistency.

### Hierarchical Boxes Overlay

Drawn **on top** of heatmap with high zorder:

- **Domain boundaries** (8 domains):
  ```python
  linewidth=3.5, edgecolor='black', facecolor='none', zorder=10
  ```

- **Within-domain subcluster boxes** (47 total, distributed across domains):
  ```python
  linewidth=1.5, edgecolor=domain_color, alpha=0.7, facecolor='none', zorder=9
  ```

- **Cross-domain subcluster boxes**:
  ```python
  linewidth=1.0, edgecolor='grey', alpha=0.5, facecolor='none', zorder=9
  ```

### Color Palette

Domain-specific colors (dark shades for high contrast):
```python
domain_colors = [
    [0.80, 0.00, 0.00],  # Dark Red - Cosmology
    [0.00, 0.30, 0.70],  # Dark Blue - Galaxy
    [0.50, 0.00, 0.70],  # Dark Purple - High Energy
    [0.00, 0.50, 0.00],  # Dark Green - Solar & Stellar
    [0.70, 0.60, 0.00],  # Dark Olive - Earth & Planetary
    [0.70, 0.35, 0.00],  # Dark Orange - Numerical Simulation
    [0.70, 0.00, 0.50],  # Dark Magenta - Instrumental
    [0.00, 0.70, 0.70],  # Dark Cyan - Statistics & AI
]
```

## Approximations and Design Decisions

### 1. Binning to 3×3 Blocks

**Why?** Reduce 9,999×9,999 matrix to ~3,333×3,333 for visualization.

**Approximation**: Takes mean of non-zero values in each 3×3 block, ignoring zeros to avoid dilution.

**Impact**: 
- Smooths fine-grained patterns
- Reduces file size and rendering time
- Still preserves ~11 million data points per frame

### 2. Square Root Intensity Scaling

**Why?** Human perception of brightness is nonlinear.

**Formula**: `intensity = sqrt(score)` where score ∈ [0, 1]

**Impact**:
- Compresses dynamic range
- Emphasizes weak signals (0.01 → 0.1, 10x brighter)
- De-emphasizes strong signals (0.81 → 0.9, 1.1x brighter)

### 3. Percentile Aggregation (Grid Version Only)

**Why?** Reduce visual noise while preserving signal in sparse 47×47 grid.

**10th percentile for color**: Selects darker pixels (higher co-occurrence) within each box.
- Robust to outliers (white background pixels)
- Captures strong signal
- Lower percentile = more aggressive filtering

**30th - 10th percentile for alpha**: Measures consistency.
- Small spread → homogeneous → opaque
- Large spread → heterogeneous → transparent

**Impact**: Grid version is **not** a direct average but a **robust signal estimator** with consistency weighting.

### 4. Color Scaling

Applied **only when score > 0** (cells with zero co-occurrence remain white background).

**Formula**:
```python
color = base_color * (0.5 + 0.5 * intensity)
```
where `intensity = sqrt(score)` and `base_color` depends on domain relationship:
- **Within-domain**: `base_color = domain_color` (e.g., [0.8, 0.0, 0.0] for Cosmology)
- **Cross-domain**: `base_color = [0.5, 0.5, 0.5]` (grey)

**Scaling behavior**:
- When score → 0+: color = 50% of base_color (faint but visible)
- When score = 1.0: color = 100% of base_color (full intensity)
- When score = 0: color = white [1.0, 1.0, 1.0] (no signal)

**Resulting color ranges**:
- Cosmology within-domain: [0.4, 0.0, 0.0] to [0.8, 0.0, 0.0] (dark red to red)
- Cross-domain: [0.25, 0.25, 0.25] to [0.5, 0.5, 0.5] (dark grey to medium grey)

**Impact**: 
- Empty boxes (score = 0) are white in both within-domain and cross-domain regions
- Cross-domain grey provides clear visual distinction from colored within-domain blocks
- Same scaling formula simplifies interpretation across all regions

### 5. Fixed 2025 Reference Clustering

**Why?** Temporal consistency.

**Approximation**: Uses 2025 data to define clusters, applies to all historical windows.

**Impact**:
- Historical concept groupings may not reflect contemporaneous understanding
- Enables direct comparison across time (same concepts always in same position)
- Trade-off: Temporal consistency > Historical accuracy of clustering

### 6. Non-overlapping Windows

**Method**: Each window contains 40,000 **disjoint** unique papers.

**Impact**:
- No smoothing between adjacent time periods
- Discrete jumps possible between frames
- Alternative (not used): Sliding windows with overlap would smooth transitions but lose independence

## Implementation Notes

### Sparse Matrices

- Full matrix: 10,000 × 10,000 = 100M entries 
- Actual non-zeros: ~10M entries (~10% sparsity)
- scipy.sparse.csr_matrix for efficient storage and operations
- Matrix multiplication X^T @ X computed with sparse routines

### Pre-calculated Ordering

- Spectral clustering is expensive (~30 seconds for 10K concepts)
- Must be identical across all frames for comparison
- Calculate once from 2025 data, save as .npy, reuse for all windows
- Files: `concept_ordering_2025.npy`, `class_boundaries_2025.npy`, `domain_labels_2025.npy`, `subcluster_boundaries_2025.npy`, `subcluster_labels_2025.npy`

### On-the-fly Co-occurrence

- Co-occurrence calculation fast (~2 seconds per window)
- Each window uses different papers
- No memory overhead from storing intermediate matrices
- Recomputes from scratch for each frame

## Data Properties

- **Papers**: ~400,000
- **Concepts**: 10,000 (9,999 connected)
- **Assignments**: 3,827,232 paper-concept pairs
