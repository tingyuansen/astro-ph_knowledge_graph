#!/usr/bin/env python3
"""
Generate temporal evolution video (scattered/detailed version) using fixed-size windows.

Requires: concept_ordering_2025.npy (run calculate_cooccurrence.py first)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.sparse import csr_matrix
import os
from tqdm import tqdm
import subprocess

# Professional fonts
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['font.weight'] = 'bold'

print("="*80)
print("TEMPORAL EVOLUTION OF CONCEPT CO-OCCURRENCE")
print("="*80)

os.makedirs('temporal_frames_scattered', exist_ok=True)

# Load pre-calculated 2025 ordering
print("\n1. Loading 2025 reference ordering...")
sort_idx_2025 = np.load('concept_ordering_2025.npy')
class_boundaries = np.load('class_boundaries_2025.npy')
domain_labels = np.load('domain_labels_2025.npy')
subcluster_boundaries = np.load('subcluster_boundaries_2025.npy')
subcluster_labels = np.load('subcluster_labels_2025.npy')
print(f"   ✓ {len(sort_idx_2025):,} concepts, {len(class_boundaries)} domains, {len(subcluster_boundaries)} subclusters")

# Load data
print("\n2. Loading data...")
papers_concepts = pd.read_csv('../papers_concepts_mapping.csv.gz', dtype={'arxiv_id': str, 'label': int})
papers_years = np.load('../papers_years.npy')
papers_index = pd.read_csv('../papers_index_mapping.csv.gz', dtype={'arxiv_id': str})

arxiv_to_idx = dict(zip(papers_index['arxiv_id'], papers_index['paper_idx']))
papers_concepts['paper_idx'] = papers_concepts['arxiv_id'].map(arxiv_to_idx)
papers_concepts['year'] = papers_concepts['paper_idx'].apply(
    lambda idx: papers_years[idx] if idx < len(papers_years) else 1991
)

# Extract month from arXiv IDs, combine with papers_years for (year, month)
def extract_month(arxiv_id):
    """Extract month from arXiv ID. Returns month (1-12) or 1 if unparseable."""
    try:
        # Format: YYMM.NNNNN or YYMMNNN (no prefix in CSV)
        if '.' in arxiv_id:
            yymm = arxiv_id.split('.')[0]
        else:
            yymm = arxiv_id[:4] if len(arxiv_id) >= 4 else None
        
        if yymm and len(yymm) >= 4 and yymm.isdigit():
            mm = int(yymm[2:4])
            if 1 <= mm <= 12:
                return mm
    except:
        pass
    return 1  # Fallback month

papers_concepts['month'] = papers_concepts['arxiv_id'].apply(extract_month)
papers_concepts['year_month'] = list(zip(papers_concepts['year'], papers_concepts['month']))
papers_concepts = papers_concepts.sort_values('year_month').reset_index(drop=True)

n_concepts = papers_concepts['label'].max() + 1
print(f"   ✓ {len(papers_concepts):,} assignments, {n_concepts:,} concepts")

# Domain colors (8 classes, natural ordering: cosmic → galactic → stellar → planetary → methods)
domain_colors = np.array([
    [0.80, 0.00, 0.00],  # Dark Red - Cosmology
    [0.00, 0.30, 0.70],  # Dark Blue - Galaxy
    [0.50, 0.00, 0.70],  # Dark Purple - High Energy
    [0.00, 0.50, 0.00],  # Dark Green - Solar & Stellar
    [0.70, 0.60, 0.00],  # Dark Olive - Earth & Planetary
    [0.70, 0.35, 0.00],  # Dark Orange - Numerical Simulation
    [0.70, 0.00, 0.50],  # Dark Magenta - Instrumental
    [0.00, 0.70, 0.70],  # Dark Cyan - Statistics & AI
])

# Domain names (natural ordering)
domain_names = [
    "Cosmology",
    "Galaxy",
    "High Energy",
    "Solar/Stellar",
    "Earth/Planetary",
    "Numerical",
    "Instrumental",
    "Statistics/AI"
]

# Generate frames
print("\n3. Generating frames...")
unique_papers_sorted = papers_concepts[['arxiv_id', 'year_month']].drop_duplicates('arxiv_id').sort_values('year_month')
n_papers_per_window = 40000
n_windows = len(unique_papers_sorted) // n_papers_per_window

block_size = 3
display_size = len(sort_idx_2025)
n_blocks = display_size // block_size
class_boundaries_binned = [b // block_size for b in class_boundaries]
labels_l1_reordered = domain_labels[sort_idx_2025]

for window_idx in tqdm(range(n_windows), desc="Generating frames"):
    start_idx = window_idx * n_papers_per_window
    end_idx = start_idx + n_papers_per_window
    window_papers = set(unique_papers_sorted.iloc[start_idx:end_idx]['arxiv_id'])
    
    start_year, start_month = unique_papers_sorted.iloc[start_idx]['year_month']
    end_year, end_month = unique_papers_sorted.iloc[end_idx-1]['year_month']
    
    papers_concepts_window = papers_concepts[papers_concepts['arxiv_id'].isin(window_papers)].copy()
    
    # Build co-occurrence for this window
    paper_id_map_window = {arxiv_id: i for i, arxiv_id in enumerate(papers_concepts_window['arxiv_id'].unique())}
    rows_window = papers_concepts_window['arxiv_id'].map(paper_id_map_window).values
    cols_window = papers_concepts_window['label'].values
    n_papers_window = len(paper_id_map_window)
    
    X_window = csr_matrix(
        (np.ones(len(papers_concepts_window), dtype=np.int32),
         (rows_window, cols_window)),
        shape=(n_papers_window, n_concepts),
        dtype=np.int32
    )
    cooccurrence_window = X_window.T @ X_window
    
    # Normalize
    concept_counts_window = np.array(cooccurrence_window.diagonal(), dtype=np.float64).flatten()
    coo_window = cooccurrence_window.tocoo()
    normalized_values_window = np.zeros(len(coo_window.data), dtype=np.float32)
    
    for idx, (i, j, count_ij) in enumerate(zip(coo_window.row, coo_window.col, coo_window.data)):
        if concept_counts_window[i] > 0 and concept_counts_window[j] > 0:
            normalized_values_window[idx] = count_ij / np.sqrt(concept_counts_window[i] * concept_counts_window[j])
    
    normalized_window = csr_matrix(
        (normalized_values_window, (coo_window.row, coo_window.col)),
        shape=(n_concepts, n_concepts),
        dtype=np.float32
    )
    
    # Apply 2025 ordering
    normalized_sym_window = normalized_window + normalized_window.T
    normalized_reordered = normalized_sym_window[sort_idx_2025, :][:, sort_idx_2025]
    normalized_display = normalized_reordered.toarray()
    
    # Set diagonal to 1.0 (concepts always co-occur with themselves)
    np.fill_diagonal(normalized_display, 1.0)
    
    # Bin with 3x3 aggregation
    binned_matrix = np.zeros((n_blocks, n_blocks))
    for i in range(n_blocks):
        for j in range(n_blocks):
            block = normalized_display[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            nonzero_in_block = block[block > 0]
            if len(nonzero_in_block) > 0:
                binned_matrix[i, j] = np.mean(nonzero_in_block)
    
    # Create visualization matrix (matching visualize_cooccurrence.py style)
    domain_matrix = np.ones((n_blocks, n_blocks, 4))
    domain_matrix[:, :, :3] = 1.0  # WHITE background
    domain_matrix[:, :, 3] = 1.0
    
    for i in range(n_blocks):
        for j in range(n_blocks):
            score = binned_matrix[i, j]
            if score > 0:
                # Get domain from middle of block
                mid_idx_i = i * block_size + block_size // 2
                mid_idx_j = j * block_size + block_size // 2
                domain_i = labels_l1_reordered[min(mid_idx_i, display_size-1)]
                domain_j = labels_l1_reordered[min(mid_idx_j, display_size-1)]
                
                # Use sqrt for intensity (matches visualize_cooccurrence.py)
                intensity = np.sqrt(score)
                
                if domain_i == domain_j and domain_i >= 0:
                    # Within-domain: domain color with unified scaling
                    base_color = domain_colors[domain_i]
                    domain_matrix[i, j, :3] = base_color * (0.5 + 0.5 * intensity)
                elif domain_i >= 0 and domain_j >= 0:
                    # Cross-domain: grey with unified scaling (same formula)
                    base_color = np.array([0.5, 0.5, 0.5])
                    domain_matrix[i, j, :3] = base_color * (0.5 + 0.5 * intensity)
    
    # Plot (use gaussian interpolation like visualize_cooccurrence.py)
    fig, ax = plt.subplots(figsize=(24, 20))
    ax.imshow(domain_matrix, aspect='auto', interpolation='gaussian', origin='upper')
    
    for boundary in class_boundaries_binned:
        ax.axvline(boundary, color='black', linewidth=3.5, alpha=0.9)
        ax.axhline(boundary, color='black', linewidth=3.5, alpha=0.9)
    
    # Draw subcluster boxes
    subcluster_boundaries_binned = subcluster_boundaries // block_size
    all_boundaries_binned = np.concatenate([[0], subcluster_boundaries_binned, [n_blocks]])
    
    # Get domain for each subcluster
    subcluster_domains = []
    for i in range(len(all_boundaries_binned) - 1):
        mid_idx = (all_boundaries_binned[i] + all_boundaries_binned[i + 1]) // 2
        mid_point_orig = mid_idx * block_size + block_size // 2
        if mid_point_orig < display_size:
            domain = labels_l1_reordered[mid_point_orig]
            subcluster_domains.append(domain)
        else:
            subcluster_domains.append(-1)
    
    for i in range(len(all_boundaries_binned) - 1):
        for j in range(len(all_boundaries_binned) - 1):
            x_start = all_boundaries_binned[i]
            x_end = all_boundaries_binned[i + 1]
            y_start = all_boundaries_binned[j]
            y_end = all_boundaries_binned[j + 1]
            
            domain_i = subcluster_domains[i]
            domain_j = subcluster_domains[j]
            
            # Determine box style based on whether within-domain or cross-domain
            if domain_i == domain_j and domain_i >= 0:
                # Within-domain: colored box
                color = domain_colors[domain_i]
                rect = Rectangle((x_start - 0.5, y_start - 0.5),
                                x_end - x_start, y_end - y_start,
                                linewidth=1.5, edgecolor=color, facecolor='none',
                                alpha=0.7, zorder=10)
            elif domain_i >= 0 and domain_j >= 0:
                # Cross-domain: gray box
                rect = Rectangle((x_start - 0.5, y_start - 0.5),
                                x_end - x_start, y_end - y_start,
                                linewidth=1.0, edgecolor='gray', facecolor='none',
                                alpha=0.5, zorder=10)
            else:
                continue
            
            ax.add_patch(rect)
    
    ax.set_title(f'Concept Co-occurrence: {start_year}-{start_month:02d} to {end_year}-{end_month:02d} — {n_papers_window:,} papers', 
                 fontsize=28, fontweight='bold', pad=30)
    
    date_label = f'{start_year}-{start_month:02d}\nto\n{end_year}-{end_month:02d}'
    ax.text(0.98, 0.98, date_label, transform=ax.transAxes,
            fontsize=48, fontweight='bold', ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=3, alpha=0.9))
    
    # Add domain labels (use shortened names like visualize_cooccurrence.py)
    class_tick_positions = []
    class_tick_labels_x = []
    class_tick_labels_y = []
    
    prev_boundary = 0
    for idx in range(len(class_boundaries_binned) + 1):
        boundary = class_boundaries_binned[idx] if idx < len(class_boundaries_binned) else n_blocks
        center = (prev_boundary + boundary) / 2
        mid_point_orig = int((prev_boundary + boundary) // 2) * block_size + block_size // 2
        
        if mid_point_orig < display_size:
            class_id = labels_l1_reordered[mid_point_orig]
            if class_id >= 0 and class_id < len(domain_names):
                class_tick_positions.append(center)
                # Shortened names for better display
                name = domain_names[class_id]
                class_tick_labels_x.append(name)
                class_tick_labels_y.append(name)
        
        prev_boundary = boundary
    
    ax.set_xticks(class_tick_positions)
    ax.set_xticklabels(class_tick_labels_x, rotation=45, ha='right')
    ax.set_yticks(class_tick_positions)
    ax.set_yticklabels(class_tick_labels_y)
    
    ax.set_xlabel('Concept Domain', fontsize=24, fontweight='bold')
    ax.set_ylabel('Concept Domain', fontsize=24, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'temporal_frames_scattered/frame_{window_idx:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()

# Create video
print("\n4. Creating video...")
subprocess.run([
    'ffmpeg', '-y', '-framerate', '2', '-i', 'temporal_frames_scattered/frame_%04d.png',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23',
    'concept_cooccurrence_evolution_40k_papers_scattered.mp4'
], check=True, capture_output=True)

print("="*80)
print("✅ Video generated: concept_cooccurrence_evolution_40k_papers_scattered.mp4")
print("="*80)
