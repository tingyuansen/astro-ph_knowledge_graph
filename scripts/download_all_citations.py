"""
Download ALL citation data from ADS with full bibcodes.
No filtering - just get everything and store it.
We'll do arXiv matching as a separate post-processing step.
"""
import pandas as pd
import requests
import time
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment
env_path = Path.home() / '.env'
load_dotenv(env_path)

ADS_SEARCH_URL = "https://api.adsabs.harvard.edu/v1/search/query"
ADS_TOKEN = os.environ.get('ADS')

if not ADS_TOKEN:
    print("ERROR: Set ADS token in ~/.env")
    exit(1)

headers = {'Authorization': f'Bearer {ADS_TOKEN}'}

PROGRESS_FILE = 'citations_full_progress.json'
OUTPUT_FILE = 'citations_full_data.jsonl'

print("="*70)
print("DOWNLOAD ALL CITATIONS FROM ADS (FULL BIBCODE DATA)")
print("="*70)

# Load papers
print("\n1. Loading papers...")
mapping = pd.read_csv('papers_index_mapping.csv.gz', dtype={'arxiv_id': str})
arxiv_ids = mapping['arxiv_id'].tolist()
print(f"   Total papers: {len(arxiv_ids):,}")

# Load or initialize progress
processed_papers = set()
if os.path.exists(PROGRESS_FILE):
    print("   Loading previous progress...")
    with open(PROGRESS_FILE, 'r') as f:
        progress = json.load(f)
        processed_papers = set(progress.get('processed', []))
    print(f"   Already processed: {len(processed_papers):,} papers")

# Open output file in append mode
output_handle = open(OUTPUT_FILE, 'a')

def convert_to_ads_format(arxiv_id):
    """Convert arXiv ID to format ADS understands"""
    if '-' in arxiv_id and arxiv_id.startswith('astro-ph'):
        # Old format: astro-ph-0704007 -> astro-ph/0704007
        parts = arxiv_id.split('-', 2)  # Split on first 2 dashes only
        if len(parts) == 3:
            return f"{parts[0]}-{parts[1]}/{parts[2]}"
    return arxiv_id

print("\n2. Downloading citations (all data with bibcodes)...")
BATCH_SIZE = 50
DELAY = 0.6

# TEST MODE
TEST_MODE = False  # Set to False for full run
TEST_LIMIT = 3000 if TEST_MODE else len(arxiv_ids)
print(f"   {'⚠️  TEST MODE: Processing first 3,000 papers only' if TEST_MODE else '✅ Full mode: Processing all papers'}")

start_time = time.time()
papers_to_process = [aid for aid in arxiv_ids[:TEST_LIMIT] if aid not in processed_papers]
total_downloaded = len(processed_papers)

for batch_start in range(0, len(papers_to_process), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(papers_to_process))
    batch = papers_to_process[batch_start:batch_end]
    
    if batch_start % 500 == 0 and batch_start > 0:
        elapsed = time.time() - start_time
        rate = batch_start / elapsed if elapsed > 0 else 0
        remaining = len(papers_to_process) - batch_start
        eta_hours = (remaining / rate / 3600) if rate > 0 else 0
        
        print(f"   [{batch_start:,}/{len(papers_to_process):,}] ({batch_start/len(papers_to_process)*100:.1f}%) "
              f"| Downloaded: {total_downloaded:,} | Rate: {rate:.1f}/s | ETA: {eta_hours:.1f}h")
    
    # Build query for this batch
    ads_ids = [convert_to_ads_format(aid) for aid in batch]
    query_str = ' OR '.join([f'identifier:arXiv:{aid}' for aid in ads_ids])
    
    params = {
        'q': query_str,
        'fl': 'bibcode,identifier,reference,citation',
        'rows': BATCH_SIZE
    }
    
    try:
        response = requests.get(
            ADS_SEARCH_URL,
            headers=headers,
            params=params,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            docs = data['response']['docs']
            
            for doc in docs:
                # Extract arXiv ID from identifiers
                identifiers = doc.get('identifier', [])
                arxiv_id = None
                
                for ident in identifiers:
                    if 'arXiv:' in ident:
                        test_id = ident.replace('arXiv:', '')
                        # Check both formats
                        if test_id in batch:
                            arxiv_id = test_id
                            break
                        # Also check if original format matches
                        for orig_id in batch:
                            if convert_to_ads_format(orig_id) == test_id:
                                arxiv_id = orig_id
                                break
                        if arxiv_id:
                            break
                
                if not arxiv_id:
                    continue
                
                # Store EVERYTHING - bibcodes and all
                paper_data = {
                    'arxiv_id': arxiv_id,
                    'bibcode': doc.get('bibcode'),
                    'identifiers': identifiers,
                    'references': doc.get('reference', []),
                    'citations': doc.get('citation', []),
                    'num_references': len(doc.get('reference', [])),
                    'num_citations': len(doc.get('citation', []))
                }
                
                # Write to file
                output_handle.write(json.dumps(paper_data) + '\n')
                output_handle.flush()
                
                processed_papers.add(arxiv_id)
                total_downloaded += 1
        
        elif response.status_code == 429:
            print(f"      Rate limited, waiting 60s...")
            time.sleep(60)
            continue
        
    except Exception as e:
        print(f"      Error in batch {batch_start}: {e}")
    
    # Save progress every 500 papers
    if batch_start % 500 == 0:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({'processed': list(processed_papers)}, f)
    
    time.sleep(DELAY)

# Final save
output_handle.close()
with open(PROGRESS_FILE, 'w') as f:
    json.dump({'processed': list(processed_papers)}, f)

print(f"\n   Total papers downloaded: {total_downloaded:,}/{TEST_LIMIT:,}")
print(f"   Coverage: {total_downloaded/TEST_LIMIT*100:.1f}%")

# Quick statistics
print("\n3. Quick statistics...")
total_refs = 0
total_cites = 0
with_refs = 0
with_cites = 0

with open(OUTPUT_FILE, 'r') as f:
    for line in f:
        data = json.loads(line)
        nr = data['num_references']
        nc = data['num_citations']
        total_refs += nr
        total_cites += nc
        if nr > 0:
            with_refs += 1
        if nc > 0:
            with_cites += 1

print(f"   Papers with references: {with_refs:,} ({with_refs/total_downloaded*100:.1f}%)")
print(f"   Papers with citations: {with_cites:,} ({with_cites/total_downloaded*100:.1f}%)")
print(f"   Total references: {total_refs:,}")
print(f"   Total citations: {total_cites:,}")
print(f"   Avg refs/paper: {total_refs/total_downloaded:.1f}")
print(f"   Avg cites/paper: {total_cites/total_downloaded:.1f}")

total_time = time.time() - start_time
print(f"\n{'='*70}")
print(f"COMPLETED in {total_time/60:.1f} minutes")
print(f"{'='*70}")
print(f"\n✅ Full citation data saved to: {OUTPUT_FILE}")
print(f"   (All bibcodes preserved - can filter to arXiv later)")

