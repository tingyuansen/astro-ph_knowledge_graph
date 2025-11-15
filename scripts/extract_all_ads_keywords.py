"""
Extract ALL ADS keywords from scratch using batched requests.
Optimized for speed with the working API key.
"""
import pandas as pd
import requests
import time
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Load environment
env_path = Path.home() / '.env'
load_dotenv(env_path)

ADS_SEARCH_URL = "https://api.adsabs.harvard.edu/v1/search/query"
ADS_TOKEN = os.environ.get('ADS')

if not ADS_TOKEN:
    print("ERROR: Set ADS token in ~/.env")
    exit(1)

headers = {'Authorization': f'Bearer {ADS_TOKEN}'}

# Configuration
BASE_DIR = Path(__file__).parent.parent
PROGRESS_FILE = BASE_DIR / 'ads_keywords_progress.json'
OUTPUT_FILE = BASE_DIR / 'ads_keywords_all.jsonl'
NUM_WORKERS = 10  # Parallel workers
BATCH_SIZE = 50   # Papers per API request
DELAY = 0.6       # Delay between requests

# Thread-safe
write_lock = Lock()
progress_lock = Lock()
total_downloaded = 0
total_processed = 0
processed_papers = set()

print("="*70)
print("EXTRACT ALL ADS KEYWORDS (FRESH START)")
print("="*70)

# Load papers
print("\n1. Loading papers...")
mapping = pd.read_csv(BASE_DIR / 'papers_index_mapping.csv.gz', dtype={'arxiv_id': str})
arxiv_ids = mapping['arxiv_id'].tolist()
print(f"   Total papers: {len(arxiv_ids):,}")

# Load progress if exists
if PROGRESS_FILE.exists():
    print("   Found previous progress...")
    with open(PROGRESS_FILE, 'r') as f:
        progress = json.load(f)
        processed_papers = set(progress.get('processed', []))
    print(f"   Already processed: {len(processed_papers):,} papers")

papers_to_process = [aid for aid in arxiv_ids if aid not in processed_papers]
print(f"   Remaining to process: {len(papers_to_process):,} papers")

def convert_to_ads_format(arxiv_id):
    if '-' in arxiv_id and arxiv_id.startswith('astro-ph'):
        parts = arxiv_id.split('-', 2)
        if len(parts) == 3:
            return f"{parts[0]}-{parts[1]}/{parts[2]}"
    return arxiv_id

def save_progress():
    with progress_lock:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({'processed': list(processed_papers)}, f)

def write_result(data):
    global total_downloaded
    with write_lock:
        with open(OUTPUT_FILE, 'a') as f:
            f.write(json.dumps(data) + '\n')
        total_downloaded += 1

def fetch_keywords_batch(batch_info):
    global total_processed
    
    batch_idx, batch = batch_info
    ads_ids = [convert_to_ads_format(aid) for aid in batch]
    query_str = ' OR '.join([f'identifier:arXiv:{aid}' for aid in ads_ids])
    
    params = {
        'q': query_str,
        'fl': 'bibcode,identifier,keyword',
        'rows': BATCH_SIZE
    }
    
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            response = requests.get(ADS_SEARCH_URL, headers=headers, params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                docs = data['response']['docs']
                
                batch_results = []
                for doc in docs:
                    identifiers = doc.get('identifier', [])
                    arxiv_id = None
                    
                    for ident in identifiers:
                        if 'arXiv:' in ident:
                            test_id = ident.replace('arXiv:', '')
                            if test_id in batch:
                                arxiv_id = test_id
                                break
                            for orig_id in batch:
                                if convert_to_ads_format(orig_id) == test_id:
                                    arxiv_id = orig_id
                                    break
                            if arxiv_id:
                                break
                    
                    if not arxiv_id:
                        continue
                    
                    keywords = doc.get('keyword', [])
                    result = {
                        'arxiv_id': arxiv_id,
                        'bibcode': doc.get('bibcode'),
                        'keywords': keywords,
                        'num_keywords': len(keywords)
                    }
                    
                    batch_results.append((arxiv_id, result))
                
                for arxiv_id, result in batch_results:
                    write_result(result)
                    with progress_lock:
                        processed_papers.add(arxiv_id)
                
                with progress_lock:
                    for arxiv_id in batch:
                        processed_papers.add(arxiv_id)
                    total_processed += len(batch)
                
                time.sleep(DELAY)
                return len(batch_results), None
            
            elif response.status_code == 429:
                print(f"\n   ⚠️  RATE LIMIT hit on batch {batch_idx}. Waiting 10s...", flush=True)
                time.sleep(10)
                retry_count += 1
                continue
            
            else:
                with progress_lock:
                    for arxiv_id in batch:
                        processed_papers.add(arxiv_id)
                    total_processed += len(batch)
                return 0, f"HTTP {response.status_code}"
        
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                with progress_lock:
                    for arxiv_id in batch:
                        processed_papers.add(arxiv_id)
                    total_processed += len(batch)
                return 0, f"Error: {str(e)}"
            time.sleep(5)
    
    return 0, "Max retries exceeded"

# Create batches
batches = [(i, papers_to_process[i:i+BATCH_SIZE]) for i in range(0, len(papers_to_process), BATCH_SIZE)]
print(f"\n2. Extracting keywords using {NUM_WORKERS} parallel workers...")
print(f"   Batch size: {BATCH_SIZE} papers per request")
print(f"   Total batches: {len(batches):,}")
print(f"   Estimated time: ~{len(batches) * DELAY / NUM_WORKERS / 60:.0f} minutes")
print(f"\n   Press Ctrl+C to stop (progress will be saved)")
start_time = time.time()

# Process batches
try:
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_batch = {executor.submit(fetch_keywords_batch, batch_info): batch_info for batch_info in batches}
        
        completed_batches = 0
        for future in as_completed(future_to_batch):
            completed_batches += 1
            batch_info = future_to_batch[future]
            batch_idx = batch_info[0]
            
            try:
                num_with_keywords, error = future.result()
                if error and "HTTP" in error:
                    print(f"\n   ⚠️  Batch {batch_idx}: {error}", flush=True)
            except Exception as e:
                print(f"\n   ⚠️  Batch {batch_idx} failed: {str(e)}", flush=True)
            
            if completed_batches % 20 == 0 or completed_batches == len(batches):
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                remaining = len(papers_to_process) - total_processed
                eta_minutes = (remaining / rate / 60) if rate > 0 else 0
                
                print(f"\r   [{total_processed:,}/{len(papers_to_process):,}] ({total_processed/len(papers_to_process)*100:.1f}%) "
                      f"| With keywords: {total_downloaded:,} | Rate: {rate:.1f}/s | ETA: {eta_minutes:.0f}m", 
                      end='', flush=True)
                
                save_progress()

except KeyboardInterrupt:
    print(f"\n\n⚠️  Interrupted by user. Saving progress...")
    save_progress()
    print(f"   Progress saved. Run again to continue.")
    exit(0)

# Final save
save_progress()

elapsed = time.time() - start_time
print(f"\n\n3. Complete!")
print(f"   Total processed: {len(processed_papers):,}")
print(f"   Papers with keywords: {total_downloaded:,}")
print(f"   Coverage: {total_downloaded/len(arxiv_ids)*100:.1f}%")
print(f"   Time elapsed: {elapsed/60:.1f} minutes")
print(f"   Average rate: {len(processed_papers)/elapsed:.1f} papers/second")
print(f"\n4. Output saved to: {OUTPUT_FILE}")

