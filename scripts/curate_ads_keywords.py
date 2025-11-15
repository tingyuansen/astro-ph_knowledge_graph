"""
Curate ADS keywords - remove arXiv classes and apply heuristics from keywords_old.
Keep only meaningful scientific keywords.
"""
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / 'ads_keywords' / 'ads_keywords_all.jsonl'
OUTPUT_FILE = BASE_DIR / 'ads_keywords' / 'ads_keywords_curated.csv'

print("="*70)
print("CURATE ADS KEYWORDS")
print("="*70)

# Step 1: Parse JSONL and flatten
print("\n1. Parsing keywords from JSONL...")
data = []
with open(INPUT_FILE, 'r') as f:
    for line in tqdm(f, desc="   Reading"):
        entry = json.loads(line)
        arxiv_id = entry['arxiv_id']
        
        for keyword in entry.get('keywords', []):
            if keyword:
                data.append({
                    'arxiv_id': str(arxiv_id),
                    'keyword': keyword.strip()
                })

df = pd.DataFrame(data)
print(f"   Total keyword entries: {len(df):,}")
print(f"   Unique keywords: {df['keyword'].nunique():,}")
print(f"   Papers with keywords: {df['arxiv_id'].nunique():,}")

# Step 2: Remove arXiv class keywords
print("\n2. Removing arXiv class keywords...")
# ArXiv categories that should be removed (too general)
arxiv_classes = [
    'astrophysics',
    'astrophysics - astrophysics of galaxies',
    'astrophysics - cosmology and nongalactic astrophysics',
    'astrophysics - earth and planetary astrophysics',
    'astrophysics - high energy astrophysical phenomena',
    'astrophysics - instrumentation and methods for astrophysics',
    'astrophysics - solar and stellar astrophysics',
    'general relativity and quantum cosmology',
    'high energy physics - phenomenology',
    'high energy physics - theory',
    'nuclear theory',
    'physics - instrumentation and detectors',
    'physics - computational physics',
    'quantum physics'
]

# Normalize keywords to lowercase for filtering
df['keyword_lower'] = df['keyword'].str.lower()
before_count = len(df)
df = df[~df['keyword_lower'].isin(arxiv_classes)]
removed_arxiv = before_count - len(df)
print(f"   Removed {removed_arxiv:,} arXiv class entries")

# Step 3: Normalize keywords to lowercase
print("\n3. Normalizing keywords to lowercase...")
df['keyword'] = df['keyword_lower']
df = df.drop(columns=['keyword_lower'])

# Step 4: Remove overly common keywords (>20,000 occurrences)
print("\n4. Removing overly common keywords (>20,000 occurrences)...")
keyword_counts = df['keyword'].value_counts()
common_threshold = 20000
common_keywords = keyword_counts[keyword_counts > common_threshold]

print(f"   Keywords with >{common_threshold:,} occurrences: {len(common_keywords)}")
if len(common_keywords) > 0:
    print(f"   Top common keywords to remove:")
    for kw, count in common_keywords.head(10).items():
        print(f"      - {kw}: {count:,}")

before_count = len(df)
df = df[~df['keyword'].isin(common_keywords.index)]
print(f"   Removed {before_count - len(df):,} entries")

# Step 5: Remove rare keywords (<10 occurrences)
print("\n5. Removing rare keywords (<10 occurrences)...")
keyword_counts = df['keyword'].value_counts()
min_occurrences = 10
rare_keywords = keyword_counts[keyword_counts < min_occurrences]

print(f"   Keywords with <{min_occurrences} occurrences: {len(rare_keywords):,}")
before_count = len(df)
df = df[~df['keyword'].isin(rare_keywords.index)]
print(f"   Removed {before_count - len(df):,} entries")

# Step 6: Remove duplicates
print("\n6. Removing duplicate arxiv_id + keyword pairs...")
before_count = len(df)
df = df.drop_duplicates(subset=['arxiv_id', 'keyword'])
print(f"   Removed {before_count - len(df):,} duplicates")

# Step 7: Sort and save
print("\n7. Sorting and saving...")
df = df.sort_values(by=['arxiv_id', 'keyword'])
df.to_csv(OUTPUT_FILE, index=False)

# Final statistics
print(f"\n8. Final Statistics:")
print(f"   {'='*60}")
print(f"   Total paper-keyword associations: {len(df):,}")
print(f"   Unique papers: {df['arxiv_id'].nunique():,}")
print(f"   Unique keywords: {df['keyword'].nunique():,}")
print(f"   Average keywords per paper: {len(df) / df['arxiv_id'].nunique():.2f}")
print(f"   {'='*60}")

# Show top keywords
print(f"\n9. Top 30 keywords by frequency:")
top_keywords = df['keyword'].value_counts().head(30)
for i, (kw, count) in enumerate(top_keywords.items(), 1):
    print(f"   {i:2d}. {kw:50s} ({count:,})")

print(f"\n✅ Curation complete!")
print(f"   Output: {OUTPUT_FILE}")

