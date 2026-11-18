import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests
import json
from database import SessionLocal, ResearchRun, Source, Evidence, save_research_run

BASE_URL = 'http://127.0.0.1:8000'

print('============================================================')
print('           PHASE 2 VERIFICATION CHECKLIST REPORT            ')
print('============================================================')

# Step 1: Health check
r_health = requests.get(f'{BASE_URL}/health')
print(f'[STEP 1] Server Health Check: Status {r_health.status_code} ({r_health.json()})')
assert r_health.status_code == 200

# Step 2: Create Research Run in DB
brief = {
    'brand': 'Puma',
    'market': 'India',
    'audience': 'Gen Z',
    'objective': 'Find opportunities for sustainable running shoes'
}
report_data = {
    'metadata': {
        'brand': 'Puma',
        'market': 'India',
        'audience': 'Gen Z',
        'objective': 'Find opportunities for sustainable running shoes',
        'generated_at': '2026-09-11T23:15:00Z'
    },
    'evidence': [
        {
            'claim': 'Puma Gen Z environmental initiative in India gained 45% engagement growth.',
            'source': {
                'title': 'Puma India Sustainability Study 2026',
                'url': 'https://about.puma.com/en/sustainability/india-gen-z',
                'publisher': 'Puma Corporate',
                'published_date': '2026-01-15T00:00:00'
            },
            'evidence': 'Puma reported significant uptake in recycled footwear lines among Indian Gen Z runners.',
            'market': 'India',
            'audience': 'Gen Z',
            'topic': 'eco-friendly preference',
            'confidence': 0.94
        }
    ]
}

run_id = save_research_run(brief, report_data)
print(f'[STEP 2] Submitted Research Run -> Generated run_id: {run_id}')

# Step 3: Verify PostgreSQL Supabase Tables
session = SessionLocal()
db_run = session.query(ResearchRun).filter_by(id=run_id).first()
print(f'[STEP 3.1] research_runs Table: Found row -> ID={db_run.id}, Brand={db_run.brand}, Market={db_run.market}, Audience={db_run.audience}')

db_ev = session.query(Evidence).filter_by(research_run_id=run_id).all()
print(f'[STEP 3.2] evidence Table: Found {len(db_ev)} evidence row(s) linked to run_id {run_id}')

db_src_ids = [e.source_id for e in db_ev]
db_sources = session.query(Source).filter(Source.id.in_(db_src_ids)).all()
print(f'[STEP 3.3] sources Table: Found {len(db_sources)} source row(s):')
for s in db_sources:
    print(f'   -> Source ID: {s.id} | Publisher: {s.publisher} | URL: {s.url}')

# Step 4: GET /research/{run_id} API endpoint
r_get = requests.get(f'{BASE_URL}/research/{run_id}')
print(f'[STEP 4] GET /research/{run_id}: Status {r_get.status_code}')
fetched = r_get.json()
print(f'   -> Retrived Metadata ID: {fetched["metadata"]["id"]}')
print(f'   -> Evidence Claim: "{fetched["evidence"][0]["claim"]}"')
assert fetched['metadata']['id'] == run_id
assert len(fetched['evidence']) == 1

# Step 5: Test 404 Case
fake_id = '00000000-0000-0000-0000-000000000000'
r_404 = requests.get(f'{BASE_URL}/research/{fake_id}')
print(f'[STEP 5] GET /research/{fake_id}: Status {r_404.status_code} ({r_404.json()})')
assert r_404.status_code == 404

# Step 6: Test Dedup
initial_source_count = session.query(Source).count()
# Insert second run with identical source URL & title
run_id_2 = save_research_run(brief, report_data)
final_source_count = session.query(Source).count()
print(f'[STEP 6] Source Deduplication Check:')
print(f'   -> Initial total sources in DB: {initial_source_count}')
print(f'   -> Sources in DB after second run with identical sources: {final_source_count}')
assert initial_source_count == final_source_count
print('   -> DEDUPLICATION CONFIRMED! Existing source was reused via content_hash.')

session.close()
print('============================================================')
print('        ALL PHASE 2 VERIFICATION CHECKLIST ITEMS PASSED!     ')
print('============================================================')
