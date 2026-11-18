# Migration Runbook — CatchyLabs Live Schema

> **Safety contract:** `migrate_live.py` never reads `.env`.  
> `DATABASE_URL` must be set explicitly in your shell, and `--apply` will not
> run unless (1) a `.dump` backup less than 24 hours old exists in `backups/`
> and (2) you type the Supabase project ID at the prompt.

---

## 0. Pre-requisites

| Tool | Minimum version | Install |
|------|----------------|---------|
| Python | 3.10 | `python --version` |
| psql / pg_dump | 14 | comes with PostgreSQL client |
| SQLAlchemy | 2.x | `pip install sqlalchemy` |

```bash
# Export the live connection string (DO NOT commit this)
export DATABASE_URL="postgresql://postgres.<project-ref>:<password>@db.<project-ref>.supabase.co:5432/postgres"
```

---

## 1. Backup

Take a compressed custom-format backup before every apply.  
`--apply` will refuse to run if no `.dump` file in `backups/` is less than 24 hours old.

```bash
mkdir -p backups
pg_dump "$DATABASE_URL" \
  --format=custom \
  --no-acl \
  --no-owner \
  --file "backups/$(date +%Y%m%d_%H%M%S).dump"
```

Verify the backup was written:

```bash
ls -lh backups/*.dump | tail -5
```

---

## 2. Dry Run (no database connection)

Print every SQL statement that would be executed.  
**No connection is made. This is always safe.**

```bash
python migrate_live.py
```

Expected output: numbered SQL statements 1–33, prefixed `-- Statement N`.

---

## 3. Inspect Live Schema (read-only)

Connect to the live database and print tables, columns, indexes, constraints,
and approximate row counts using `SELECT` statements only.  
No DDL or DML is executed.

```bash
python migrate_live.py --inspect
```

Run this before and after `--apply` to confirm the schema changed as expected.

---

## 4. Apply Migration

### 4a. Pre-flight checks (automatic)

`--apply` runs these checks before opening any connection:

1. **Backup freshness** — aborts if no `.dump` in `backups/` is less than 24 hours old (exit code 3).  
2. **Live target display** — prints host, database name, and Supabase project ID.  
3. **Typed confirmation** — you must type the project ID exactly (Ctrl-C aborts safely).

### 4b. Run

```bash
python migrate_live.py --apply
```

You will see:

```
[PREFLIGHT OK] Recent backup found: 20261024_181500.dump  (age: 0.1h)

============================================================
  LIVE TARGET DETAILS
  Host       : db.<project-ref>.supabase.co
  Database   : postgres
  Project ID : <project-ref>
============================================================

Type the Project ID exactly to confirm (Ctrl-C to abort):
  Project ID: <project-ref>

[CONFIRMED] Proceeding with migration on project '<project-ref>'.

Migration applied successfully to database 'postgres'.
```

> **Transaction guarantee:** all 33 statements run inside a single
> `BEGIN … COMMIT`. If any statement fails, Postgres rolls back the entire
> transaction automatically — the database is left unchanged.

---

## 5. Verification Queries

Run these `SELECT`-only queries after a successful apply to confirm the
migration landed correctly.

```sql
-- 1. Confirm new columns exist on the five tables
SELECT table_name, column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_schema = 'public'
  AND column_name IN ('superseded_at', 'claims_to_verify')
ORDER BY table_name;

-- 2. Confirm content_feedback table exists with all columns
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name   = 'content_feedback'
ORDER BY ordinal_position;

-- 3. Confirm the partial unique index on competitive_intelligence
SELECT indexname, indexdef
FROM pg_indexes
WHERE schemaname = 'public'
  AND tablename  = 'competitive_intelligence'
  AND indexname  = 'ix_ci_research_run_active';
-- Expected: ... WHERE (superseded_at IS NULL)

-- 4. Confirm the old unconditional unique constraint is gone
SELECT conname, pg_get_constraintdef(oid)
FROM pg_constraint
WHERE conrelid = 'public.competitive_intelligence'::regclass
  AND contype  = 'u';
-- Expected: 0 rows (the old UNIQUE (research_run_id) was dropped)

-- 5. Confirm FK constraints on opportunity_scores and content_pieces
SELECT tc.table_name, tc.constraint_name, rc.update_rule, rc.delete_rule
FROM information_schema.table_constraints tc
JOIN information_schema.referential_constraints rc
  ON rc.constraint_name = tc.constraint_name
WHERE tc.table_schema = 'public'
  AND tc.table_name IN ('opportunity_scores', 'content_pieces')
  AND tc.constraint_name LIKE 'fk_%restrict%'
ORDER BY tc.table_name;
-- Expected: 4 rows, all delete_rule = 'RESTRICT'

-- 6. Confirm Row Level Security is enabled on all 10 tables
SELECT relname, relrowsecurity
FROM pg_class
WHERE relnamespace = 'public'::regnamespace
  AND relkind      = 'r'
  AND relname IN (
    'research_runs','sources','evidence','api_calls',
    'consumer_insights','competitive_intelligence',
    'creative_gaps','opportunity_scores','content_pieces','content_feedback'
  )
ORDER BY relname;
-- Expected: relrowsecurity = true for all 10 rows

-- 7. Confirm content_feedback FK is SET NULL on delete
SELECT pg_get_constraintdef(oid)
FROM pg_constraint
WHERE conrelid = 'public.content_feedback'::regclass
  AND contype  = 'f'
  AND conname  = 'fk_content_feedback_piece_set_null';
-- Expected: FOREIGN KEY (content_piece_id) REFERENCES content_pieces(id) ON DELETE SET NULL
```

---

## 6. Rollback

If anything goes wrong **after** a successful apply (the transaction already
committed), restore from the backup taken in Step 1.

> ⚠️ This restores the entire database to its pre-migration state.
> All rows inserted after the backup was taken will be lost.
> Coordinate with your team before running.

```bash
# Identify the backup file to restore from
ls -lh backups/*.dump

# Restore (replace <project-ref> and <password>)
pg_restore \
  --dbname "$DATABASE_URL" \
  --clean \
  --if-exists \
  --no-acl \
  --no-owner \
  --verbose \
  backups/<filename>.dump
```

After the restore, re-run the verification queries to confirm the old schema
is back and row counts match expectations.

---

## 7. Statement-by-Statement Analysis

> Reference for the dry-run output (`python migrate_live.py`).

| # | Statement summary | Can lose data? | Idempotent? |
|---|-------------------|---------------|-------------|
| 1 | Install `pgvector` extension | No | ✅ `IF NOT EXISTS` |
| 2 | Add `superseded_at TIMESTAMPTZ NULL` to `consumer_insights` | No | ✅ `IF NOT EXISTS` |
| 3 | Add `superseded_at TIMESTAMPTZ NULL` to `competitive_intelligence` | No | ✅ `IF NOT EXISTS` |
| 4 | Add `superseded_at TIMESTAMPTZ NULL` to `creative_gaps` | No | ✅ `IF NOT EXISTS` |
| 5 | Add `superseded_at TIMESTAMPTZ NULL` to `opportunity_scores` | No | ✅ `IF NOT EXISTS` |
| 6 | Add `superseded_at TIMESTAMPTZ NULL` to `content_pieces` | No | ✅ `IF NOT EXISTS` |
| 7 | Add `claims_to_verify TEXT[] DEFAULT '{}'` to `content_pieces` | No | ✅ `IF NOT EXISTS` |
| 8 | Create `content_feedback` table | No | ✅ `IF NOT EXISTS` |
| 9 | Create index `ix_content_feedback_content_piece_id` | No | ✅ `IF NOT EXISTS` |
| 10 | Add `snapshot_title TEXT NULL` to `content_feedback` | No | ✅ `IF NOT EXISTS` |
| 11 | Add `snapshot_hook TEXT NULL` to `content_feedback` | No | ✅ `IF NOT EXISTS` |
| 12 | Add `snapshot_body JSON NULL` to `content_feedback` | No | ✅ `IF NOT EXISTS` |
| 13 | Add `snapshot_claims_to_verify VARCHAR[] NULL` to `content_feedback` | No | ✅ `IF NOT EXISTS` |
| 14 | Add `snapshot_creative_gap_id INTEGER NULL` to `content_feedback` | No | ✅ `IF NOT EXISTS` |
| 15 | Add `status VARCHAR(20)` (nullable, no default) to `content_feedback` | No | ✅ `IF NOT EXISTS` |
| 16 | Add `note TEXT NULL` to `content_feedback` | No | ✅ `IF NOT EXISTS` |
| 17 | Add `created_at TIMESTAMPTZ DEFAULT now()` to `content_feedback` | No | ✅ `IF NOT EXISTS` |
| 18 | DROP any `UNIQUE (research_run_id)` constraint on `competitive_intelligence` | **Yes — drops constraint** (no row data deleted) | ✅ DO/FOR loop is a no-op if constraint absent |
| 19 | Replace FK on `content_feedback.content_piece_id` with `ON DELETE SET NULL` version | No (SET NULL is safer than CASCADE) | ✅ DO/FOR loop is no-op if old FK absent |
| 20 | DROP index `ix_competitive_intelligence_research_run_id` | **Yes — drops index** (no row data deleted) | ✅ `IF EXISTS` |
| 21 | DROP index `ix_ci_research_run_active` | **Yes — drops index** (no row data deleted) | ✅ `IF EXISTS` |
| 22 | CREATE partial UNIQUE INDEX `ix_ci_research_run_active WHERE superseded_at IS NULL` | No (replaces #20/#21 with a narrower index) | ✅ idempotent if combined with #20/#21 |
| 23 | Replace FKs on `opportunity_scores` and `content_pieces` with `ON DELETE RESTRICT` variants | **Yes — briefly drops FKs** (no row data deleted) | ✅ DO/FOR loop is no-op if old FKs absent; ADD CONSTRAINT adds them back |
| 24–33 | `ENABLE ROW LEVEL SECURITY` on 10 tables | No | ✅ no-op if RLS already enabled |

**Transaction:** statements 1–33 execute inside a single `BEGIN … COMMIT`.
Any failure rolls back all changes automatically.

---

## 8. Quick-Reference Commands

```bash
# Dry run (always safe — no connection)
python migrate_live.py

# Inspect live schema (read-only SELECT)
python migrate_live.py --inspect

# Apply (requires fresh backup + typed project ID)
python migrate_live.py --apply

# Run the test suite (uses TEST_DATABASE_URL only)
pytest -v tests/
```
