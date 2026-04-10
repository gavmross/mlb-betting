---
name: db-migrate
description: >
  Safely apply a schema migration to data/mlb.db. Use when the user says
  "add a column", "change the schema", "migrate the database", or
  "update the table structure". Never drops or truncates data.
---

# Database Migration Workflow

## Step 1 — Understand the Change

Ask the user if not already clear:
1. Which table is being modified?
2. What is the change? (add column / add index / add table)
3. Is this additive only? (adding columns/indexes is safe; renaming/removing is risky)

**If the user wants to rename or remove a column or table — pause and warn:**
"Renaming/removing columns in SQLite requires recreating the table. This can
be destructive. Please confirm you want to proceed and that data/mlb.db
is backed up."

## Step 2 — Backup First

```bash
cp data/mlb.db data/mlb_backup_$(date +%Y%m%d_%H%M%S).db
echo "Backup created"
ls -lh data/mlb_backup_*.db | tail -1
```

## Step 3 — Inspect Current Schema

```bash
sqlite3 data/mlb.db ".schema {table_name}"
```

Confirm the current structure before making any changes.

## Step 4 — Apply Migration

For adding a column (most common case):
```bash
sqlite3 data/mlb.db "ALTER TABLE {table_name} ADD COLUMN {column_name} {type} DEFAULT {default};"
```

Example:
```bash
sqlite3 data/mlb.db "ALTER TABLE predictions ADD COLUMN dispersion_alpha REAL;"
sqlite3 data/mlb.db "ALTER TABLE predictions ADD COLUMN lambda_home REAL;"
sqlite3 data/mlb.db "ALTER TABLE predictions ADD COLUMN lambda_away REAL;"
```

For adding an index:
```bash
sqlite3 data/mlb.db "CREATE INDEX IF NOT EXISTS idx_{table}_{col} ON {table}({col});"
```

For adding a new table:
Write the full CREATE TABLE statement to a migration .sql file first,
then apply it:
```bash
sqlite3 data/mlb.db < migrations/{migration_name}.sql
```

## Step 5 — Verify

```bash
sqlite3 data/mlb.db ".schema {table_name}"
```

Confirm the new column/index/table appears correctly.

```bash
sqlite3 data/mlb.db "SELECT COUNT(*) FROM {table_name}"
```

Confirm row count is unchanged (no accidental data loss).

## Step 6 — Update Code

1. Update the CREATE TABLE statement in `mlb/db.py` to match the new schema
2. Update any INSERT or SELECT statements that touch the modified table
3. Update `docs/ARCHITECTURE.md` — add a note to the relevant table's schema
   block describing the new column and why it was added

## Step 7 — Run Tests

```bash
pytest tests/ -v -q
```

Confirm all tests still pass after the schema change.

## Step 8 — Document

Add a one-line entry to the schema change log in docs/ARCHITECTURE.md:
```
## Schema Change Log
- YYYY-MM-DD: Added `lambda_home`, `lambda_away`, `dispersion_alpha` to predictions
              table to support two-target Poisson model output
```
