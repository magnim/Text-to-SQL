# Text-to-SQL

Generative Text-to-SQL project with constrained Option A beam search.

## Database backends

The runtime supports both SQLite and MySQL.

### SQLite

The packaged project includes a `.env` file with safe local defaults:

```dotenv
DB_ENGINE=sqlite
DB_PATH=text_to_sql.db
```

Before starting the app, change `DB_PATH` in `.env` to the SQLite database you want to query. If you prefer shell environment variables, they can still be used instead.

Then run:

```bash
streamlit run app.py
```

SQLite uses Python's built-in `sqlite3` module. The runtime scans the live SQLite schema, generates SQL against the current tables/columns, validates executable beams against the same connection, and returns SELECT results as dictionaries.


### Refreshing the live schema

After adding, removing, or changing tables/columns in the configured database, click **Refresh Schema** in the Streamlit sidebar.

Refresh now:

- opens a fresh database connection;
- rescans the live schema;
- rebuilds the SQL pipeline with the new table/column map;
- closes the old connection.

For SQLite, relative `DB_PATH` values are resolved from the project root, not from the directory where `streamlit` happened to be launched. This prevents the app from silently opening a different `.db` file.

If you edit the SQLite database in an external tool, save/commit that tool's changes before clicking **Refresh Schema**.

### MySQL

The existing MySQL path is preserved. Set:

```bash
export DB_ENGINE=mysql
export DB_HOST=localhost
export DB_USER=your_user
export DB_PASSWORD=your_password
export DB_NAME=your_database
```

PyMySQL is required only for the MySQL backend.

## Runtime files

- `app.py` — Streamlit UI
- `text_to_sql_runtime.py` — model/runtime wrapper
- `database/connection.py` — SQLite/MySQL connection selection
- `database/schema_scanner.py` — backend-aware live schema scanning
- `text_to_sql/sql_beam.py` — Option A constrained beam search

## Verification

See `VERIFICATION.md` for the executed regression, generalization, runtime, Streamlit, and cleanup checks.

## V2 zero-shot verification

The current V2 package includes learned zero-shot schema semantic grounding and a dedicated acceptance runner. From the repository root:

```bash
python evaluation/run_v2_zero_shot_acceptance.py
python evaluation/run_v2_regression_suite.py --suite unseen
python evaluation/run_v2_regression_suite.py --suite heldout
python evaluation/run_v2_regression_suite.py --suite legacy
```

The regression runner also accepts `--start` and `--end` for bounded CPU-friendly batches. See `Text-to-SQL_V2_ZERO_SHOT_PACKAGING_VERIFICATION.md` for the latest packaging verification and artifact hashes.

