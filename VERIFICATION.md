# Text-to-SQL Option A Verification

Final verification was executed against the packaged code state.

## Scope

- Option A only.
- No model retraining.
- No architecture redesign.
- Existing schema linking, semantic WHERE linking, SELECT, WHERE `=/>/<`, IN, COUNT, AVG, DISTINCT, ORDER BY ASC/DESC, LIMIT, and multi-column/compositional SQL were preserved.
- SQLite remains the verified runtime backend while the existing MySQL backend is preserved.

## Latest operator-intent regression

A real Streamlit/runtime failure was reproduced from:

```text
show customers older than 30
```

Before the fix it generated:

```sql
SELECT * FROM customers WHERE age < 30;
```

The failure was added as a regression test before changing code.

The smallest Option A fix was applied in beam ranking: explicit age-comparison language is used to disambiguate the operator branch (`older than` -> `>`, `younger than` -> `<`) while all other operator phrasing continues to use the existing operator head/ranking path.

After the fix the exact unseen/current-schema query generates:

```sql
SELECT * FROM customers WHERE age > 30;
```


## Live schema refresh regression

The Streamlit schema refresh path was reproduced and fixed test-first.

Two failure modes are now covered:

1. relative SQLite `DB_PATH` values are resolved from the project root rather than the launcher working directory;
2. `TextToSQLRuntime.refresh_schema()` opens a fresh configured database connection, rescans the schema, rebuilds the pipeline, and closes the old connection.

Executed Streamlit UI smoke:

- initial sidebar schema: `employees`;
- an external SQLite connection created and committed `customers`;
- **Refresh Schema** was clicked;
- refreshed sidebar schema: `customers`, `employees`;
- result: **passed**.


## Unseen `students` row-filter regression

Two real unseen-schema failures were reproduced against `students(ID, Name, Age)`:

```text
show all students older than 15
show students older than 15
```

Both are now required to generate:

```sql
SELECT * FROM students WHERE Age > 15;
```

The regressions were added before changing code. The smallest Option A ranking fix was then applied:

- explicit comparison intent outranks generic `all/every` list wording, so a requested filter is not removed;
- when a question names a schema table/entity but does not request a projection column, `STAR` is the preferred projection family, preventing drift to `COUNT`, `DISTINCT`, or a model-selected single column.

The exact two regression tests pass in a fresh process. A Streamlit + SQLite smoke using an unseen `students` table also passed for both questions.

## Executed checks

- Project/targeted pytest suite: **33/33 passed**.
- Required 32-case regression evaluation: **32/32 passed**.
- Unseen/generalization evaluation: **33/33 passed**.
- Exact unseen/current-schema runtime smoke for `customers`: **passed**.
  - Question: `show customers older than 30`
  - Generated: `SELECT * FROM customers WHERE age > 30;`
  - Returned only the seeded rows with ages 35 and 42.
- Streamlit + SQLite smoke for that exact query: **passed**.
  - Generated SQL was the expected `age > 30` query.
  - Non-empty result rows rendered successfully through the existing JSON fallback because the available wheel bundle does not include a Linux-compatible PyArrow wheel.
- Existing direct SQLite runtime smoke with returned rows: **passed**.
- SQLite live schema scanning: **passed**.
- SQLite Streamlit cross-thread connection regression: **passed**.
- Syntax/bytecode compile checks: **passed**.
- Import-time demo/schema execution remains guarded by `if __name__ == "__main__":`.
- Core beam-search debug prints: **none**.
- Local `.env` credentials are excluded from the distributable ZIP; use `.env.example` as the configuration template.

## SQLite runtime configuration

```bash
export DB_ENGINE=sqlite
export DB_PATH=/absolute/path/to/database.db
streamlit run app.py
```

The SQLite connection uses `check_same_thread=False` because Streamlit caches the runtime and can reuse it from a different script-runner thread on rerun.

## MySQL

The MySQL backend remains available with `DB_ENGINE=mysql`. The end-to-end verification was performed with SQLite, as permitted, so a live MySQL server is not required for this verified build.
