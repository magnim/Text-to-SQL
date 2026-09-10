# Text-to-SQL V2 Zero-Shot Operator-Scope Verification

Date: 2026-09-10

## Scope

This continuation starts from the packaged V2 role-separation build, preserves that fix, and removes full-schema contamination from comparison-operator prediction while keeping zero-shot schema semantic grounding unchanged.

The TinyGPT checkpoint and semantic schema encoder checkpoint were **not retrained or changed**. Experimental role-head retraining was investigated, rejected because it did not robustly solve the production-path failure, and the original semantic checkpoint was restored byte-for-byte.

## Root cause found

The previous schema-aware Transformer test helper instantiated `TextToSQLPipeline` without the production `SchemaSemanticEncoderV2`. When the tests were aligned with the real runtime, three composition cases exposed that the full V2 semantic weight (`4.0`) could make the ORDER semantic role reuse the WHERE column:

```text
WHERE age > 30      ORDER BY salary       -> ORDER BY age
WHERE price > 10    ORDER BY rating       -> ORDER BY price
WHERE quantity > 2  ORDER BY total_amount -> ORDER BY quantity
```

The semantic ORDER head was not reliably separating the two clause roles on longer composed questions, while TinyGPT's existing clause-conditioned ORDER scorer selected the correct ORDER columns when the V2 ORDER override was reduced.

## Implemented fix

`TextToSQLPipeline` and `SQLBeamSearch` now use separate semantic weights:

- table / WHERE V2 semantic grounding: **4.0**
- ORDER BY V2 semantic grounding: **0.25**

This is schema-generic scoring fusion. No table names, column names, question phrases, synonym maps, or hand-coded `price -> rating`/similar rules were added.

The schema-aware Transformer test helper now loads the same `semantic_schema_encoder_v2.pt` checkpoint used by production, so model-backed tests exercise the actual V2 path.

## Fresh verification

Executed against the current working tree after the fix:

- Movie temporal regressions: **2/2**
  - `movies released after 2014` -> `SELECT * FROM movies WHERE release_year > 2014;`
  - `movies release year after 2014` -> `SELECT * FROM movies WHERE release_year > 2014;`
- Single-numeric WHERE-vs-LIMIT role guard: **passed**.
- WHERE + ORDER BY + LIMIT composition: **3/3**.
- Zero-shot semantic acceptance: **5/5**.
- Unseen current-schema suite: **4/4**.
- Held-out semantic generalization: **6/6**.
- Legacy generation suite: **21/21**, freshly rerun in completed bounded chunks.
- Unseen-schema ORDER column grounding (new regression coverage): **3/3**.
  - `movies.release_year`
  - `vehicles.mileage`
  - `meal_plans.preparation_time`
- Fast evaluation/runtime/schema tests: **13/13**.
- Production anti-hardcoding source guard: **passed**.
- `python -m compileall`: **passed**.
- Strict TinyGPT checkpoint load: **passed**.
- Strict semantic schema encoder V2 checkpoint load: **passed**.

## Checkpoint integrity

TinyGPT SHA-256 remains:

```text
edfebc0d701c3e4a3da76aca273709110d125bf08a0f65bc1ac2d8b4095e79c0  tiny_gpt_schema_aware_final.pt
```

Semantic schema encoder V2 SHA-256 remains:

```text
f42c231655ba42e725b7f54bc458fb0da39e5cb7f83cbee6efeee58f3131cd24  semantic_schema_encoder_v2.pt
```


## Operator-intent schema-contamination fix

A production-path regression was reproduced for:

```text
movies release year after 2014
```

The same trained operator head changed its comparison prediction when unrelated tables were added to the schema. In the reproducible regression schema, the generated SQL flipped from the correct `release_year > 2014` to `release_year < 2014`. This confirms that comparison intent was being contaminated by irrelevant full-database schema context.

The fix keeps the existing pretrained operator classifier and schema conditioning, but changes only its inference boundary: after V2 semantic schema scoring selects the highest-scoring table, operator logits are recomputed from a prompt containing that selected table plus the original question. Other intent heads still use the complete schema. No phrase-specific mapping such as `after -> >` was added.

Fresh regression evidence after this change:

- exact movie temporal regressions: **2/2**;
- noisy-schema operator invariance regression: **1/1**;
- WHERE + ORDER BY + LIMIT composition: **3/3**;
- zero-shot semantic acceptance: **5/5**;
- unseen schema suite: **4/4**;
- held-out semantic generalization: **6/6**;
- legacy generation suite: **21/21**, freshly rerun in completed bounded chunks;
- fast evaluation/runtime/schema tests: **13/13**;
- `python -m compileall`: **passed**.

The TinyGPT and semantic-schema checkpoints remain byte-for-byte unchanged.

## Packaged `.env`

The ZIP now intentionally includes a `.env` file. It contains safe/default SQLite configuration only:

```dotenv
DB_ENGINE=sqlite
DB_PATH=text_to_sql.db
```

Users should replace `DB_PATH` with the database they actually want to query. No real credentials are included.

## Known separate limitation discovered during verification

An exploratory unseen-schema query using the explicit direction word `ascending` selected the correct unseen ORDER column but generated `DESC`. This is a **direction-intent** issue, not an ORDER-column semantic-grounding issue, and was not changed as part of this role-separation fix. Existing verified legacy ASC cases still pass.

Example observed during the probe:

```text
show recipes ordered by preparation time ascending
```

Column grounding: `preparation_time` (correct)  
Direction: `DESC` (incorrect; expected `ASC`)

## Reproducible commands

From the repository root:

```bash
python evaluation/run_v2_zero_shot_acceptance.py
python evaluation/run_v2_regression_suite.py --suite unseen --start 0 --end 4
python evaluation/run_v2_regression_suite.py --suite heldout --start 0 --end 3
python evaluation/run_v2_regression_suite.py --suite heldout --start 3 --end 6
python evaluation/run_v2_regression_suite.py --suite legacy --start 0 --end 4
python evaluation/run_v2_regression_suite.py --suite legacy --start 4 --end 8
python evaluation/run_v2_regression_suite.py --suite legacy --start 8 --end 12
python evaluation/run_v2_regression_suite.py --suite legacy --start 12 --end 16
python evaluation/run_v2_regression_suite.py --suite legacy --start 16 --end 20
python evaluation/run_v2_regression_suite.py --suite legacy --start 20 --end 21
python -m pytest -q tests/test_schema_aware_transformer.py -k 'where_order_limit_composition or movie_temporal_value_role_regression or zero_shot_order_column_grounding'
python -m pytest -q tests/test_evaluation_entrypoints.py tests/test_sql_beam_unit.py tests/test_sqlite_runtime_support.py tests/test_schema_refresh.py
python -m compileall -q app.py text_to_sql_runtime.py database text_to_sql pytorch_impl src training evaluation tests
```

## Verified scope

The package remains SELECT-only and preserves the existing supported subset: projections, DISTINCT, COUNT, AVG, WHERE comparisons/IN, ORDER BY ASC/DESC for the established regression set, LIMIT, and the verified `WHERE -> ORDER BY -> LIMIT` composition path. Zero-shot semantic WHERE grounding remains fully green in the current acceptance suite.
## Simple-WHERE semantic grounding fix

A production-path failure was reproduced for:

```text
movies released after 2019
```

With a realistic `movies` schema containing `director`, the pipeline selected:

```sql
SELECT * FROM movies WHERE director > 2019;
```

The comparison operator was already correct. The failure was in WHERE-column grounding: TinyGPT's WHERE scorer preferred `director`, while the V2 generic semantic encoder strongly preferred `release_year`. The role-specific WHERE encoder only weakly separated the two columns and therefore could be overruled.

The fix is schema-generic and clause-aware:

- simple WHERE questions use the V2 generic semantic column scores for WHERE grounding;
- questions whose learned lexical structure predicts ORDER BY keep the role-specific WHERE score, preserving WHERE/ORDER column separation;
- no movie-specific, `released -> release_year`, or other phrase-to-schema rule was added;
- TinyGPT and both semantic checkpoints remain unchanged.

Fresh verification after this change:

- noisy movie simple-WHERE regressions: **3/3**
  - `movies released after 2019` -> `release_year > 2019`
  - `movies release year after 2019` -> `release_year > 2019`
  - `movies released before 2019` -> `release_year < 2019`
- WHERE + ORDER BY + LIMIT role separation: **3/3**
- zero-shot semantic acceptance: **5/5**
- unseen schema suite: **4/4**
- held-out semantic generalization: **6/6**
- legacy generation suite: **21/21**, freshly rerun in completed chunks
- unseen-schema ORDER column grounding: **3/3**
- fast runtime/package tests: **13/13**
- movie temporal + noisy-operator + anti-hardcoding source guards: **4/4**
- `python -m compileall`: **passed**
- strict TinyGPT checkpoint load: **passed**
- strict semantic schema encoder V2 checkpoint load: **passed**

The packaged `.env` remains included with safe/default SQLite values and no real credentials.



## Large-schema generation-headroom fix

A production-path failure was reproduced for:

```text
podcasts released after 2019
```

The full schema/question prompt was **486 tokens**, below TinyGPT's 512-token maximum, so the old initial prompt check passed. During beam generation, SQL tokens were appended to that prompt and the sequence crossed 512, raising:

```text
ValueError: Input sequence length exceeds maximum_sequence_length.
```

The fix reserves **96 tokens** for SQL generation. Normal schemas that fit within the remaining prompt budget continue to use the complete schema unchanged. When a schema would consume the generation reserve, the pipeline compacts only TinyGPT's prompt to one relevant table. Exact normalized table mentions in the question are preferred; otherwise the V2 semantic table score selects the prompt table. The decoder is restricted to that selected prompt table for that compacted request, while the complete database schema remains available to execution validation and full-schema semantic scoring.

No `podcasts`, `release_year`, or phrase-to-schema mapping is hardcoded. TinyGPT and the V2 semantic checkpoint were not retrained or modified.

Fresh verification after this change:

- large noisy-schema podcast regression: **1/1** (`podcasts released after 2019` -> `SELECT * FROM podcasts WHERE release_year > 2019;`)
- movie temporal/noisy-schema regressions: **6/6** across completed runs
- WHERE + ORDER BY + LIMIT composition: **3/3**
- zero-shot semantic acceptance: **5/5**
- unseen schema suite: **4/4**
- held-out semantic generalization: **6/6**
- legacy generation suite: **21/21**, freshly rerun in completed chunks
- fast evaluation/runtime/schema tests: **13/13**
- production anti-hardcoding source guard: **passed**
- `python -m compileall`: **passed**
- TinyGPT and semantic checkpoint SHA-256 values remain unchanged
- packaged `.env`: included with safe SQLite defaults and no real credentials

The context reserve is an inference-time budget only; model positional embeddings remain at the trained 512-token size.
