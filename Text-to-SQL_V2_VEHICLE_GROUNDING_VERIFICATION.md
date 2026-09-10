# Text-to-SQL V2 Vehicle Grounding Verification

## Scope

This update fixes numeric/temporal WHERE-column grounding for cases such as:

- `vehicles released after 2018`
- `vehicles released after 2019`

Ordinal ranking / `OFFSET` support is intentionally not included in this update.

## Root cause

The semantic schema encoder could rank text columns such as `model` or `fuel_type` above `year`, and the ranking could change when only the numeric literal changed. The runtime previously retained column names but did not expose declared database column types to inference.

## Fix

- Added separate column-type scanning for SQLite and MySQL while preserving the existing `schema: dict[str, list[str]]` interface.
- Runtime now passes column-type metadata into `TextToSQLPipeline` and refreshes it with schema changes.
- Numeric `>` / `<` WHERE comparisons apply a generic type-compatibility prior:
  - text columns are strongly disfavored;
  - numeric/temporal candidates retain and strengthen semantic ranking;
  - a year-shaped literal may favor a schema column whose identifier itself contains `year`, unless a different column is explicitly named in the question.
- No `released -> year`, vehicle-specific, table-specific, or phrase-to-column mapping was added.

## Focused regression

Verified typed-schema behavior:

- `vehicles released after 2014` -> `SELECT * FROM vehicles WHERE year > 2014;`
- `vehicles released after 2018` -> `SELECT * FROM vehicles WHERE year > 2018;`
- `vehicles released after 2019` -> `SELECT * FROM vehicles WHERE year > 2019;`
- `vehicles released after 2025` -> `SELECT * FROM vehicles WHERE year > 2025;`
- `show vehicles with mileage greater than 2025` -> `SELECT * FROM vehicles WHERE mileage > 2025;`

## Regression verification

- Fast runtime/package tests: 13/13
- WHERE + ORDER BY + LIMIT composition: 3/3
- Zero-shot ORDER + movie temporal: 5/5
- Noisy movie WHERE: 3/3
- Operator/context-budget checks: 2/2
- V2 zero-shot acceptance: 5/5
- Unseen schema: 4/4
- Held-out semantic: 6/6
- Legacy generation: 21/21
- Anti-hardcoding source guard: passed
- `python -m compileall`: passed

## Checkpoint integrity

TinyGPT remains unchanged:

`edfebc0d701c3e4a3da76aca273709110d125bf08a0f65bc1ac2d8b4095e79c0`

Semantic schema encoder V2 remains unchanged:

`f42c231655ba42e725b7f54bc458fb0da39e5cb7f83cbee6efeee58f3131cd24`
