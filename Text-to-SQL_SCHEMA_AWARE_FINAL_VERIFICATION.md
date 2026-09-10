# Text-to-SQL Schema-Aware Transformer — Final Verification Manifest

Date: 2026-09-09

## Architecture

The production path is:

CURRENT DATABASE SCHEMA + QUESTION
→ schema/prompt encoder
→ BPE tokenizer
→ token + position + 5-way schema-role embeddings
→ decoder-only TinyGPT Transformer
→ learned semantic/schema grounding heads
→ autoregressive SQL token probabilities
→ schema/syntax constrained beam search
→ execution validator
→ final SQL

Model configuration:

- Vocabulary size: 1132
- Embedding dimension: 64
- Context length: 512
- Attention heads: 4
- Feed-forward hidden dimension: 128
- Transformer layers: 4
- Schema roles: NORMAL, TABLE, COLUMN, QUESTION, SQL
- Role count: 5
- Tied LM output/token embedding weights: yes
- Beam width: 12
- Complete active schema is encoded before the question.
- Decoder constraints enforce SQL grammar and identifiers from the active schema only.
- Execution validation checks executability; it does not supply semantic mappings.

Learned auxiliary components include projection/arity, clause, operator,
direction, continuation, contextual table/column pointers, lexical
schema grounding, WHERE/ORDER column pointers, case-normalized identifier
grounding, and learned WHERE/LIMIT numeric-role pointers.

## Training

- Fixed seed: 42
- Schema-conditioned source examples: 2942
- Held-out semantic cases: 6
- Held-out schema names `consultants` and `vehicles` are absent from the training corpus.
- Tokenizer: `tokenizer_balanced.json`
- Final checkpoint: `tiny_gpt_schema_aware_final.pt`
- Checkpoint SHA-256: `edfebc0d701c3e4a3da76aca273709110d125bf08a0f65bc1ac2d8b4095e79c0`

Final recorded calibration metrics:

- Lexical structure loss: 0.02707248734706604
- Lexical structure clause-bit accuracy: 0.9989802855200544
- Lexical continuation accuracy: 0.9979605710401088
- Numeric role-pointer loss: 0.006670856097457545
- Numeric role-pointer accuracy: 1.0
- Legacy structure loss: 0.40540788429124014
- Legacy structure projection accuracy: 1.0
- Legacy structure arity accuracy: 1.0
- Legacy structure direction accuracy: 1.0
- Projection-pointer accuracy: 0.8970588235294118
- SQL-target-only LM loss measurement: 5.294198311378248 over 3587 SQL target tokens from 160 schema-conditioned examples.

Important training limitation: the legacy Transformer backbone was retained/frozen
during the final schema-aware auxiliary-head calibration. The full active schema
is still passed through the Transformer at inference and the learned semantic
heads consume Transformer states, but this final artifact does not claim that
all backbone weights were schema-conditioned fine-tuned.

## Verification Results

- Pytest collected: 54
- Pytest passed: 54
- Pytest failed: 0
- Legacy generation: 21/21
- Unseen current-schema cases: 4/4
- Held-out semantic generalization: 6/6
- WHERE + ORDER BY + LIMIT compositions: 3/3
- Pure LIMIT regression guard: 3/3
- Movie temporal/value-role regressions: 2/2
- Fast unit/runtime/schema tests: 18/18
- Trained-model semantic suite: 6/6
- Reset-model ablation: 0/6
- `python -m compileall`: PASS
- Production anti-hardcoding source audit: PASS
- Strict final model load: PASS
- SQLite runtime generation: PASS
- SQLite execution: PASS
- Streamlit startup smoke: PASS

The reset-model ablation demonstrates a material semantic dependence on the
trained model: trained 6/6 versus deterministic reset 0/6 on the held-out
semantic suite.

## Production Hardcoding Audit

Production inference files were scanned for phrase-to-schema mappings and for
production instantiation of legacy `SchemaLinker`, `SemanticColumnLinker`, and
`SemanticColumnRanker`. No prohibited production inference mappings were found.

Numeric extraction uses a generic number regex only. Semantic table/column/value
selection is model-scored against the complete active schema.

## Known Limitations

- SELECT-only generation is the supported production scope.
- Explicit projection lists are limited to two columns by the constrained decoder.
- The final schema-aware calibration trains auxiliary Transformer-derived heads
  while retaining the legacy Transformer backbone weights.
- The decoder supports the verified SQL feature subset rather than arbitrary SQL,
  including SELECT *, explicit columns, multi-column projections, DISTINCT,
  COUNT, AVG, WHERE comparisons/IN, ORDER BY ASC/DESC, LIMIT, and verified
  WHERE→ORDER BY→LIMIT composition.


## Post-verification bug fix

A generic numeric-literal consumption guard prevents a WHERE literal from being reused as LIMIT unless that literal appears again in the question. A high-confidence morphology-only active-schema lexical grounding signal supports forms such as `released` → `release` without phrase-to-column mappings. Exact identifier-component matches remain model-scored and do not receive this fuzzy boost.
