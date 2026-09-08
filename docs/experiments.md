# Paper experiments

The official, resumable experimental suite lives in
[`experiments/`](../experiments/README.md). It covers canonical GraphTopic runs,
ANN fidelity, graph diagnostics, five-seed external agreement, controlled baselines,
lexical quality, ablations, and scaling.

Large datasets, embeddings, graph caches, and assignments are deliberately excluded
from source control. The default runner downloads pinned public sources and model
revisions, creates missing embeddings, validates checksums, and reuses complete local
artifacts. It also enforces the exact paper dependency lock before doing work.

Passing unit tests proves that the harness works; it does not by itself reproduce the
paper's numbers. Earlier uncontrolled partial outputs were retired. The completed
locked run was reviewed, promoted to `experiments/expected/reference-results.json`,
and reconciled with the LaTeX manuscript.
