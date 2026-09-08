# Paper experiments

The official, resumable experimental suite lives in
[`experiments/`](../experiments/README.md). It covers canonical GraphTopic runs,
ANN fidelity, graph diagnostics, five-seed external agreement, controlled baselines,
lexical quality, ablations, and scaling.

Large datasets, embeddings, graph caches, and assignments are deliberately excluded
from source control. The default runner downloads pinned public sources and model
revisions, creates missing embeddings, validates checksums, and reuses complete local
artifacts. It also enforces the exact paper dependency lock before doing work.

Passing unit tests proves that the harness works; it does not prove that the paper's
numbers were reproduced. Earlier uncontrolled partial outputs were retired. A compact
reference report will be committed only after the new locked run is complete and the
paper has been reconciled with it.
