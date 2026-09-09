# GraphTopic paper experiments

This directory is the self-bootstrapping, checkpointed protocol for every experiment
reported in the current GraphTopic paper. It uses the public `graphtopic` package in
`src/`; it is not a second implementation of the method.

## Reproducibility contract

The default runner requires no pre-downloaded corpora, embeddings, or model files. On
its first run it downloads the pinned sources and Sentence Transformer revisions,
creates embeddings on CPU, verifies their checksums, and caches every large artifact
under a protocol-specific directory below the ignored `experiments/artifacts/`
directory. Later runs reuse only compatible files and resume an interrupted embedding
batch or experiment stage.

Paper-quality runs use fixed seeds, exact dependency versions, CPU embeddings,
deterministic PyTorch operations, one ANN worker, one classical-baseline worker, and
one numerical-library thread. The runner refuses to start work in a mismatched Python
or package environment. This contract targets the reported quality metrics in the
locked environment. Runtime and peak memory are hardware-specific; floating-point
files are not promised to be byte-identical across operating systems and CPU models.

The scaling stages intentionally use the available processors. They preserve their
sample and algorithm seeds, but timing and memory are descriptive measurements.
The exact-search comparison uses the same frozen nested AG News samples at 10,000
and 25,000 documents and the same candidate count. It remains hardware-specific.

## Create the reference environment on Windows

From Command Prompt (`cmd.exe`):

```bat
cd /d C:\path\to\graphtopic
py -3.14 -m venv .paper-venv
.paper-venv\Scripts\python.exe -m pip install --upgrade pip
.paper-venv\Scripts\python.exe -m pip install -r requirements\experiments-reference.txt
.paper-venv\Scripts\python.exe -m pip install -e . --no-deps
.paper-venv\Scripts\python.exe -m experiments.paper
```

From PowerShell, use the same commands without `cd /d`. Do not paste a PowerShell
leading `&` or backtick continuation into Command Prompt.

The first invocation can download several datasets/models and create multi-gigabyte
artifacts. Run the identical final command after interruption to resume. Inspect stage
names without starting work with:

```bat
.paper-venv\Scripts\python.exe -m experiments.paper --list
```

Run selected stages with repeated `--only STAGE`. The default order is bootstrap,
canonical GraphTopic, classical baselines, BERTopic, sensitivity/graph/encoder
ablations, scaling, lexical audits, and deterministic qualitative examples.

Bootstrap is split by artifact. For example, prepare only the two cleaned 20
Newsgroups embedding artifacts with:

```bat
.paper-venv\Scripts\python.exe -m experiments.paper --only bootstrap-20newsgroups --only bootstrap-20newsgroups-alternate
```

## Inputs and provenance

`configs/paper.json` pins the Hugging Face Parquet filenames and repository revisions,
Sentence Transformer commits, split order, text construction, dimensions, row/class
counts, seeds, graph settings, resolutions, and baseline settings. Parquet files are
downloaded directly through Hugging Face Hub, avoiding the Python-version-sensitive
`datasets` fingerprint/cache layer. 20 Newsgroups is fetched by the locked
scikit-learn release with `shuffle=False` and with headers, footers, and quoted text
removed so that metadata cues do not drive the semantic or lexical comparisons.

Every generated artifact contains compressed documents, labels, float32 embeddings,
and metadata with source/model identity plus SHA-256 checksums. Corrupt or incomplete
caches fail validation rather than being silently reused.

ANN Recall@20 is tie-aware: when several documents have exactly the cosine score at
the twentieth-neighbor boundary, any of those documents is accepted for the available
tie slots. The report also retains strict deterministic-index recall and the number of
queries affected by boundary ties. This prevents identical cleaned documents from
being counted as retrieval errors solely because their row identifiers differ.

Qualitative topics are selected by size with fixed tie-breaking. Their representative
documents are selected by maximum cosine similarity to the topic centroid, not by
manual preference.

When a protocol revision changes only one corpus, unchanged controlled artifacts may
be linked or copied into the new protocol namespace after validating their source and
encoder metadata. The carry-forward command deliberately excludes 20 Newsgroups:

```bat
.paper-venv\Scripts\python.exe -m experiments.carry_forward --previous-artifacts PATH\TO\artifacts --previous-results PATH\TO\results
```

The command requires a passing previous comparison record, writes an audit manifest,
and uses hard links when source and destination are on the same filesystem.

## Official Graph2Topic baseline

The nearest prior pipeline is evaluated with the upstream `graph2topictm==2.0`
package, not with a local reimplementation. Its 2023 dependency stack is isolated
from the main reference environment. After the cleaned 20 Newsgroups core runs have
established the matched topic counts, create a Python 3.9 environment and run:

```bat
py -3.9 -m venv .graph2topic-venv
.graph2topic-venv\Scripts\python.exe -m pip install -r requirements\graph2topic-reference.txt
set PYTHONHASHSEED=0
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
.graph2topic-venv\Scripts\python.exe -m experiments.graph2topic --artifact experiments\artifacts\graphtopic-paper-v2-clean-20newsgroups\20newsgroups --results experiments\results\graphtopic-paper-v2-clean-20newsgroups
.paper-venv\Scripts\python.exe -m experiments.paper --only lexical-20newsgroups
```

The adapter calls the official dimensionality-reduction and graph/community methods
with their published defaults and shared frozen embeddings. It reports the native
coverage because upstream Graph2Topic discards isolated nodes, communities smaller
than five, and communities beyond the requested count. For the complete-assignment
table it additionally applies the same declared nearest-centroid rule used for
BERTopic residual outliers. The comparison is limited to 20 Newsgroups: the upstream
implementation materializes a dense cross-half similarity matrix, which makes a
full AG News run impractical and would undermine the scalability comparison.

Advanced users may copy `local-paths.example.json` to the ignored `local-paths.json`
to use audited external artifacts. That disables automatic bootstrap and all paths
must already exist.

## Outputs and checkpoints

Working outputs, assignments, logs, caches, and `paper-checkpoint.json` are ignored by
Git. Completion checks inspect expected files, not merely checkpoint flags. The
checkpoint also records the complete validated environment.

Build the compact candidate report after all stages finish:

```bat
.paper-venv\Scripts\python.exe -m experiments.report build
```

The report contains artifact hashes, core metrics, all baselines, BERTopic, lexical
audits, ablations, and scaling summaries. After reviewing it and reconciling the
paper, promote it once:

```bat
.paper-venv\Scripts\python.exe -m experiments.report promote
```

Subsequent users can validate a fresh run with:

```bat
.paper-venv\Scripts\python.exe -m experiments.report compare
```

Scientific scores use declared numerical tolerances, graph counts use a 0.1% portable
tolerance, and identities/configuration are exact. Environment details and scaling
time/RSS are recorded but excluded from cross-machine pass/fail comparison.

Large datasets, embeddings, model caches, and assignments must never be committed.
