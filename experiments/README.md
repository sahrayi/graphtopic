# GraphTopic paper experiments

This directory is the self-bootstrapping, checkpointed protocol for every experiment
reported in the current GraphTopic paper. It uses the public `graphtopic` package in
`src/`; it is not a second implementation of the method.

## Reproducibility contract

The default runner requires no pre-downloaded corpora, embeddings, or model files. On
its first run it downloads the pinned sources and Sentence Transformer revisions,
creates embeddings on CPU, verifies their checksums, and caches every large artifact
under the ignored `experiments/artifacts/` directory. Later runs reuse valid files and
resume an interrupted embedding batch or experiment stage.

Paper-quality runs use fixed seeds, exact dependency versions, CPU embeddings,
deterministic PyTorch operations, one ANN worker, one classical-baseline worker, and
one numerical-library thread. The runner refuses to start work in a mismatched Python
or package environment. This contract targets the reported quality metrics in the
locked environment. Runtime and peak memory are hardware-specific; floating-point
files are not promised to be byte-identical across operating systems and CPU models.

The scaling stages intentionally use the available processors. They preserve their
sample and algorithm seeds, but timing and memory are descriptive measurements.

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
ablations, scaling, and lexical audits.

## Inputs and provenance

`configs/paper.json` pins the Hugging Face Parquet filenames and repository revisions,
Sentence Transformer commits, split order, text construction, dimensions, row/class
counts, seeds, graph settings, resolutions, and baseline settings. Parquet files are
downloaded directly through Hugging Face Hub, avoiding the Python-version-sensitive
`datasets` fingerprint/cache layer. 20 Newsgroups is fetched by the locked
scikit-learn release with `shuffle=False`.

Every generated artifact contains compressed documents, labels, float32 embeddings,
and metadata with source/model identity plus SHA-256 checksums. Corrupt or incomplete
caches fail validation rather than being silently reused.

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
