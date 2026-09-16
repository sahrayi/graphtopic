# Paper revision: protocol and execution

## Status

The existing `configs/paper.json` and `expected/reference-results.json` remain the
frozen main reference. Supplementary experiments use `configs/revision.json` and
write under ignored `experiments/results/paper-supplement-v1/`. Reviewed aggregate
results for independent counts, final-$k$ graph design, and lexical cleanup are
frozen separately under `experiments/expected/`. These results do not overwrite
the original reference report.

Completed diagnostics and table generation:

- Two LaTeX ablation tables generated from the frozen reference, including actual
  topic counts. The AG News graph-design rows have now been rerun at the final
  `k=20` setting and frozen in `expected/graph-design-k20-agnews.json`.
- Saved AG News NMF NPMI reconstructed using only the union of saved top words.
  All ten three/seven-topic records reproduce their stored metrics to `1e-12`.
  The portable diagnostic is `experiments/expected/nmf-diagnostic.json`.
- The misleading interpretation of high NMF coherence was corrected: seed 11 at
  three topics contains 104,438 / 6,265 / 16,897 documents, with NPMI
  -0.068020 / 0.745314 / 0.227783. The high-coherence small topic contains HTML/link
  boilerplate. The reported `.000` external-score SDs result from rounding.

Completed supplementary runs:

- Independent targets 20/4 on both full corpora, with GraphTopic, KMeans, NMF,
  LDA and BERTopic across five seeds. The checked aggregate is frozen in
  `expected/independent-counts.json`. GraphTopic realizes 18.2 and 2.8 mean topics;
  the manuscript therefore calls this independently anchored, not exact-count.
- Final-$k$ graph design on AG News is frozen in
  `expected/graph-design-k20-agnews.json`.
- Lexical-only HTML cleanup on both development corpora is frozen in
  `expected/lexical-html-v2.json`. Graph2Topic is omitted from this lexical rerun
  because its saved assignments were unavailable.

## Inputs and public reproducibility

Run commands from the repository root. No personal paths or pre-downloaded files
are required. Follow `experiments/README.md` to create `.paper-venv` from
`requirements/experiments-reference.txt` first. Preparation downloads the pinned
public dataset and model if absent, resumes incomplete embedding generation, and
validates source/encoder metadata and checksums before reuse. Canonical artifacts
live under `experiments/artifacts/graphtopic-paper-v3-fidelity-independent-bertopic/`.
The cache is an optimization, never a prerequisite. Generated data are ignored by Git.

Existing compatible copies can stay in that canonical directory. Do not copy
uncleaned 20 Newsgroups embeddings into the cleaned protocol or repair incompatible
metadata by changing its source fields. Cached graphs/candidates are keyed by input
embedding hash and graph settings. Timing intentionally does not reuse them.

List stages without starting calculations:

```powershell
.paper-venv/Scripts/python.exe -m experiments.revision --list
```

## 1. Independent target counts

```powershell
.paper-venv/Scripts/python.exe -m experiments.revision --stage robustness --dataset 20newsgroups
.paper-venv/Scripts/python.exe -m experiments.revision --stage robustness --dataset agnews
```

These commands prepare missing artifacts, reuse/build the graph, select GraphTopic
resolution by topic count only, then run GraphTopic, MiniBatchKMeans, NMF, LDA, and
BERTopic. Targets are 20 and 4: independent of GraphTopic, but derived from benchmark
class cardinalities, not fully label-free. A fixed 15-point resolution grid and
pilot seed 1729 are declared before scoring. Select the smallest absolute count
error, breaking ties by lower resolution; freeze it and evaluate seeds
11/23/37/51/71. Report actual topic counts: an exact match is not guaranteed.
Do not expand the grid after inspecting ARI/NMI. Pilot selection receives no labels.

Graph2Topic remains in its Python 3.9 environment described in the existing README.
After the 20NG robustness command has prepared the artifact, run this additional
comparison (PowerShell):

```powershell
$env:PYTHONHASHSEED = '0'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
.graph2topic-venv/Scripts/python.exe -m experiments.graph2topic --artifact experiments/artifacts/graphtopic-paper-v3-fidelity-independent-bertopic/20newsgroups --results experiments/results/paper-supplement-v1/independent --target-topics 20
```

## 2. Graph design at the final neighborhood size

```powershell
.paper-venv/Scripts/python.exe -m experiments.revision --stage graph-design --dataset agnews
```

This compares directed, mutual and union-max with `k=20` at both original AG News
resolutions and reuses compatible ANN candidates. The completed five-seed output
has been checked for all six graph/resolution combinations and incorporated into
the manuscript ablation table. Repeating the command uses its checkpoints.

## 3. Shared-subset TriTopic comparison

Inspection of the official [TriTopic 2.3.0 distribution](https://pypi.org/project/tritopic/2.3.0/)
found `ConsensusLeiden._compute_consensus` materializes a dense co-occurrence matrix
and further distance/linkage intermediates. One float64 matrix is 130.25 GB for
127,600 documents, excluding the other arrays. Full-corpus default consensus is
therefore outside the manuscript machine's memory budget. This is source inspection,
not a measured peak-memory result or a claim about every TriTopic version.

Use an independently sampled 5,000-document subset from each corpus, uniformly
without replacement using seed 1729; preserve sorted source row IDs. All six methods
(GraphTopic, TriTopic, BERTopic, MiniBatchKMeans, NMF, LDA) receive exactly these rows
and shared embeddings. Targets remain 20/4. Keep these results in a separate table;
never compare subset TriTopic numbers against full-corpus GraphTopic numbers.

Create a separate environment so installing TriTopic cannot change the frozen one:

```powershell
py -3.14 -m venv .tritopic-venv
.tritopic-venv/Scripts/python.exe -m pip install -r requirements/experiments-reference.txt -r requirements/tritopic.txt
.tritopic-venv/Scripts/python.exe -m pip install -e . --no-deps
.tritopic-venv/Scripts/python.exe -m experiments.tritopic_smoke
```

The 60-document synthetic smoke test exercises the real package and forbids internal
encoding when embeddings are supplied. This environment/API check must pass before
the scientific comparison; its runtime has not been validated in this revision.
If dependency resolution or the smoke test fails, retain the error and resolve it
before running the subsets. Do not modify consensus to get a passing result.

```powershell
.paper-venv/Scripts/python.exe -m experiments.revision --stage subset --dataset 20newsgroups --tritopic-python .tritopic-venv/Scripts/python.exe
.paper-venv/Scripts/python.exe -m experiments.revision --stage subset --dataset agnews --tritopic-python .tritopic-venv/Scripts/python.exe
```

The adapter preserves the official ten-run consensus and iterative-refinement
defaults, disables optional metadata input, uses native target-count handling,
and records the full package environment. It reports native coverage and complete
assignments separately; residual outliers use the common nearest-centroid extension.
It does not claim that the package's native output necessarily has 100% coverage.

## 4. Lexical cleanup

```powershell
.paper-venv/Scripts/python.exe -m experiments.revision --stage lexical --dataset 20newsgroups
.paper-venv/Scripts/python.exe -m experiments.revision --stage lexical --dataset agnews
```

The public route first builds missing canonical GraphTopic/classical/BERTopic
assignments. It may therefore be expensive on a fresh checkout. Graph2Topic is
audited too when its assignments are present; generate those using the existing
README before finalizing a table that includes Graph2Topic, then repeat this stage.

`html-entities-v2` restores recognizable numeric and named references whose leading
ampersand was lost in the source (notably `#39;` and `quot;`), decodes entities at
most three times, and strips HTML tags while
preserving content and ordinary numeric tokens. This cleanup affects only the
common lexical audit and the qualitative word lists, not encoder inputs or baseline
training. All saved assignments remain fixed. The new word lists, coherence,
diversity and per-topic sizes/scores are saved separately. Qualitative topic and
representative-document selection rules remain unchanged. The reviewed manuscript
values exclude Graph2Topic from the lexical-only comparison.

## 5. Full pipeline timing (run last)

```powershell
.paper-venv/Scripts/python.exe -m experiments.revision --stage timing --dataset agnews
.paper-venv/Scripts/python.exe -m experiments.revision --stage timing --dataset dbpedia14
```

These are intentionally expensive: three full encoding-and-pipeline measurements
per corpus. They preserve existing cached embeddings. Each records model loading,
encoding, normalization, ANN, graph, Leiden, lexical representation and total time.
Data download/read and imports are excluded. The first repeat includes first-use
initialization; model loading may include a download on a fresh system. Report
these conditions and separate first-use from subsequent measurements. Do not call
the result a completely cold-cache benchmark or compare its RSS directly against
the old embedding-only baseline. Representation uses the lexical cleanup above.

## 6. Collect and review

```powershell
.paper-venv/Scripts/python.exe -m experiments.revision_report
```

The report lists missing comparisons explicitly and never promotes the reference.
Review the supplementary comparisons, `ablations/`, `lexical/`, `qualitative/`,
and `timing/` outputs. Only then revise claims, main/appendix tables and limitations,
and promote a reviewed new reference. Keep historical reports.

No LaTeX compilation or PDF inspection is part of these commands.
