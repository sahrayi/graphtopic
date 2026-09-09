# Development validation

The development suite was validated locally on September 7, 2026. The first controlled
paper suite completed on September 8, 2026 under Windows and Python 3.14.6 using the
scientific pipeline later released unchanged in GraphTopic 0.1.1. On September 9,
2026, the paper-v2 protocol completed after removing headers, footers, and quoted
text from 20 Newsgroups. Its reconciled report was promoted and independently
compared without failures. This file records observed checks, not unexecuted results.

## Results

| Check | Result |
| --- | --- |
| Current full pytest suite | 120 passed, 1 opt-in model-download test skipped |
| Real default encoder test | 1 passed separately with model download enabled |
| Branch-aware coverage | Passed the configured 85% release threshold |
| Expected warnings | 14 neighborhood-budget reductions plus 1 upstream SciPy deprecation warning |
| Ruff lint | Passed |
| Ruff format | Passed, 54 Python files |
| Git whitespace check | Passed |
| Isolated sdist then wheel build | Passed |
| Twine distribution metadata checks | Both passed |
| Wheel/sdist content inspection | Passed |
| Installed-wheel offline examples | Both passed outside checkout, Python isolated mode |
| Core without Sentence Transformers | Passed; no encoder dependency installed |
| Dependency consistency, pip check | Passed |
| Pinned runtime lower bounds on Python 3.11 | 110 passed, 1 optional test deselected |

The 121 collected tests exercise exact neighbor search against a brute-force oracle, candidate
cleanup and rescoring, max rather than sum symmetrization, sparse graph validation,
isolated nodes, real weighted Leiden, real NNDescent candidate fidelity, hand-derived
c-TF-IDF values, custom components, optional encoder behavior, defensive snapshots,
failed refits and graph reuse across resolutions. The optional encoder adapter is
tested both with a fake Sentence Transformer in the default suite and with the
downloaded `all-MiniLM-L6-v2` model in an opt-in test.

Tests are grouped into pure unit/contract tests, core backend integration tests,
and an opt-in `optional_embedding` test that downloads the real default encoder. They also
exercise the artifact loader, atomic checkpoints, external and lexical metrics, exact ANN
recall, graph ablations, and a real resumable NNDescent/Leiden experiment run.
This makes `pytest -m "not integration and not optional_embedding"` a meaningful
backend-independent development check.

The first run exposed that PyNNDescent's Numba kernels require writable input.
The adapter now passes a private writable copy and keeps user inputs protected.
The final full suite includes regression coverage for that integration path.

## Observed dependency versions

NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.0, pandas 3.0.5,
PyNNDescent 0.6.0, igraph 1.0.0, leidenalg 0.12.0, Numba 0.67.0,
pytest 9.1.1. Runtime result metadata records installed versions for each fit.

## Reproduce the checks

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest --cov=graphtopic --cov-report=term-missing
python examples/precomputed.py
python examples/custom_components.py
python -m build
python -m twine check dist/*
python -m pip check
```

Install the wheel into a separate virtual environment and run both offline examples
with `python -I` to verify the installed package rather than the editable source.
The CI workflow performs wheel smoke checks after uninstalling the editable package.

## Published release and current paper follow-up

- GraphTopic 0.1.2 is published on GitHub and PyPI. Its configured release checks,
  package upload, clean PyPI installation, and installed-package example were completed.
- All 17 paper-v1 self-bootstrapping stages completed in the locked reference environment.
  The compact report was promoted, its independent comparison passed, residual
  BERTopic outliers are forbidden, and the LaTeX manuscript was reconciled with the
  controlled public-data results.
- All 23 paper-v2 runner stages completed. Unchanged controlled AG News and DBpedia14
  artifacts were reused only after source/model validation; every affected 20
  Newsgroups stage was rerun from cleaned text.
- Ten official Graph2Topic 2.0 runs completed in an isolated Python 3.9.13 environment.
  The shared lexical audit then processed their complete assignments.
- The paper-v2 candidate includes tie-aware and strict ANN recall, five-seed uncertainty,
  exact-search timings, and prespecified qualitative examples. It was reconciled with
  the LaTeX source, promoted, and `experiments.report compare` passed with no failures.

The published package validation and paper experimental validation are separate
records; neither is used as a substitute for the other.

## Validated paper-v3 protocol correction

On September 9, 2026, the eight affected stages were rerun under the locked reference
environment. The protocol distinguishes Candidate Recall@20|50 from Final Retained
Recall@20, independently fits BERTopic's natural partition before each direct target
reduction, and locks encoder maximum sequence lengths. Unchanged AG News embeddings
were reused only after checksum and provenance validation; cleaned 20 Newsgroups
embeddings were regenerated because the available archived artifact predated metadata
removal. The refreshed aggregate was reconciled with the LaTeX manuscript, promoted,
and independently compared without failures. The promoted paper-v3 report is now the
current manuscript oracle; paper-v2 remains a historical record.

More specifically, the primary cleaned 20 Newsgroups embeddings were regenerated in
paper-v3, whereas the AG News embedding file was carried forward byte-for-byte after
source, encoder revision, and all artifact checksums were verified. DBpedia14 and the
alternate-encoder 20 Newsgroups experiment were not rerun because neither their inputs
nor downstream computations changed; their validated paper-v2 aggregate sections were
retained. For carried or retained artifacts, the previously implicit maximum sequence
lengths (256 for the pinned primary model and 128 for the pinned alternate model) were
verified from the pinned model configurations and then recorded explicitly in the
paper-v3 metadata. This metadata extension does not represent fresh embedding runs.
