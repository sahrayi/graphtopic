# Development validation

The development suite was validated locally on September 7, 2026. The controlled
paper suite completed on September 8, 2026 under Windows and Python 3.14.6 for the
GraphTopic 0.1.0 release candidate. This records observed checks, not unexecuted CI
results.

## Results

| Check | Result |
| --- | --- |
| Release-candidate pytest suite | 114 passed, 1 opt-in model-download test skipped |
| Real default encoder test | 1 passed separately with model download enabled |
| Branch-aware coverage | Passed the configured 85% release threshold |
| Expected warnings | 14 neighborhood-budget reductions plus 1 upstream SciPy deprecation warning |
| Ruff lint | Passed |
| Ruff format | Passed, 52 Python files |
| Git whitespace check | Passed |
| Isolated sdist then wheel build | Passed |
| Twine distribution metadata checks | Both passed |
| Wheel/sdist content inspection | Passed |
| Installed-wheel offline examples | Both passed outside checkout, Python isolated mode |
| Core without Sentence Transformers | Passed; no encoder dependency installed |
| Dependency consistency, pip check | Passed |
| Pinned runtime lower bounds on Python 3.11 | 110 passed, 1 optional test deselected |

The 115 collected tests exercise exact neighbor search against a brute-force oracle, candidate
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

## Remaining release checks

- GitHub's configured Linux/Windows and Python 3.11–3.13 matrix has not run yet.
- The pinned lower-bound combination passed locally on Python 3.11; its Linux CI job
  has not run on GitHub yet.
- All 17 self-bootstrapping paper stages completed in the locked reference environment.
  The compact report was promoted, its independent comparison passed, residual
  BERTopic outliers are forbidden, and the LaTeX manuscript was reconciled with the
  controlled public-data results.
- No GitHub push, package upload, or installation from a new PyPI release occurred.

These are release/integration follow-ups, not claims of completed validation.
