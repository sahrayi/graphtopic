# Contributing

Use English for code, docstrings, documentation, tests and commit messages.

Install the checkout with `python -m pip install -e ".[dev]"`. Keep virtual
environments and generated outputs out of version control.

Run Ruff lint and formatting checks, then the full pytest suite with coverage.
Integration tests use actual NNDescent and Leiden; no pretrained model downloads
are required. Test algorithmic invariants and failure behavior, not incidental
label numbering from stochastic backends.

Preserve the boundaries in docs/api-contract.md. Component changes should include
mathematical fixtures or integration evidence, relevant documentation and a
CHANGELOG entry. Do not equate passing unit tests with reproducing the paper.

When work is interrupted, update docs/IMPLEMENTATION_STATUS.md with completed
steps, exact commands/results, known failures and the next action. Never record
credentials or tokens. CI validates Python 3.11 through 3.14 on Linux and Windows.

Publishing and pushing are separate release actions; see docs/releasing.md.
