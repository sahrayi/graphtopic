# Release checklist

No distribution is uploaded by the build or test commands. Publication requires
a separate release decision. Version 0.1.0 is currently a release candidate.

1. Complete the tests, lint and format checks from CONTRIBUTING.md.
2. Run CI on all declared Python/OS targets, including the core without embedding extra.
3. Review dependencies, CHANGELOG, README and the method/experimental validation boundary.
4. Confirm the release version in pyproject.toml, CITATION.cff, and the source fallback.
5. Build in a clean checkout with python -m build; run python -m twine check dist/*.
6. Install the wheel in a fresh environment outside the checkout and run the offline example.
7. Test the optional embedding extra with an actual model download and a real corpus.
8. Check wheel and sdist contents for tests/docs/license, no caches and no secrets.
9. After authorization, commit, push and tag the release; verify CI.
10. Publish to TestPyPI first if desired, install the exact distribution, then publish
    to PyPI using approved credentials or trusted publishing.
11. Verify installation from PyPI in a clean environment and update release notes.

The `Publish distributions` workflow uses GitHub trusted publishing. Configure
`testpypi` and `pypi` repository environments and matching trusted publishers before
dispatching it. A manual dispatch publishes only to TestPyPI; publishing a GitHub
Release publishes the built distributions to PyPI.

Do not overwrite a published version. The paper claim is tied to the promoted,
self-bootstrapped reference report and its locked environment.
Persistence/loading of full fitted Python objects and pretrained encoder download
tests are outside the current automated offline test suite.
