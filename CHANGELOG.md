# Changelog

## 0.1.1 — 2026-09-08

- Update the public README for installation from PyPI and the supported Python versions.
- Add PyPI, Python, CI, and license status badges.
- Make repeated publication workflow runs safely ignore existing distributions.

## 0.1.0 — 2026-09-08

- Replace the Graphion-based legacy implementation with the current paper pipeline.
- Accept precomputed embeddings or lazy optional Sentence Transformers encoders.
- Add exact and NNDescent candidate search, exact cosine top-k union-max graphs,
  weighted Leiden RB-configuration communities and sparse class-based TF-IDF.
- Define structural component protocols, validation and stable document identities.
- Add transactional fit/update, defensive result snapshots, inspection tables and
  resolution paths that share a fixed graph.
- Separate the class-weighting formula into a replaceable sklearn-style
  `ClassTfidfTransformer` without adding BERTopic as a runtime dependency.
- Separate immutable base-fit provenance from per-partition run metadata.
- Add unit/integration/optional-embedding test groups and a paper-artifact
  validation CLI with prespecified tolerance checks.
- Add offline examples, contract tests, backend integration tests and CI.
- Add a self-bootstrapping paper suite with pinned public data and encoders,
  checkpointed experiments, immutable provenance, and a promoted reference report.
- Align the implementation and manuscript with the controlled public-data results;
  keep runtime and memory explicitly hardware-specific.
- Add GitHub templates, CodeQL, Dependabot, citation, and security metadata.
- Preserve legacy 0.0.2 under the legacy-v0.0.2 Git tag.

This is an incompatible redesign and is not migration-compatible with legacy 0.0.2.
