# GraphTopic

[![PyPI](https://img.shields.io/pypi/v/graphtopic)](https://pypi.org/project/graphtopic/)
[![Python](https://img.shields.io/pypi/pyversions/graphtopic)](https://pypi.org/project/graphtopic/)
[![CI](https://github.com/sahrayi/graphtopic/actions/workflows/ci.yml/badge.svg)](https://github.com/sahrayi/graphtopic/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Discover topics by partitioning a sparse semantic document graph.

GraphTopic encodes documents (or accepts your embeddings), retrieves approximate
candidate neighbors, recomputes cosine weights, builds a union-max graph, partitions
it with Leiden, and explains the resulting communities with class-based TF-IDF.

Version 0.1.2 is the official implementation of the current GraphTopic paper protocol.
The earlier 0.0.2 implementation remains available under the Git tag `legacy-v0.0.2`.

## Installation

GraphTopic requires Python 3.11 or newer and is tested on Python 3.11–3.14.

Install the library from PyPI:

```bash
python -m pip install graphtopic
```

The core installation accepts precomputed embeddings. To also encode documents
with Sentence Transformers:

```bash
python -m pip install "graphtopic[embedding]"
```

To install from a source checkout:

```bash
python -m venv .venv
# Activate the environment using your shell's activation command.
python -m pip install .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

## Quick start: precomputed embeddings, no model download

This complete offline example uses illustrative two-dimensional embeddings. For
real data, pass one semantic embedding row per document in exactly the same order.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from graphtopic import CTFIDFRepresentation, ExactNeighborSearch, GraphTopic

documents = [
    "apple fruit red",
    "apple fruit green",
    "football goal team",
    "football goal match",
]
embeddings = np.array(
    [
        [1.0, 0.05],
        [1.0, 0.10],
        [0.05, 1.0],
        [0.10, 1.0],
    ]
)

model = GraphTopic(
    embedding_model=None,
    neighbor_search=ExactNeighborSearch(n_candidates=3),
    n_neighbors=1,
    representation_model=CTFIDFRepresentation(CountVectorizer()),
)
topics = model.fit_transform(documents, embeddings=embeddings)
print(topics)
print(model.get_topic_info())
print(model.get_document_info())
print(model.graph_.diagnostics)
```

The small example explicitly changes search, degree and lexical thresholds.
Canonical defaults are NNDescent with 50 non-self candidates and 20 retained
neighbors. Leiden uses two optimization iterations, matching the frozen paper
experiments; pass a custom `LeidenDetector(n_iterations=-1)` to optimize until no
further improvement. Tiny corpora clamp neighborhood budgets with warnings. The default English
vectorizer uses min_df=5, max_df=.95 and at most 20,000 features; use the example's
vectorizer for tiny corpora or configure it for your language.

## Encode documents with a model

After installing the embedding extra:

```python
from graphtopic import GraphTopic

model = GraphTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    resolution=1.0,
    random_state=42,
)
# documents is your ordered sequence of text strings.
topics = model.fit_transform(documents)
```

The encoder loads only when needed and may download model files on first use.
You can pass any object with `encode(documents)`, including a configured Sentence
Transformer. To control device and batch size, use `SentenceTransformerEmbedding`.

Supplying `embeddings=...` always bypasses the encoder, even if one is configured.
Save and reuse your own embedding arrays to avoid recomputation. Inputs are copied
for normalization; your array is not modified.

## Inspect and reuse

```python
words = model.get_topics()  # topic -> [(word, score), ...]
one_topic = model.get_topic(0)
graph = model.get_graph()  # defensive graph copy
adjacency = graph.adjacency  # writable SciPy CSR copy
result = model.result_  # snapshot of this fit

path = model.resolution_path([0.1, 0.5, 1.0, 2.0])
print(path.get_summary())
```

The path reuses the graph and embeddings without running search again. It does
not change the model's original result and is not a nested topic hierarchy.
A higher resolution generally favors finer partitions, but no universal optimum
or monotonic topic-count guarantee is supplied.

```python
model.update_topics(
    representation_model=CTFIDFRepresentation(
        CountVectorizer(ngram_range=(1, 2), min_df=1),
        top_n_words=5,
    )
)
```

This changes only lexical descriptions. It does not rebuild or repartition the graph.
`CTFIDFRepresentation` also accepts a replaceable `ctfidf_model`. GraphTopic's
default `ClassTfidfTransformer` uses SciPy and scikit-learn sparse operations and
implements the formula in the paper without requiring BERTopic as a dependency.

## Replace pipeline components

```python
from graphtopic import LeidenDetector, NNDescentSearch, UnionMaxGraph

model = GraphTopic(
    embedding_model=my_encoder,
    neighbor_search=NNDescentSearch(n_candidates=100, random_state=7, n_jobs=1),
    graph_model=UnionMaxGraph(n_neighbors=30),
    community_model=LeidenDetector(resolution=0.3, random_state=7),
    representation_model=my_representation,
)
```

Custom classes need only implement the methods in
[the component guide](docs/components.md); inheritance is optional. Supplied
components own their settings. Facade scalars configure only default components.
Replacing a component can change the method's scientific behavior.

## What the model returns

- `fit()` returns the fitted model.
- `fit_transform()` returns a list of integer topic assignments.
- Every document belongs to exactly one topic, including isolated vertices.
- Word scores are lexical weights, not membership probabilities.
- There is no noise topic, unseen-document `transform()`, automatic resolution
  selection, or probabilistic topic mixture in this version.

The method follows the paper, but resolution=1.0 and seed=42 are software starting
defaults, not promises to reproduce its result tables. See the
[API contract](docs/api-contract.md) for numerical, graph, state and boundary rules.

## Development and validation

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest --cov=graphtopic --cov-report=term-missing
python -m build
python -m twine check dist/*
```

The first ANN run may take longer due to Numba compilation. Tests use synthetic
inputs without downloading embedding models; real ANN and Leiden integration tests
run by default. The optional encoder adapter is tested with a lightweight fake model.

See [CONTRIBUTING](CONTRIBUTING.md), [release checks](docs/releasing.md),
[validation evidence](docs/VALIDATION.md), and [examples](examples).
The paper suite is self-bootstrapping and resumable: it downloads pinned public data
and model revisions, creates and verifies local embeddings, and runs against a locked
numerical environment. Previous uncontrolled partial outputs are not treated as
reference results. See [paper experiments](experiments/README.md) for the exact setup,
scope of reproducibility, and execution command.

## License

MIT. See [LICENSE](LICENSE).
