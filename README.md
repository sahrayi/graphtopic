# GraphTopic

**Graph-based topic modeling built on semantic graphs.**

GraphTopic is a Python library for discovering document topics using graph-based community detection instead of traditional clustering algorithms.

It is built on top of **Graphion**, a modular graph processing framework, allowing every stage of the topic modeling pipeline to be customized.

---

## Features

- Graph-based topic discovery
- Pluggable embedding models
- Pluggable graph construction
- Pluggable community detection
- Pluggable topic representation
- Simple sklearn-like API
- Built on Graphion

---

## Installation

Install the core library:

```bash
pip install graphtopic
```

To use the built-in SentenceTransformer embedding model:

```bash
pip install "graphtopic[embedding]"
```

---

## Quick Start

```python
from graphtopic import GraphTopic
from graphtopic.embedding_models import (
    SentenceTransformerEmbedding,
)

documents = [
    "Apple released a new iPhone.",
    "Samsung announced a new Galaxy phone.",
    "Barcelona won the football match.",
    "Real Madrid signed a new player.",
]

model = GraphTopic(
    embedding_model=SentenceTransformerEmbedding(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
)

topics = model.fit_transform(documents)

for topic in model.get_topic_info():
    print(topic)
```

---

## Pipeline

GraphTopic follows a modular processing pipeline:

```

Documents
↓
Embeddings
↓
FeatureSet
↓
Relation Graph
↓
Community Detection
↓
Topic Representation
↓
Topic Information

```

Each stage can be replaced with custom implementations.

---

## Custom Components

GraphTopic is designed around interchangeable components.

You can replace:

- Embedding Model
- Reducer
- Relation Builder
- Graph Builder
- Graph Refiner
- Community Detector
- Partition Refiner
- Representation Model

This makes it easy to experiment with different graph construction and topic discovery strategies.

---

## Built on Graphion

GraphTopic uses Graphion as its graph processing backend.

Graphion provides reusable components for:

- similarity computation
- graph construction
- graph refinement
- community detection
- graph processing pipelines

GraphTopic focuses only on topic modeling while delegating graph operations to Graphion.

---

## Requirements

- Python 3.11+
- NumPy
- scikit-learn
- Graphion

---

## Project Status

GraphTopic is currently in **Pre-Alpha (0.0.x)**.

The public API is expected to evolve as new functionality is added.

---

## License

MIT License