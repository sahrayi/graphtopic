# Custom pipeline components

Use the protocols in graphtopic.protocols as typing aids. Inheritance is optional.

| Component | Required method | Output |
| --- | --- | --- |
| Encoder | encode(documents) | Dense real array, n rows |
| Search | search(normalized_embeddings) | n integer candidate rows |
| Graph | build(embeddings, candidates, *, document_ids) | DocumentGraph |
| Detector | fit_predict(graph) | n nonnegative integer labels |
| Representation | fit(documents, topics), get_topics() | Exactly all topic keys and word-score lists |

Within the default lexical component, `ClassTfidfTransformer` is independently
replaceable through `CTFIDFRepresentation(ctfidf_model=...)`. It receives a sparse
topic-term count matrix and must return a sparse matrix with the same shape.

The library copies search, graph, detector and representation templates before use.
Their deepcopy must return independent objects. Encoders are not copied.
Inputs are read-only by contract. Documents are tuples; normalized embedding
arrays are read-only. Do not modify private attributes of DocumentGraph.

For resolution_path, implement with_resolution(value) on the detector to return a
fresh configured detector. Arbitrary clustering backends without this concept
remain usable for fit but cannot provide a resolution path.

A graph component must return an undirected positive sparse graph, preserve IDs,
and retain isolated vertices. Directed or signed graph backends require a future
expansion of this boundary. A custom graph can use weights other than cosine.

## Minimal example

```python
import numpy as np
from graphtopic import GraphTopic


class AllCandidates:
    def search(self, embeddings):
        # Educational only: quadratic storage, not suitable for large corpora.
        n = len(embeddings)
        self.effective_config_ = {"algorithm": "all_candidates"}
        return [np.delete(np.arange(n), i) for i in range(n)]


class FirstWords:
    def fit(self, documents, topics):
        self.words = {}
        for text, topic in zip(documents, topics, strict=True):
            self.words.setdefault(topic, [])
            if not self.words[topic] and text.split():
                self.words[topic] = [(text.split()[0], 1.0)]
        return self

    def get_topics(self):
        return self.words


model = GraphTopic(
    embedding_model=None,
    neighbor_search=AllCandidates(),
    representation_model=FirstWords(),
)
```

Components may publish effective_config_ to describe actual execution settings.
Without it, metadata reports unknown settings. Do not treat facade defaults as
settings of replacement components.

## Errors and state

Exceptions retain their type and include a pipeline-stage note. Invalid outputs
are rejected rather than repaired silently. Failed fit, refit or update_topics
does not publish a partially valid result.

The default detector always makes isolates singletons. Custom detectors may assign
them differently as long as every input row receives exactly one nonnegative label.
Topic IDs are normalized by size and first-member position before representation.

## Performance

Avoid dense n-by-n graphs and Python objects per edge at corpus scale. Use numeric
sparse arrays. Public graph.adjacency returns a copy to preserve snapshots;
built-in components use the internal graph arrays to avoid unnecessary copies.
External components should obtain one adjacency copy per operation, not per edge.
