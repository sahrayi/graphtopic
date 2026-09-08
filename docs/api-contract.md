# GraphTopic API contract

Status: implemented API for version 0.1.0. This contract defines the
current API; it does not describe the legacy 0.0.2 API.

The scientific reference is the Method section of the current GraphTopic paper:
Semantic neighborhoods, Sparse graph and communities, Resolution as a topic path,
Topic representation, and Operational workflow. Earlier manuscripts are not the
reference. All project documentation is written in English.

## 1. Scope and canonical pipeline

Documents → sentence embeddings → L2 normalization → NNDescent candidates →
exact cosine rescoring → positive directed top-k → union-max weighted sparse graph →
Leiden RB-configuration partition → complete hard assignments → c-TF-IDF.

There is no dimensionality reduction in the canonical pipeline. Embedding, search,
graph construction, community detection and lexical representation are replaceable.
Replacement components must obey their boundary contracts, but may change the
scientific method. Their identity and effective settings are recorded.

Version one does not implement unseen-document transform, online updates,
probabilities, overlapping topics, a noise class, automatic resolution selection,
or a target topic count. Evaluation labels are not accepted by fit.

## 2. Public interface and defaults

```python
model = GraphTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    n_candidates=50,
    n_neighbors=20,
    resolution=1.0,
    random_state=42,
    neighbor_search=None,
    graph_model=None,
    community_model=None,
    representation_model=None,
    verbose=False,
)
model.fit(documents, embeddings=None, document_ids=None)  # returns self
topics = model.fit_transform(documents, embeddings=None, document_ids=None)
```

fit_transform returns list[int], not a tuple containing probabilities. Training
configuration belongs in the constructor. None selects a default component,
except embedding_model=None, which requires supplied embeddings.

| Setting | Default | Origin |
| --- | --- | --- |
| Encoder | all-MiniLM-L6-v2 | Paper experiment encoder; custom dimensions are unrestricted |
| Normalization | L2 | Paper method, including supplied embeddings |
| Search | NNDescent, cosine | Paper method |
| Non-self candidates | 50 | Paper budget, with explicit software counting convention below |
| Retained neighbors | 20 | Paper practical default |
| Graph | Weighted undirected union-max | Paper method |
| Detector | Leiden RBConfigurationVertexPartition | Paper objective, not CPM |
| Representation | Class-based TF-IDF | Paper formula |
| Resolution | 1.0 | Software starting point, not universal optimum |
| Seed | 42 | Software choice, not an experimental seed claim |
| Top words | 10 | Display choice consistent with lexical evaluation |
| Precision | Normalized float32 | Software choice |
| NNDescent details | 5 trees, 10 iterations, 1 worker, low_memory=True, delta=.001 | Explicit software settings |
| Leiden iterations | 2 | Frozen paper setting; users may pass -1 for convergence |

The backend API references are the official
[PyNNDescent API](https://pynndescent.readthedocs.io/en/stable/api.html) and
[leidenalg reference](https://leidenalg.readthedocs.io/en/stable/reference.html).
These references specify backend behavior, not paper experimental settings.

## 3. Documents and identities

- Documents must be a nonempty ordered sequence of strings: lists, tuples and
  one-dimensional arrays are accepted. Scalar strings, sets, generators, None,
  multidimensional arrays and non-string members are rejected.
- Empty, whitespace-only and duplicate documents are retained without merging.
- Original text reaches the encoder without lexical preprocessing.
- Optional document_ids must have length n and unique values, either all strings
  or all integers excluding bool. The default is 0..n-1.
- Internal vertex indices always mean input row positions. External IDs are an
  explicit mapping, never implicitly interpreted as adjacency indices.
- Document and ID sequences are copied into tuples; input mutation cannot reorder
  an existing result.

## 4. Embeddings

Explicit embeddings always take precedence over the configured encoder. No model
initialization, download or encode call occurs on that path. Sentence Transformers
is optional and imported lazily; importing GraphTopic does not import torch.

Embeddings must be a dense real numeric array-like of shape (n,d), d >= 1.
Memmaps are accepted. Sparse, complex, boolean and string arrays are rejected.
Rows must align with documents; semantic alignment cannot be inferred by the library.

NaN, infinity and zero-norm rows raise a ValueError identifying row positions.
Normalization uses a scale-stabilized float64 calculation and produces float32,
without modifying the caller's matrix. This also handles very large or small
finite magnitudes. embeddings_ is the normalized matrix, not the raw input.

Custom encoders implement encode(documents) -> dense matrix. Their internal
state may include a loaded model; they are not deep-copied. Reusing embeddings
across fits is explicit; automatic disk caching is not implemented.
The facade reuses its lazy encoder adapter across fits with the same model name.
This model-loading cache is independent of transactional fitted-result state.

## 5. Candidate search and top-k

Positive integer budgets exclude bool. When both default search and graph models
are used, n_candidates must be >= n_neighbors. Custom components own their budgets.

For small corpora, effective_candidates=min(requested,n-1) and
effective_k=min(requested,n-1); reductions issue warnings and are recorded.
Public candidates exclude self. The ANN adapter requests up to effective_candidates+1
raw entries, capped at n, and records that backend request. This convention must
be checked against experimental code when reproducing paper results.

Search returns exactly n one-dimensional integer candidate rows; ragged lengths
are allowed. Backend adapters remove padding. The canonical graph builder:

1. Validates indices (out-of-range or non-integer values are errors).
2. Removes self and duplicates.
3. Recomputes dot products of normalized embeddings, clipping roundoff to [-1,1].
4. Discards weights <= 0.
5. Sorts by descending score, then ascending row index, retaining at most effective_k.

Backend distances never define final canonical weights. Fewer valid candidates
produce fewer edges; no synthetic neighbors or hidden exact fallback fill the gap.
Diagnostics record shortages. ANN and exact search share the same graph builder.

Candidate storage is O(n × n_candidates); retained directed storage is
O(n × n_neighbors). Exact search still takes quadratic time, using bounded row
blocks rather than an n-by-n similarity allocation.

The paper explicitly excludes negative weights. Omitting stored zero weights is
a software sparse-storage convention; zero edges do not count as connectivity.

## 6. DocumentGraph

DocumentGraph is the interchange object: SciPy CSR adjacency plus document IDs.
igraph is internal to the default detector.

- Shape is exactly (n,n), including all isolated vertices.
- No self-loops, duplicate entries, negative weights or nonfinite weights.
- Stored weights must be strictly positive.
- Sparsity is symmetric; mirrored weights agree within rtol=1e-5, atol=1e-7.
  Invalid custom graphs are rejected, not silently symmetrized.
- Canonical weights are cosine and at most one. Custom graph weights may exceed
  one; they must not be falsely described as cosine.
- Vertex ordering and input identities cannot change.

Union-max includes an edge if either direction selects it and uses the maximum
score, never the sum. A final vertex degree can exceed k. There are at most n*k
directed arcs before symmetrization. CSR stores each undirected edge twice, so
n_edges=nnz/2. Backend conversion sends each upper-triangle edge once and explicitly
preserves n vertices.

Diagnostics include vertex/edge counts, isolates, connected components, min/mean/max
degree and, for the canonical builder, candidate shortages and effective k.
Directed custom graphs are outside this version's boundary contract.

Public adjacency and diagnostics access returns defensive copies; snapshots share
the internal validated graph. Private underscore attributes are not a public
mutation API.

## 7. Custom components and configuration precedence

Inheritance is optional. Structural protocols are available in graphtopic.protocols:

```python
EmbeddingModel.encode(documents) -> EmbeddingMatrix
NeighborSearch.search(embeddings) -> CandidateNeighbors
GraphModel.build(embeddings, candidates, *, document_ids) -> DocumentGraph
CommunityModel.fit_predict(graph) -> IntegerLabels
RepresentationModel.fit(documents, topics) -> self
RepresentationModel.get_topics() -> dict[int, list[tuple[str, float]]]
```

Components own constructor settings. Facade scalars configure only the corresponding
default component: n_candidates for search, n_neighbors for graph, resolution for
detector, random_state for default search and detector. Nondefault scalars alongside
replacement components produce warnings identifying the ignored setting.

Custom components may expose effective_config_ after execution. Without it,
effective settings are recorded as unknown, not guessed from facade defaults.
repr identifies selected component classes. Custom search only proposes candidates;
custom graph builders own their construction and weight semantics.

All non-encoder components must support deepcopy returning an independent object.
Work runs on independent copies, preserving user templates. Component errors keep
their original exception type and cause with a pipeline-stage note where executed.

## 8. Partitions and boundary cases

Detectors return exactly n nonnegative integer labels, excluding bool, with one
assignment per document. Missing, negative or malformed labels are errors.

Public topic IDs are contiguous 0..T-1, sorted by descending community size, breaking
ties by the smallest member row index. Raw-to-public mapping is recorded. This
normalizes presentation, not stochastic partitions or cross-resolution identities.

The default detector assigns isolated vertices singleton topics and optimizes all
remaining edges together; it does not change the global null model by optimizing
components separately. On an entirely edgeless graph, it returns n singletons with
a warning without invoking Leiden's zero-total-weight objective. For n=1, default
search and detector bypass their third-party backends and return topic 0.

Resolution must be finite and > 0. Seeds are None or integers in [0,2**32-1].

## 9. Lexical representation

Count document terms first, then aggregate by topic:

```text
n[t,c] = sum of term counts in topic c
L[c]   = sum_t n[t,c]
f[t]   = sum_c n[t,c]
A      = mean_c L[c]
w[t,c] = (n[t,c] / L[c]) * log(1 + A / max(f[t], 1))
```

This is not ordinary TfidfVectorizer. Vocabulary filtering occurs at the original
document level. The default CountVectorizer uses lowercase unigrams, token pattern
`(?u)\b\w\w+\b`, English stop words, min_df=5, max_df=.95 and max_features=20000.
Applying the paper's reported English lexical preprocessing to the library default
is an explicit software choice; other languages should configure a vectorizer.

Use CTFIDFRepresentation(vectorizer_model=..., ctfidf_model=..., top_n_words=...)
to customize. The default ClassTfidfTransformer is a small sklearn-style
transformer implemented with SciPy and scikit-learn primitives. Keeping it inside
GraphTopic avoids depending on the complete BERTopic package while preserving the
paper's floating-point mean class length. A replacement transformer must implement
fit_transform(class_counts) and return a finite, non-negative sparse matrix of the
same shape.
The common document-IDF lexical audit in the paper is not this default extractor.

Only positive terms in each topic are selected, sorted by descending score then
lexical term order. Word lists may be shorter than top_n_words. Empty class rows
have zero scores and remain in the mean A. An empty vocabulary preserves all topics
with empty word lists and a warning; invalid vectorizer parameters remain errors.
The default small-corpus min_df conflict is treated as an empty vocabulary, without
silently changing thresholds.

Custom representations return exactly all fitted integer topic keys, each mapping
to a list of string/finite-real-score pairs. Scores are not assumed to be probabilities.
Lexical changes cannot change embeddings, graph or partition.

## 10. Outputs and inspection

| Attribute | Value |
| --- | --- |
| documents_ | Tuple of input texts |
| document_ids_ | Tuple of IDs |
| embeddings_ | Read-only normalized array |
| graph_ | Validated DocumentGraph |
| topics_ | Defensive list of topic IDs in document order |
| topic_sizes_ | Defensive topic-to-count mapping, summing to n |
| topic_representations_ | Defensive topic-to-word/score mapping |
| result_ | Successful logical snapshot |

GraphTopicResult exposes documents, document_ids, topics (tuple), graph,
topic_words, topic_sizes and metadata. Large graphs are shared between snapshots.

get_topics() and get_topic(id) return copies; unknown IDs raise KeyError.
get_graph() returns a defensive DocumentGraph copy.
get_topic_info() returns a pandas DataFrame with Topic, Count, Name, Representation.
get_document_info() returns Document_ID, Document, Topic, Name in input order.
Names use the topic ID and up to three leading words, or topic_<id> for empty
representations. There is no Probability column and no duplicate labels_ state.

Metadata has three explicit levels. `versions` records runtime packages.
`base_fit` records embedding provenance, requested facade settings, search and graph
components, dimensions and graph diagnostics. `run` records the effective partition
resolution, graph-reuse flag, raw label mapping, detector and representation
components, and timings for that partition. Supplied embeddings have unknown encoder
provenance; the configured but unused encoder is not attributed to them.
Arbitrary custom metadata is inspectable but not promised to be JSON serializable.

## 11. Reuse

resolution_path([.1,.25,.5,1.0]) requires a successful fit and reuses its graph.
Default Leiden supports with_resolution(value) -> fresh detector. Custom detectors
must implement this capability or the path raises NotImplementedError.

Resolution lists must be nonempty, valid and unique; user order is preserved.
Each point starts independently with the configured seed, without warm starts.
Its `run.resolution` is the path value while `base_fit.requested.resolution` remains
the original fit request, so provenance and effective execution are not conflated.
path.results is an ordered tuple of snapshots; path.get_summary() reports
Resolution, Topics, MinSize, MedianSize, MaxSize. The original model is unchanged.

Paths are not hierarchies and do not guarantee monotonic topic counts, nested
partitions or matching topic IDs. They do not select an optimum. c-TF-IDF is fitted
for each partition; document count caching is a future optimization, not a current
performance guarantee.

update_topics(representation_model=...) refits only the lexical stage, returns self
and publishes a new snapshot without altering the graph or assignments. Calling it
without a replacement reuses the latest representation template. A later full fit
uses constructor configuration.

## 12. State, errors and reproducibility

fit and update_topics publish only fully validated successful results. Failed
refits preserve the previous snapshot; a failed first fit remains unfitted.
Inspection before fit raises sklearn.exceptions.NotFittedError. No partial
resolution path is returned on failure.

Invalid documents, IDs, embeddings, graph outputs and partitions raise ValueError.
Uncopyable components raise TypeError. Backend and custom execution errors preserve
their causes. Empty documents are retained; empty vocabularies and reduced budgets
warn. No documents are silently dropped.

Concurrent mutation of one model instance is unsupported. A fixed seed is not a
cross-version, cross-platform bitwise guarantee. Experimental reproduction requires
the same input rows, embeddings, backend settings, versions, preprocessing and seeds.

## 13. Acceptance criteria

Tests must cover precomputed encoder bypass, exact rescoring and candidate cleanup,
union-max weights and edge counts, isolates, hand-computed c-TF-IDF including empty
classes, component substitution and validation, graph reuse, immutable snapshots,
transactional failures, invalid inputs, and actual ANN/Leiden integration.
ANN fixture recall tests are not a claim of reproducing published Recall@20.

## 14. Scientific boundary

The canonical implementation follows the paper's method. It does not claim that
software defaults reproduce every table. Seed, starting resolution, numerical
precision, non-self candidate counting, zero omission, tie breaking, empty-graph
fallback and topic naming are documented software decisions. Compare with archived
experimental code before asserting numerical reproduction.
