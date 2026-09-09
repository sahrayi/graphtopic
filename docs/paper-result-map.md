# Paper-result provenance map

This map prevents narrative claims from drifting away from executable evidence. The
paths below refer to sections in the compact output of `python -m experiments.report
build`.

| Paper evidence | Report source | Reproduction status |
| --- | --- | --- |
| Corpus sizes, labels, dimensions, encoder/source identity | `artifacts` | Direct |
| ANN candidate Recall@20\|50 and final retained Recall@20 | `core.*.fidelity_at_20` | Direct |
| Canonical topic counts, ARI, NMI | `core.*.resolutions` | Direct, five seeds |
| Sparse graph edges/connectivity | `core.*.graph` | Direct |
| KMeans, NMF, and LDA comparison | `baselines` | Direct, five seeds |
| BERTopic comparison and coverage | `bertopic` | Direct, five seeds, residual outliers forbidden |
| Official Graph2Topic comparison and native coverage | `graph2topic` | Direct on 20 Newsgroups, five seeds |
| Common-extractor NPMI and diversity | `lexical` | Direct, five seeds |
| Neighborhood/resolution sensitivity | `ablations.k-resolution` | Direct |
| Directed, mutual, union-max comparison | `ablations.graph-design` | Direct |
| Alternate encoder sensitivity | `ablations.encoder` | Direct |
| AG News and DBpedia14 scaling | `scaling_observational` | Direct; time/RSS hardware-specific |
| Exact versus ANN search timing at 10k/25k | `scaling_exact_comparison` | Direct; hardware-specific |
| DBpedia14 resolution path | `core.dbpedia14.resolutions` | Direct, five seeds |
| Exact-KNN versus ANN downstream partition equality | None | Unsupported; remove or limit to Recall@20 |
| Automatic-resolution selector numerical example | None | Unsupported; remove the numerical claim |
| Prespecified qualitative topic examples | `qualitative` | Direct; fixed topic/document rules |

The current public-data protocol uses 18,846 20 Newsgroups records and 630,000
DBpedia14 records. The LaTeX manuscript has been reconciled with the promoted
reference report for the corrected paper-v3 protocol, and its independent comparison
passes. Historical paper-v1 and paper-v2 values are no longer the current manuscript
evidence.
