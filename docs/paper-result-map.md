# Paper-result provenance map

This map prevents narrative claims from drifting away from executable evidence. The
paths below refer to sections in the compact output of `python -m experiments.report
build`.

| Paper evidence | Report source | Reproduction status |
| --- | --- | --- |
| Corpus sizes, labels, dimensions, encoder/source identity | `artifacts` | Direct |
| ANN Recall@20 | `core.*.recall_at_20` | Direct |
| Canonical topic counts, ARI, NMI | `core.*.resolutions` | Direct, five seeds |
| Sparse graph edges/connectivity | `core.*.graph` | Direct |
| KMeans, NMF, and LDA comparison | `baselines` | Direct, five seeds |
| BERTopic comparison and coverage | `bertopic` | Direct, five seeds, residual outliers forbidden |
| Common-extractor NPMI and diversity | `lexical` | Direct, five seeds |
| Neighborhood/resolution sensitivity | `ablations.k-resolution` | Direct |
| Directed, mutual, union-max comparison | `ablations.graph-design` | Direct |
| Alternate encoder sensitivity | `ablations.encoder` | Direct |
| AG News and DBpedia14 scaling | `scaling_observational` | Direct; time/RSS hardware-specific |
| DBpedia14 resolution path | `core.dbpedia14.resolutions` | Direct, five seeds |
| Exact-KNN versus ANN downstream partition equality | None | Unsupported; remove or limit to Recall@20 |
| Automatic-resolution selector numerical example | None | Unsupported; remove the numerical claim |
| Selected qualitative topic examples | Raw lexical records | Must be regenerated for the new matched topic counts |

The current public-data protocol uses 18,846 20 Newsgroups records and 630,000
DBpedia14 records. The LaTeX manuscript has been reconciled with the promoted
reference report; historical artifact sizes, granularities, and table values are not
part of the 0.1.1 evidence.
