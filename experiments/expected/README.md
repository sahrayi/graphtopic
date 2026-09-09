# Controlled reference results

Run `python -m experiments.report build`, review the ignored candidate, reconcile the
paper, and then run `python -m experiments.report promote`. The resulting
`reference-results.json` is the compact oracle for public reproduction checks.
Historical or uncontrolled values must not be used as an oracle.

When `configs/paper.json` changes its `protocol_id`, the existing oracle remains
historical until every affected stage is rerun, the manuscript is reconciled, and a
reviewed candidate is promoted. A protocol mismatch must never be treated as a pass.
