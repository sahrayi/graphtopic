# Controlled reference results

Run `python -m experiments.report build`, review the ignored candidate, reconcile the
paper, and then run `python -m experiments.report promote`. The resulting
`reference-results.json` is the compact oracle for public reproduction checks.
Historical or uncontrolled values must not be used as an oracle.
