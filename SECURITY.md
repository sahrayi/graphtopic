# Security policy

## Supported versions

Until the first stable release, security fixes are made only on the latest `main`
revision. Published support windows will be listed here after release.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's private
security-advisory reporting interface for this repository. Include affected versions,
impact, reproduction steps, and any suggested mitigation. You should receive an
acknowledgement within seven days.

GraphTopic processes user-supplied text and numeric artifacts. Treat pickle files and
other executable serialization formats as untrusted; the official experiment tools
use `allow_pickle=False` and do not require secrets.
