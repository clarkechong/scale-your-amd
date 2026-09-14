# Artifact manifests

This directory stores lightweight manifests and schemas for the book's measured
claims. Large logs, HLO dumps, profiles, and checkpoints should live in immutable
release or object storage and be referenced by checksum.

Validate a run manifest with `artifacts/schema.json` before labeling a result
`[measured]`. Appendix F defines the required directory layout and claim-specific
files.

No current case study has a complete validated bundle.
