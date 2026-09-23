# Ordered split inventories

The eight JSONL files contain only image IDs, original-image SHA256, caption counts and hashes of ordered caption lists. No raw images or caption text are redistributed. See [index.json](index.json) for exact sources and exclusions.

All splits are custom. Historical `sealed_test` filenames are retained for identity; these test sets were already observed in round one and are regression tests in this release. DOCCI never enters training or model selection.

Run `python reproduce.py --stage prepare` from the repository root to reconstruct and verify all 14,089 records. Caption hashes are SHA256 of UTF-8 `json.dumps(captions, ensure_ascii=False, separators=(',', ':'))`, with no trailing newline. Preserve row and caption order, whitespace and original image bytes.

Exact-byte duplicate checks do not rule out all near duplicates or pretrained-model exposure. See [attribution](../ATTRIBUTION.md) and [data provenance](../research/DATA_SOURCES.md).
