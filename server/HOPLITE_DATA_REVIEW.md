# Compatibility with the existing Hoplite database

Inspected read-only on 2026-11-02: `content/osa_db/hoplite.sqlite`, installed
`perch-hoplite==1.0.2` datatypes, SQLite implementation, and classifier data loader.
No source data was changed. Counts describe the observed database, not an audit
of the audio files or the USearch index.

## Observed schema and mapping

| Existing structure | Observed contents | Proposed server mapping |
|---|---|---|
| `deployments` | 12 rows; integer ID, name, project, latitude, longitude; unique name/project | Add `Deployment`; preserve source ID, grouping, and optional coordinates |
| `recordings` | 784 rows; integer ID, filename, datetime, deployment ID | `Recording` with separate server/source IDs, deployment reference, original filename, optional recorded time and content hash |
| `windows` | 19,098 rows; integer ID, recording ID, offsets | `Window`; preserve source ID and exact source offsets |
| `annotations` | 176 rows; integer ID, recording ID, offsets, label, integer label type, provenance | Historical annotation records with source attribution; link to exact matching windows where verified |
| `hoplite_metadata` | Audio-source configuration, model configuration, USearch configuration | Preserve raw source configuration and extract structured model/audio metadata |
| `usearch.index` | Embeddings stored outside SQLite, looked up using window ID | Leave in place initially; training bridge retrieves vectors by checked source window IDs |

`Deployment` was missing from the initial proposed structures. It matters for
project/site filtering, audio lookup, and future call-density analysis. Current
projects include `silencio`, `grillos`, and `SWIFT`. These are not separate
Hoplite database identity namespaces: one database contains all three.

Recordings use `filename`, not a hosted URL. For example,
`H23/H23_20230423_183500.WAV` is resolved using the deployment/project and
`audio_sources` configuration. Preserve case and relative path. All 784 current
recording datetimes are NULL; do not treat a parsed filename time as verified
source metadata.

Offsets are declared `FLOAT_LIST` and stored as binary values in SQLite. Use
Hoplite's converters or explicitly verified decoding, not JSON/string parsing.
The Python `Window` object additionally exposes an optional NumPy embedding;
that embedding is not a SQLite column.

## Labels are species plus label type

Hoplite separates the species code (`label`) from its observation state
(`label_type`). The server should preserve this distinction even though the
frontend calls its observation state `label`.

| Frontend | Hoplite label | Hoplite label type |
|---|---|---|
| `speciesId: DENEBR`, `label: yes` | `DENEBR` | `POSITIVE = 1` |
| `speciesId: DENEBR`, `label: no` | `DENEBR` | `NEGATIVE = 0` |
| `speciesId: DENEBR`, `label: unsure` | `DENEBR` | `UNCERTAIN = 2` |

Observed annotation counts:

| Code | Positive | Negative | Uncertain |
|---|---:|---:|---:|
| BOAROS | 34 | 19 | 0 |
| DENEBR | 44 | 3 | 0 |
| LEPINS | 7 | 16 | 0 |
| PHYVIT | 25 | 0 | 0 |
| SCIBOU | 28 | 0 | 0 |

Provenance values are `Fabrizio`, `Patricia`, and `testing`. Retain these strings
and source annotation IDs. There are no historical annotation timestamps,
submission IDs, or explicit correction relationships. Do not manufacture those
fields or infer authenticated identities from provenance.

The notebook's commented possible-label list is not the authoritative inventory:
the database includes `LEPINS` and `PHYVIT`, for example. Verify scientific-name
mappings against the project's species catalog before populating configuration.

## The important semantic mismatch: intervals versus windows

Hoplite annotations reference a recording and time interval, not `window_id`.
`get_window_annotations` retrieves strictly overlapping intervals in the same
recording. An annotation can therefore label multiple embedding windows.

All 176 current annotations match exactly one window by recording and offsets.
That makes a window-based frontend appropriate for the current data. Keep the
original interval in historical imports and fail or explicitly handle unmatched
or multiply matched intervals in future imports. Do not silently approximate
them with the nearest window.

The SQL overlap query uses strict inequalities. The installed
`Window.intersects` Python helper uses inclusive boundaries, while
`Annotation.intersects` uses strict boundaries. Use the actual classifier/SQL
retrieval semantics for compatibility tests, including intervals that only touch
at an endpoint.

## Training behavior needs an explicit choice

The installed `get_multihot_labels` counts positive and negative overlapping
annotations per species and uses:

```text
target = positive_count / (positive_count + negative_count)
mask = positive_count + negative_count > 0
```

Uncertain labels do not contribute to either count. Thus one positive and one
negative vote produces a target of 0.5, rather than selecting one annotator or
automatically adjudicating the conflict.

The current database has one recording/interval/species group with both positive
and negative labels across provenances. There are no such conflicts within a
single provenance. This means the distinction already matters to current data.

Recommended bridge behavior:

- Preserve separate annotator/source-attribution labels and correction history.
- Offer explicit selected-annotator or vote-fraction snapshot policies.
- Use vote fractions for a compatibility comparison with the existing loader.
- Replace a person's previous effective vote on correction; do not count revision
  history as additional votes. A correction to `unsure` removes their binary vote.
- Resolve how imported legacy provenance corresponds to a new local identity
  before combining their votes. Do not count the same historical label twice.

## Integrity observations

- No duplicate `(recording_id, offsets)` window groups were found.
- No duplicate `(recording_id, offsets, label, label_type, provenance)` annotation
  groups were found.
- Existing UNIQUE constraints include row ID for windows/annotations, so those
  constraints alone do not enforce semantic uniqueness. Keep importer checks and
  explicit server business keys.
- Window IDs have gaps: the maximum is 26,159 although there are 19,098 rows.
  Preserve IDs; do not renumber or assume contiguous arrays.
- `PRAGMA foreign_key_check` reports recording 694 referencing missing deployment
  12. Its filename is `SWIFT29_2021-04-14/SWIFT29_20210414_043000.wav`. It has no
  windows or annotations, so it does not block candidate export. Full recording
  migration must report/quarantine or explicitly repair it; do not silently map
  it to deployment 13 based on filename resemblance.
- The installed Hoplite implementation enables foreign keys on its own
  connections. A fresh raw SQLite connection defaults to foreign keys disabled;
  that setting alone does not prove Hoplite writes normally bypass enforcement.

## Structures that are intentionally new

Species display metadata, content hashes, classifier-run provenance, immutable
candidate sets, clip assets, issued batches, submission idempotency, annotation
events/revisions, and training snapshots have no equivalent in the current SQL
schema. They provide the application and reproducibility behavior needed by the
frontend; they do not need to become Hoplite tables.

Keep the initial server focused: export candidate windows and their metadata,
serve clips, and persist labels. Preserve enough source context to rejoin the
original embeddings. A full migration of every recording and embedding is not
required to demonstrate the local annotation loop.
