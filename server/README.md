# Local annotation server

FastAPI with a four-table SQLite store. Hoplite is read-only input; embeddings
remain in its USearch index. The frontend's ServerProvider fetch implementation,
authentication, BigQuery, and annotation-to-training snapshots are later work.

## Start

From `C:\agile`:

```powershell
.venv/Scripts/python.exe -m pip install -r server/requirements.txt
.venv/Scripts/python.exe -m uvicorn server.app.main:create_app --factory --host 127.0.0.1 --port 8080
```

Open <http://127.0.0.1:8080/docs>. The HTTP runtime does not need Hoplite, NumPy,
TensorFlow, or Google credentials. The exporter uses the research environment.

Optional configuration, set before startup:

```powershell
$env:OSA_SERVER_LOCAL = 'C:\agile\server\local'
$env:OSA_ANNOTATOR = 'your-local-name'
$env:OSA_AUDIO_ROOTS = '{"silencio":"C:/agile/data/audio","grillos":"C:/agile/data/grillos","SWIFT":"C:/agile/data/SWIFT"}'
```

Those audio roots are the defaults. `recordings.file_path` is relative to the
configured project's root, matching Hoplite filenames. Original WAV files must
remain accessible. Missing files return 404. Absolute paths, traversal, and
symlinks escaping the audio root are rejected. Paths come from registered records;
clients cannot request arbitrary filesystem paths.

Each process uses one configured annotator (default `local-development`), which
is local attribution rather than authentication. Bind to loopback. CORS permits
`http://localhost:8000` and `http://127.0.0.1:8000`.

## Data structures

| Table | One row represents |
|---|---|
| `recordings` | Original file path, dataset name, and Hoplite source recording ID |
| `windows` | Recording interval and explicit Hoplite source window ID |
| `candidates` | Window selected for review for a species: only `id`, `window_id`, `species` |
| `annotations` | One answer, with person/species/window, submission ID and one server timestamp |

There are no separate clip files, asset fields, assignments, candidate-set tables,
or effective-label tables. Windows contain only identity, recording reference,
source window ID, and start/end seconds. `source_window_id` preserves the exact
Hoplite embedding key for future training joins. Server IDs scope source IDs to
the corpus, e.g. `osa-local:window:123`.

`numAnnotations` is computed with an indexed SQL count per window/species and
returned in the candidate response; it is not a stored counter. It counts answer
rows, including corrections. A distinct-reviewer count would use `COUNT(DISTINCT
annotator)` instead. Current training votes must use the latest row per
window/species/annotator, ordered by insertion (`rowid`), not all historical rows.

The indexes are lookup shortcuts, not additional tables: `candidate_queue` filters
species queues; `annotation_latest` supports window/species/person lookups and
annotation counts; `annotation_submission` supports retry checks and saved counts.
They cost a little storage and write work but avoid scanning the whole table.

## Export and import

The optional cell after boundary selection in `03_Classify.ipynb` exports the
selected windows. Alternatively:

```powershell
.venv/Scripts/python.exe -m server.tools.export_candidates --source-dataset-id osa-local --species DENEBR --count 15 --output server/local/bundles/new-selection
.venv/Scripts/python.exe -m server.tools.import_candidates server/local/bundles/new-selection/manifest.json
```

Output directories must be new. Keep `sourceDatasetId` stable for the same corpus.
The exporter recomputes signed logits and orders by distance to the target, with
source window ID as tie breaker. The saved classifier `.pt` is JSON, not pickle.

Bundle schema version 2 is metadata-only: `manifest.json` without exported WAVs.
Selection scores and classifier provenance remain in this research artifact and
are not stored in the application database. Version 1 clip bundles must be
converted or re-exported. `server/local/bundles/denebr-recordings/manifest.json`
contains the existing 15 real DENEBR candidates in the new format.

Import checks configured source files and window durations before writing. Source
identity conflicts reject the entire transaction. Candidates are unique by
window/species, so repeated selections or imports do not duplicate them or reset
reviews. New export provenance does not create another candidate row. Existing
source files are not hashed or copied.

## HTTP contract and playback

Public JSON uses camelCase. Species codes/display names come from
`config/species.json`; reference clip lists are currently empty.

| Request | Result |
|---|---|
| `GET /health` | Readiness and database schema version 5 |
| `GET /api/species` | Canonical species and display names |
| `GET /api/species/{id}/references` | Target and comparison species with `clips` |
| `GET /api/species/{id}/candidates` | `{batchId, clips}` with up to 15 unreviewed windows |
| `POST /api/annotations` | Atomic `{submissionId, savedCount}` acknowledgement |
| `GET /api/submissions/{id}` | Original acknowledgement, or 404 |
| `GET /media/recordings/{id}` | Original full WAV, supporting HTTP byte ranges |

Candidate clips include `id`, source `recordingId`, `offsets`, `uri`, `datasetName`,
`audioUrl`, `startSeconds`, `endSeconds`, and `numAnnotations`. `audioUrl` identifies
the full recording; start/end match offsets. The frontend bounds playback with
Play/Pause controls; standalone reference/demo audio retains native controls.
Boundaries are approximate, and the full recording is accessible through its URL.
The removed `/media/clips` endpoint is not supported.

Review lists are queries in insertion order. Refresh after saving returns remaining
unreviewed windows. `unsure` counts as reviewed. No work returns
`{ "batchId": null, "clips": [] }`. `batchId` identifies a species/annotator context,
not a fixed assignment or authentication credential. Corrections remain allowed.

Saves include `submissionId`, `speciesId`, `batchId`, UTC `createdAt` (accepted for
frontend compatibility, not persisted), and
`annotations`: each has `windowId`, `label`, and the original source metadata.
Labels are `yes`, `no`, or `unsure`. Unanswered windows are omitted. The server
checks candidate membership, source fields, species, and configured annotator.

Each new save appends individual answer rows with one server UTC `created_at`. Exact retries with the same ID and
answers create no rows and return the original acknowledgement, including after
restart. Array/property order and client timestamps are ignored. Changed answers,
source metadata, species, review context, or owners return 409. Stored rows are
compared directly; counts come from SQL, without payload hashes or saved sizes.
All answers commit atomically. SQLite errors return 503; retry
with the same ID. Correction to unsure supersedes the earlier binary answer.

## Storage and tests

The rewritten `app/migrations/001_initial.sql` creates database version 5. Older
versions are rejected without resetting them. Back up old stores with SQLite's
backup API before rebuilding from bundles. Do not copy only the main file while
WAL writers run. Existing real annotations need explicit conversion, not deletion.
Generated databases, backups, bundles and caches are ignored by git.

```powershell
$env:OSA_REAL_BUNDLE = 'server/local/bundles/denebr-recordings/manifest.json'
.venv/Scripts/python.exe -m unittest discover -s server/tests -v
```

Tests cover minimal table/column structure, computed annotation counts, source
identity, metadata-only export, import rollback, recording ranges and missing
files, path safety, queue advancement, retries/concurrency, corrections and
annotator isolation. Real-corpus acceptance verifies scores/source IDs and full
recording range responses; synthetic labels stay in temporary stores.

Next: connect frontend fetches, then export versioned annotation snapshots with
checked source window joins to Hoplite embeddings. SQLite tests do not establish
BigQuery transaction, latency or cost behavior.
