# Local annotation server

FastAPI + a separate SQLite application store. This implements chunks 1 and 2:
real classifier candidate export/import, species and reference metadata, stable
review batches, WAV delivery, and durable annotations with retries and revisions.
The existing Hoplite database is read-only input. BigQuery, frontend changes,
authentication, paging, and annotation-to-training snapshots are later work.

## Start

From `C:\agile`, using the existing research environment:

```powershell
.venv/Scripts/python.exe -m pip install -r server/requirements.txt
.venv/Scripts/python.exe -m uvicorn server.app.main:create_app --factory --host 127.0.0.1 --port 8080
```

Open <http://127.0.0.1:8080/docs> for interactive API documentation.
<http://127.0.0.1:8080/health> checks storage readiness.

For a separate lightweight server environment, create a venv and install only
`server/requirements.txt`. No TensorFlow, Hoplite, NumPy, or Google credentials
are required to serve previously exported bundles. The export helper uses the
research environment; the standalone exporter additionally needs Hoplite.

Optional environment variables (set before starting):

```powershell
$env:OSA_SERVER_LOCAL = 'C:\agile\server\local'
$env:OSA_ANNOTATOR = 'your-local-name'
```

The default local directory is `server/local`, independent of working directory;
the default annotator is `local-development`. Each process has one configured
annotator. This is local attribution, not authentication. Bind to loopback.
Allowed browser origins are `http://127.0.0.1:8000` and
`http://localhost:8000`. Keep the same annotator name when restarting.

## Export and import real candidates

The optional cell immediately after boundary selection in `03_Classify.ipynb`
exports the selected results. The selection default is now 15. Run training and
selection first, then that cell. It prints the import command. Keep
`source_dataset_id="osa-local"` stable across exports of this same database;
give a different source corpus a different ID. Do not renumber source windows.

Alternatively, export using the saved classifier without running the notebook:

```powershell
.venv/Scripts/python.exe -m server.tools.export_candidates --source-dataset-id osa-local --species DENEBR --count 15 --output server/local/bundles/denebr-first
.venv/Scripts/python.exe -m server.tools.import_candidates server/local/bundles/denebr-first/manifest.json
```

The output directory must be new. `--source-db`, `--classifier`, and
`--target-score` are optional. Default paths are `content/osa_db` and
`content/osa_db/agile_classifier.pt`; run the standalone exporter from the repo
root so the existing audio-source relative paths resolve. It opens Hoplite in
read-only mode and checks that classifier/source embedding configurations match.
The saved `.pt` artifact is Hoplite JSON, not a Python pickle.

`server/local/bundles/denebr-first` has already been generated and imported in
this workspace using 15 real DENEBR candidates. No synthetic labels have been
saved into the default application store. Export to a different directory for
another run. Importing an unchanged bundle again returns `imported: false`.

Bundles contain `manifest.json` (schema version 1) plus PCM16 WAV clips. The
exporter recomputes signed logits from source embeddings and classifier weights;
Hoplite boundary-search scores contain only negative absolute distance. Candidates
sort by ascending distance to target, then source window ID. No embeddings are
copied into the bundle or served to the browser. Unknown training-snapshot
provenance remains null.

Import validates every candidate and clip before publishing the set. Immutable
source identities and classifier-run metadata reject conflicting reimports.
Files are staged and installed by content hash before the database transaction;
a failed SQL import may leave unregistered, harmless clip files, never a partial
published set. Do not manually edit registered clips. The importer checks and can
restore a missing/corrupted content-addressed file on an identical reimport.

## HTTP contract

All public JSON uses camelCase. Source recording IDs remain integers; window
IDs are namespaced strings such as `osa-local:window:123`.

Both reference and candidate audio use the shared `AudioClip` media fields:
`audioUrl` and optional `spectrogramUrl`. Reference species return `clips` lists;
candidate batches return `{batchId, clips}`. Candidate clips additionally carry
their window `id` and source metadata for annotation validation. A `recordingId`
continues to identify the original source recording, rather than a review item.
API version 0.2 replaces `/recordings` with `/candidates` and removes the old
`recordings` and `examples` collection names. Stored source recordings and
candidate bundles are unchanged, so no database migration or reimport is needed.

| Request | Result |
|---|---|
| `GET /health` | Readiness and schema version |
| `GET /api/species` | Canonical codes and display names |
| `GET /api/species/DENEBR/references` | Target and comparisons, currently empty reference clip lists |
| `GET /api/species/DENEBR/candidates` | Stable `batchId` and up to 15 candidate clips with absolute `audioUrl` |
| `POST /api/annotations` | Atomic `{submissionId, savedCount}` acknowledgement |
| `GET /api/submissions/{id}` | Original acknowledgement, or 404 |
| `GET /media/clips/{id}` | Registered WAV only, with byte-range support |

No candidates returns `{ "batchId": null, "clips": [] }`. Once issued,
a batch remains the current batch, including after completion, until explicit
paging is added. New candidate imports cannot replace its membership. Different
annotators can review the same windows. Partial saves leave the batch incomplete;
`unsure` counts as reviewed. Corrections are allowed after completion.

Example save (copy source fields from the actual batch response):

```json
{
  "submissionId": "a-new-client-uuid",
  "speciesId": "DENEBR",
  "batchId": "batch-id-from-response",
  "createdAt": "2026-11-02T18:00:00Z",
  "annotations": [{
    "windowId": "osa-local:window:123",
    "recordingId": 7,
    "offsets": [0, 5],
    "uri": "site/recording.wav",
    "datasetName": "grillos",
    "label": "yes"
  }]
}
```

Labels are `yes`, `no`, or `unsure`. Unanswered windows are omitted. Source
metadata, batch membership, species, and configured annotator must match.
Timestamps must include UTC; server time determines audit timestamps and
monotonic per-window revisions determine correction order.

Reuse a submission ID and unchanged payload after a timeout. Annotation-array
order and JSON property order are ignored for retries. A changed payload or
different owner under the same ID returns 409. New IDs with unchanged labels
are accepted without duplicating annotation events. `savedCount` is the number
of accepted answers, including unchanged ones. Corrections append history and
replace the effective label only for that annotator/species/window. `unsure`
supersedes a previous binary label; later training exports must exclude it.

Errors use `{"error":{"message":"..."}}`, with validation details when
available. Invalid requests return 422, unknown resources 404, conflicts 409,
and SQLite failures 503. Retry storage failures using the original submission
ID. The client can resolve an uncertain outcome with the submissions endpoint.

The mapping in `config/species.json` was copied from the existing OsaAnurans
species catalog by explicit code/name columns (52 species). This repository owns
the API mapping going forward. Reference media configuration is deferred; empty
references do not prevent candidate playback or saving.

## Tests

```powershell
.venv/Scripts/python.exe -m pip install -r server/requirements-test.txt
.venv/Scripts/python.exe -m unittest discover -s server/tests -v
```

The fixture tests generate short WAVs and temporary SQLite stores. They verify
export scoring/ties/offsets, import validation, atomic rollback, identity
conflicts, batch stability, concurrency, idempotency, revisions, annotator
isolation, restart persistence, CORS, media range requests, and migration safety.

Opt-in real-corpus acceptance uses the existing research environment (or
`server/requirements-research.txt`) and the exported bundle:

```powershell
$env:OSA_REAL_BUNDLE = 'server/local/bundles/denebr-first/manifest.json'
.venv/Scripts/python.exe -m unittest discover -s server/tests -v
Remove-Item Env:OSA_REAL_BUNDLE
```

It checks every exported logit, ID, source offset, and waveform against the
read-only corpus, then imports, fetches clips, saves synthetic labels, restarts,
retries, and corrects a label **in temporary storage**. It expects the classifier
artifact used by the bundle to remain unchanged. Run a fresh export after retraining.

Validated in this workspace on 2026-11-02: all 21 tests passed with real-corpus
acceptance enabled. A separate live Uvicorn check on loopback returned readiness,
15 real candidates, and a 320,044-byte five-second WAV over HTTP. The smoke-test
process was stopped afterward. The default local store contains imported
candidates and a stable issued batch, with no test annotations.

## Storage and next steps

Schema migration `app/migrations/001_initial.sql` is applied transactionally to
new stores. Existing schema version 1 is preserved; unknown versions fail startup.
Connections enable foreign keys and close after each operation. SQLite immediate
transactions serialize batch assignment and complete saves. Back up with SQLite's
backup API or stop all server/import processes before copying the store; copying
only the main file while WAL writers run can lose committed data.

To start a separate experiment, point `OSA_SERVER_LOCAL` at a new directory and
import a bundle there. This preserves earlier annotations without a destructive
reset. Generated databases, bundles, clips, and caches are ignored by git.

The next chunk implements `ServerProvider` fetch calls in `C:\OsaAnurans` and
switches the frontend to server mode. Its existing demo provider, server method
names, UI, styles, tests, and contract now use the shared clip terminology.
Afterward, add
versioned annotation snapshots and a checked training bridge. SQLite passing
these tests does not establish BigQuery transaction, latency, or cost behavior.
