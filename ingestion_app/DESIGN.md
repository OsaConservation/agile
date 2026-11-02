# Audio folder ingestion app

## Goal and scope

A nontechnical user opens a local app, drops an audio folder, and starts processing.
The app embeds audio windows, applies the existing trained classifier, and sends
up to 15 windows per species nearest to logit zero to the annotation server.
Human labeling remains in the annotation frontend.

Implementation is staged. Stage 1 in this directory is the folder-to-manifest
Python pipeline and CLI. The browser UI, launcher, resumable jobs, remote upload,
and server ingestion routes below are planned interfaces, not implemented features.
No training database, existing classifier, or annotation server store is modified
by stage 1. Run the CLI from the repository root in the existing research venv.

## User flow

1. Open a launcher that starts a loopback-only API and opens its browser page.
2. Drop a folder or use a folder picker. Include nested folders. Show supported
   file count and errors before starting.
3. Click Process audio. Show checking, embedding, classifying, and transferring
   stages, with progress and friendly errors. Species come from the configured
   classifier; model paths and server settings belong in administrator config.
4. Show candidates sent per species and files that could not be processed.
   Retry failed transfers using the same run, without recomputing embeddings.

A browser cannot hand arbitrary local filesystem paths to Python. The eventual
browser interface must enumerate folder files and upload their bytes with relative
paths to the local API. A native folder picker would require a desktop bridge.
Use a managed local snapshot so removing/changing the dropped folder does not
invalidate a running job. Budget disk space for this copy and the embeddings.

## Architecture and data flow

Local browser -> local job API -> separate Python worker -> Hoplite embedding DB
-> saved classifier -> exact boundary selection -> candidate manifests + selected
source recordings -> remote ingestion API -> existing SQL candidate queue
-> existing annotation frontend.

The Perch model runs locally. First use may download model weights; later runs use
the cache. CPU is allowed; runtime depends on corpus size and hardware. Model
loading and TensorFlow stay out of HTTP request handlers.

Use the saved classifier's full model_config, not a separately chosen preset.
The .pt artifact is JSON with base64 float32 arrays. Its classes list determines
weight column order. Each window/species logit is dot(embedding, weights) + bias.
For every species, score every eligible window in the new run. Keep the first
15 by (abs(logit), sourceWindowId); return fewer when fewer windows are available.
No epsilon threshold, probability filter, or million-window sampling limit.
The same window can be selected for multiple species.

Each run has its own Hoplite database and source namespace. Integer recording
and window IDs are only meaningful within that namespace. Never point a new
folder run at content/osa_db. The current server does not store embeddings or
selection logits in its application tables; scoring provenance stays in manifests.

## Key data structures

| Structure | Fields / invariant | Owner |
|---|---|---|
| PipelineRequest (implemented dataclass) | audio_dir, classifier_path, output_dir: Path; count: int = 15; output is new and outside input | pipeline.py |
| SavedClassifier (implemented dataclass) | classes: tuple[str]; model_config: dict; weights: float32[D,C]; biases: float32[C]; unique classes and finite values | classification.py |
| Selection (implemented return value) | dict[species, list[sourceWindowId]], closest first; at most count per species | classification.py |
| Run record (implemented JSON v1) | schemaVersion, runId, sourceDatasetId, datasetName, audioRoot, fileCount, createdAt, status, manifests; eligibleWindowCount and filesWithoutEligibleWindows after embedding; error on failure | run.json |
| CandidateBundle (existing Pydantic contract v2) | candidateSetId, speciesId, sourceDatasetId, createdAt, classifierRun, targetScore=0, selectionMode=boundary, candidates | server/app/models.py |
| Candidate (existing contract) | sourceWindowId, recordingId, offsets[start,end], uri (relative POSIX), datasetName, logit, boundaryDistance, rank, deployment | server/app/models.py |
| ClassifierRun (existing contract) | id, artifactSha256, embeddingModel, trainingSnapshotId (null until training provenance exists) | server/app/models.py |
| Job (planned persistent record) | run fields plus stage counters, per-file outcomes, snapshot hashes, worker ownership/heartbeat, transfer acknowledgements, cancellation request | jobs.py |
| TransferRecording (planned) | sourceDatasetId, sourceRecordingId, relativePath, sizeBytes, sha256; no client-supplied absolute server path | server_client.py / server ingestion |

Window records now also have a nullable `spectrogram_path` in server schema v6.
`server.tools.generate_spectrograms` renders each imported window's exact audio
interval to a WebP in server/local/spectrograms and persists its relative path.
Candidate APIs return `spectrogramUrl` served by
`GET /media/windows/{id}/spectrogram`. Future remote ingestion should invoke this
generation step after recording availability and candidate import. Reference
images copied from the frontend remain separate and cannot represent new windows.

Stage 1 records files that produced no complete windows, but does not capture
per-shard decoder failures hidden by Hoplite. These outcomes must be made explicit
before the UI claims every file succeeded. Invalid WAV headers fail preflight.

## Interfaces and files

### Implemented processing interfaces

```python
SavedClassifier.load(path: Path) -> SavedClassifier
embed_folder(audio_dir, db_path, dataset_name, model_config) -> HopliteDB
select_candidates(db, window_ids, classifier, count=15, batch_size=1024)
    -> dict[str, list[int]]
run_pipeline(request: PipelineRequest, progress=print) -> dict
```

embed_folder returns an open DB; the caller closes db.db. run_pipeline owns the
DB lifecycle. Its progress callback receives stage messages, not precise file
percentages. Research imports are deferred until embedding. The exporter is reused
to recompute signed logits and validate artifact provenance and bundle ordering.

| File | Responsibility |
|---|---|
| __main__.py | CLI arguments; delegates to pipeline |
| pipeline.py | Preflight WAVs; allocate run; snapshot classifier; embed; remove padded windows; select/export; persist outcomes |
| embedding.py | Configure and call the installed Hoplite EmbedWorker |
| classification.py | Validate saved classifier; exact batched top-k selection |
| tests/test_pipeline.py | Boundary selection, artifact validation, pipeline/export/import compatibility |
| README.md | Developer launch command and current limitations |
| app.py (planned) | Loopback job API and static UI; start workers, never embed in handlers |
| jobs.py (planned) | Persistent state, worker recovery, cancellation and checkpoints |
| server_client.py (planned) | Recording uploads, manifest submission, retry and acknowledgement |
| web/index.html, web/app.js, web/styles.css (planned) | Folder intake, stages, errors and completion |
| launcher (planned) | Start local app and open browser without a terminal workflow |

### Planned local API

- POST /api/jobs: allocate intake job; return jobId and upload requirements.
- PUT /api/jobs/{id}/files/{fileId}: stream one file into managed staging; validate
  declared relative path, size and content; never trust a browser path as an OS path.
- POST /api/jobs/{id}/start: finalize intake, validate, enqueue worker; return 202.
- GET /api/jobs/{id}: stage, counts, errors and final per-species results; poll initially.
- POST /api/jobs/{id}/retry-transfer: reuse artifacts and transfer IDs.
- POST /api/jobs/{id}/cancel: request cancellation between safe checkpoints.

Bind locally, validate request origins, and protect mutation routes with a
per-launch token. Job persistence belongs in a small local SQLite store; limit
initial concurrency to one embedding job. Credentials stay in administrator
configuration and must not be included in run artifacts or browser responses.

### Planned remote server ingestion API

The existing server has no HTTP candidate import or recording upload endpoint.
Add an authenticated ingestion boundary while reusing CandidateBundle validation
and transactional candidate insertion:

- PUT /api/ingestion/runs/{runId}/recordings/{sourceRecordingId}: streamed bytes
  plus recording identity/hash metadata; stage on disk; validate WAV; acknowledge.
- POST /api/ingestion/runs/{runId}/candidate-sets: CandidateBundle; verify every
  referenced recording exists and offsets fit; import atomically per species.
- GET /api/ingestion/runs/{runId}: accepted recording hashes and candidate sets
  so a client can recover after losing an acknowledgement.

Identical retries return original acknowledgements. Reusing an identity with
different bytes/metadata returns 409. The client sends a selected full recording
once even when multiple windows/species reference it. Commit queue entries only
after media is available; clean abandoned staging separately. The server owns
its media locations and resolves dataset names through its recording registry.
Current Settings.audio_roots is a static mapping; extend source resolution for
managed uploads instead of requiring manual environment edits for every new run.

Keep original-recording playback plus window offsets, matching the current
frontend contract. Compressed audio support requires decoding to a canonical WAV
snapshot before embedding, preserving relative identity and timing; do not just
add MP3 extensions to discovery while the server requires WAV.

## Persistence, failure and identity

Stage 1 writes a new UUID namespace for each invocation. Reimporting the same
manifest is safe through existing server behavior. Rerunning a folder with a new
output directory is a new corpus namespace and can duplicate review content;
cross-run deduplication is not implemented. Reuse requires an immutable file
inventory with hashes, classifier hash, embedding config, and stable namespace.

Stage 1 statuses: embedding -> classifying -> ready_for_transfer, or failed.
run.json is replaced atomically. A failed output directory is retained and must
not be reused by the CLI. Earlier completed species bundles from a failed run
are diagnostic artifacts; do not transfer the run until its status is ready.
The CLI snapshots the classifier, but original WAV files must remain unchanged.

Planned job states: receiving -> validating -> embedding -> classifying ->
transferring -> completed; failed and cancelled include a last safe checkpoint.
Persist outcomes per file/shard and transfer acknowledgement. Resume only against
the same input snapshot, embedding config and classifier hash. Cancellation must
not delete original files or already accepted candidates. Show skipped/failed
files explicitly and distinguish processing completion from successful delivery.

Hoplite can emit windows padded beyond the end of a recording. Stage 1 excludes
these because the annotation server requires complete source windows. Files
shorter than a model window may therefore yield no candidates. IDs of windows
are held in memory; embeddings are fetched/scored in batches, retaining top-k
per species. Very large corpora may later need cursor-based ID iteration.

## Delivery and validation

1. Implement folder -> isolated embeddings -> saved classifier -> schema v2
   manifests. Test signed logits, zero-boundary ties, batch independence, source
   identity, end-window filtering, and compatibility with the actual importer.
2. Add recording/manifest transfer, authentication, durable retry acknowledgements,
   and integration tests against temporary server stores.
3. Add local browser UI, background process, persistent jobs and launcher. Test
   actual browser folder dropping, nested relative paths, large files, cancellation,
   restart and offline transfer recovery.
4. Run a small real Perch acceptance corpus, compare selected windows with scalar
   notebook scoring, then measure realistic folder runtime and disk usage.

Training/retraining and returning frontend annotations to Hoplite are separate
follow-up work. Synthetic integration tests validate pipeline contracts, not
real embedding model accuracy or end-to-end remote delivery.
