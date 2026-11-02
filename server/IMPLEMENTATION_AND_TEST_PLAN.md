# Local annotation server: implementation and test plan

Status: chunks 1 and 2 implemented: FastAPI, separate SQLite storage, versioned
candidate bundles, real Hoplite export/import, stable batches/audio, durable
retry-safe submissions, and revision history. See [README.md](README.md) for
commands and test coverage. Frontend integration and training snapshots remain
planned work; the logical structures below include those future stages.
API version 0.2 and the frontend provider contract use `/candidates` and shared
`clips` lists for candidate batches and reference species. Source `Recording`
records and `recordingId` retain their original-file meaning.

## Goal

Run the annotation workflow locally using real decision-boundary candidates from
`03_Classify.ipynb`, a FastAPI server in this repository, and the existing
`C:\OsaAnurans\docs\annotate.html` frontend. Save annotations durably and make
them available for subsequent training. Keep storage behind an interface so a
future BigQuery adapter can replace local storage without changing the frontend.

This experiment validates the application workflow. It does not validate
BigQuery SQL, latency, billing, transactions, or concurrency behavior.

## Existing integration points

- The frontend contract is in `C:\OsaAnurans\docs\ANNOTATION_API.md`.
- `annotation-config.js` already specifies `http://127.0.0.1:8080/api`, but selects
  the demo provider. `ServerProvider` in `annotation-data.mjs` is unimplemented.
- The frontend expects species, reference sounds, review batches, and submission
  endpoints. It uses `yes`, `no`, and `unsure`, supports partial batches and
  corrections, and retains submission IDs for unchanged retries.
- Notebook cell 7 currently requests five candidates using
  `brutalism.threaded_brute_search`, classifier weights, bias, and a target logit.
  Export its actual results; do not substitute frontend sample candidate clips.
- The notebook currently trains from the existing Hoplite database. The new local
  annotation store must be separate from `content/osa_db`.

## Architecture and boundaries

```text
03_Classify -> versioned candidate bundle -> import command -> SQLite repository
                                                               ^
                                                               |
OsaAnurans browser -> FastAPI routes -> application services -> repository interface
                           |
                           +-> local clip files
```

Use FastAPI/Uvicorn for HTTP, Pydantic for request/response validation, and a
separate SQLite database for local persistence. This is an application store,
not a BigQuery emulator. Keep TensorFlow, Hoplite, and classifier training out of
the server's import path and runtime dependencies.

Proposed layout:

```text
server/
  IMPLEMENTATION_AND_TEST_PLAN.md
  README.md
  requirements.txt
  app/
    main.py
    settings.py
    models.py
    services.py
    repository.py
    sqlite_repository.py
  config/species.json
  tools/
    export_candidates.py
    import_candidates.py
    export_annotations.py
  tests/
  local/                  # ignored: SQLite, imported bundles, clips
```

The notebook exporter may use the research environment; the HTTP server should
have its own small dependency set. Exact package versions will be selected and
verified during implementation.

## Coding design and integration practices

Use a small layered design with explicit adapters at integration boundaries:

```text
Frontend -> HTTP routes -> application services -> repository protocol
                                                       |
                                                SQLite / future BQ
Hoplite notebook -> export adapter -> candidate import service
```

### Responsibilities and dependency direction

- **HTTP routes:** parse and validate API payloads, call application services,
  translate known errors into HTTP responses, and serialize results. Keep SQL,
  assignment policies, and annotation revision logic out of route handlers.
- **Application services:** implement batch selection, trusted-metadata checks,
  species mapping, corrections, training export policies, and submission
  semantics. They depend on application models and a repository protocol, not
  on FastAPI or concrete database connections.
- **Repository protocol:** expose focused application operations such as
  `import_candidate_set`, `get_review_batch`, `save_submission`, and
  `export_annotation_snapshot`. Specify atomicity, idempotency, ordering, and
  conflict behavior in the contract. Avoid a generic CRUD framework that leaves
  these guarantees to callers.
- **Storage adapters:** contain SQL, transactions, schema migrations, and
  database-specific error translation. Concurrency-sensitive validation and
  writes must execute within the same repository operation; a service-level
  check followed by an unrelated write is insufficient.
- **Hoplite adapters:** translate source objects and label semantics into
  versioned application exports. They may depend on research libraries; the
  HTTP server must not depend on those adapters at import time.

### Models and contracts

Separate API request/response models, internal application records, and Hoplite
objects. Translate between them explicitly, particularly source versus server
IDs and species code versus observation label. Avoid one shared mutable object
whose field meanings change between layers.

Maintain the frontend contract in documentation and generated OpenAPI. Version
candidate bundles and annotation snapshots. Evolve the two repositories through
compatible API changes, or fail clearly on unsupported versions. Keep species
mapping authoritative in this repository and expose it through the API rather
than duplicating constants in the frontend.

### Dependency construction and configuration

Use an application factory with explicit settings and repository dependencies.
Construct connections/resources during application startup and close them during
shutdown; do not create global connections as an import side effect. Tests inject
temporary SQLite repositories. Configuration selects the storage implementation
in one composition point rather than scattering backend checks through routes.

Give SQLite connections explicit transaction and foreign-key settings. Keep
schema changes in versioned migrations; initialization must not destructively
reset an existing annotation store. Keep generated databases, clips, snapshots,
and local settings outside version control.

### Integration with the existing research workflow

Keep notebook cells short: call reusable exporter and training-bridge modules
instead of embedding server logic in the notebook. Initially read the existing
Hoplite database without changing its annotation tables. Export actual candidates
into the new store and return annotations through an identified snapshot with
explicit label precedence and source-ID validation.

Do not add two-way database synchronization. The local server owns new frontend
annotation writes; the existing store supplies source windows, embeddings, and
historical labels. Preserve provenance when combining these for training.

### Verification and implementation discipline

Test behavior at boundaries: real frontend payloads, candidate import, clip
delivery, atomic saves, retries, corrections, restart persistence, and source-ID
joins. Reuse repository contract tests for each storage adapter. Use small
fixtures for routine tests and real notebook results for the acceptance walkthrough.

Implement the first complete slice before expanding features:
export actual candidates -> import -> fetch a batch -> play clips -> save labels
-> restart and verify persistence. Then complete snapshot-based retraining.

The repository protocol expresses required guarantees, not an assumption that
every backend provides them. If a future BigQuery adapter cannot meet atomic
retry-safe submission handling or other requirements, revise the design and
document the tradeoff rather than silently weakening the contract.

## Canonical species mapping

Maintain a single mapping in `server/config/species.json` with stable code `id`,
`scientificName`, and `commonName`. Begin with verified species used by the
notebook, including `DENEBR`; do not guess expansions for other codes. Optional
reference images and example audio can be configured separately by canonical ID.

The exporter validates its target label against this mapping. The API returns
the mapping to the frontend, which uses codes in requests and displays names.
Unknown labels fail import with a clear error. Frontend reference assets may
retain their filenames; their association with canonical IDs must be explicit.

## Candidate bundle and notebook integration

Define a versioned JSON manifest plus a directory of browser-playable WAV clips.
Keep the export helper in a normal Python module and add a small notebook cell
after candidate selection to call it. No training or scoring work occurs in an
HTTP request.

Manifest fields:

- `schemaVersion`, `candidateSetId`, `speciesId`, export timestamp.
- Classifier artifact hash/version, embedding model identity, target score,
  selection mode, and available training-snapshot provenance. Represent unknown
  historical provenance explicitly rather than inventing it.
- Per candidate: opaque window ID, source recording ID, dataset, source URI,
  start/end seconds, signed classifier logit, absolute distance from target,
  rank, and relative clip filename.
- Source dataset identity so local numeric window IDs cannot collide across
  unrelated Hoplite databases. Keep source IDs available for a checked training
  join; do not use a bare integer as a globally stable identity.

Compute/check raw logits as `dot(embedding, weights) + bias`; inspect Hoplite's
score/result conventions before deriving ranking from `score` or `sort_score`.
Use ascending `abs(logit - target)` with a deterministic tie breaker. Preserve
the selected windows, offsets, and audio loader behavior. Allow export count to
be configured; use 15 for the first frontend walkthrough.

Generate actual window clips during export using existing notebook audio-loading
support. Import validates the complete manifest and media before publishing a
candidate set. Reimporting identical content is safe; conflicting content under
the same identifier is rejected. Stage media before committing database records
so failed imports never expose batches with missing files.

## Repository interface and persistence

Define operations around application behavior, without exposing SQLite
connections, SQL strings, or BigQuery jobs to routes:

- List species and retrieve reference metadata.
- Import a candidate set and retrieve candidates with provenance.
- Create/retrieve a stable review batch for a species and annotator.
- Save an annotation submission and retrieve its acknowledgement.
- Export effective annotations with provenance for training.

Persist candidate sets/windows, issued batches and membership, submissions,
append-only annotation events, and effective label state. Use a database schema
version and a documented initialization/migration path.

For the MVP, different annotators may review the same window; exclusive claims
and leases are out of scope. Serve at most 15 candidates per batch in stored rank
order. Repeated fetches return the current batch until it is completed or the
client explicitly requests the next one. Define a next-batch operation when
adding frontend paging; do not silently change batch membership after a save.
Partial submissions do not complete a batch. `unsure` counts as reviewed but is
excluded from binary training labels.

Use an explicit local-development annotator configured on the server. Do not
claim authenticated identity or accept arbitrary identities from request bodies.
Multi-annotator tests can use separately configured test sessions. Real
authentication is required before remote/shared deployment.

Saving a submission must:

1. Validate species, batch membership, unique window IDs, labels, and submitted
   source metadata against trusted stored values.
2. Use a server timestamp; retain the client timestamp only as metadata.
3. Persist the whole submission, events, and effective revisions atomically.
4. Return the original acknowledgement for the same submission ID and payload;
   reject reuse with a different payload with HTTP 409. Compare validated,
   canonical payloads, not raw JSON bytes.
5. Preserve per-annotator history. A correction changes that annotator's current
   label without deleting earlier events or overwriting another annotator.

These are interface requirements to verify independently for a future BigQuery
adapter. SQLite transactions do not establish that BigQuery can meet them.

## Key data structures

These are logical application models, independent of SQLite table layout or
future BigQuery schemas. Internal Python fields can use snake_case; public JSON
uses the frontend's camelCase names. IDs are opaque strings unless explicitly
identified as source-system IDs. Timestamps are UTC.

| Structure | Key fields | Identity and purpose |
|---|---|---|
| `Species` | `id`, `scientificName`, `commonName` | Canonical code such as `DENEBR`; shared by exports, labels, and API requests |
| `ReferenceSpecies` | Species fields, optional `imageUrl`, `clips[]` containing `audioUrl` and optional `spectrogramUrl` | Frontend reference material; separate from review candidates |
| `SourceDataset` | `id`, `name`, embedding model/version, optional source configuration hash | Identifies one source corpus/identity namespace; IDs must persist across exports |
| `Deployment` | `id`, `sourceDatasetId`, `sourceDeploymentId`, `name`, `project`, optional latitude/longitude | Preserves Hoplite site/project grouping; project also supplies the current audio-source dataset name |
| `Recording` | `id`, `sourceDatasetId`, `sourceRecordingId`, `uri`, optional `contentHash`, duration | Server identity plus original Hoplite identity; URI alone is not a global key |
| `Window` | `id`, `recordingId`, `sourceWindowId`, `startSeconds`, `endSeconds` | Stable review unit; unique source dataset/window identity with validated recording and offsets |
| `ClassifierRun` | `id`, artifact hash, embedding model/version, training snapshot ID or explicit unknown provenance, creation time | Identifies the model that produced scores; independent of candidate selection settings |
| `CandidateSet` | `id`, `speciesId`, `classifierRunId`, `schemaVersion`, `targetScore`, `selectionMode`, creation time, manifest hash | Immutable, versioned output of a candidate selection run |
| `Candidate` | `candidateSetId`, `windowId`, `logit`, `boundaryDistance`, `rank`, `clipId` | Unique per set/window; `boundaryDistance = abs(logit - targetScore)`; scores are predictions, not human labels |
| `Clip` | `id`, `windowId`, relative file path, content hash, sample rate, frame count | Imported media asset; API exposes a URL, never a raw filesystem path |
| `ReviewBatch` | `id`, `speciesId`, `annotatorId`, `candidateSetId`, ordered `windowIds`, creation time, completion time | Persisted membership, at most 15 windows; a later candidate import cannot change an issued batch |
| `AnnotationSubmission` | `id`, `batchId`, `speciesId`, `annotatorId`, canonical payload hash, client/server timestamps, acknowledgement | One atomic save request; ID and payload determine retry behavior |
| `AnnotationEvent` | `id`, `submissionId`, `windowId`, `speciesId`, `annotatorId`, `label`, revision number, server timestamp | Append-only history; one event per changed answer in an accepted submission |
| `EffectiveAnnotation` | `windowId`, `speciesId`, `annotatorId`, `label`, latest event ID/revision | Derived current state; unique per annotator/species/window; may be a view or transactionally maintained projection |
| `AnnotationSnapshot` | `id`, schema version, creation time, selection/conflict policy, included event IDs, content hash | Immutable effective-label export used for reproducible training |

### Relationships and identity rules

```text
SourceDataset -> Deployment -> Recording -> Window -> Clip
ClassifierRun -> CandidateSet -> Candidate -> Window
Species ---------------------> CandidateSet
CandidateSet -> ReviewBatch -> AnnotationSubmission -> AnnotationEvent
AnnotationEvent -> EffectiveAnnotation -> AnnotationSnapshot -> ClassifierRun
```

Arrows show references or derivation, not ownership or cascading deletion.
Recording also retains optional `deploymentId` and `recordedAt`, original filename,
and source audio configuration. A missing source deployment is an explicit import
issue, never a reason to invent site metadata. `SourceDataset.id` namespaces the
source Hoplite database; it is distinct from frontend `datasetName`, which comes
from Hoplite deployment `project` in this corpus.
Reference media can exist without any candidate sets. One window can appear in
many candidate sets and batches; its labels remain attached to the window and
species, rather than being duplicated for each classifier run.

Retain the original Hoplite recording/window IDs as source fields. Public API
`recordingId` currently means the source recording ID, whereas internal
`Recording.id` is the server identity; use explicit conversion at the API
boundary. A public window `id` must match the `windowId` accepted on submission.
Do not silently change the meaning of an ID when a second corpus is imported.

Do not include embedding vectors in review responses or candidate manifests.
The first training bridge reads them from the existing Hoplite store using
checked source identities. If embeddings move into the server store later,
introduce a separate `Embedding` model keyed by window and embedding model/version.

### Example review response

Illustrative IDs and scores; filenames and names must be verified from exports.

```json
{
  "batchId": "batch-001",
  "clips": [
    {
      "id": "osa-local:window:25435",
      "recordingId": 633,
      "offsets": [35, 40],
      "uri": "sw2/1Jsw2_20261009_014000.wav",
      "datasetName": "grillos",
      "audioUrl": "http://127.0.0.1:8080/media/clips/clip-001"
    }
  ]
}
```

The enclosing batch supplies species, candidate-set, model, and assignment
context on the server. The browser need not send prediction scores back.

### Example submission and effective state

Preserve the existing frontend submission shape:

```json
{
  "submissionId": "client-generated-uuid",
  "speciesId": "DENEBR",
  "batchId": "batch-001",
  "createdAt": "2026-11-02T18:00:00Z",
  "annotations": [
    {
      "windowId": "osa-local:window:25435",
      "recordingId": 633,
      "offsets": [35, 40],
      "uri": "sw2/1Jsw2_20261009_014000.wav",
      "datasetName": "grillos",
      "label": "yes"
    }
  ]
}
```

The server derives annotator identity and validates all redundant source fields.
An accepted save returns `{ "submissionId": "client-generated-uuid",
"savedCount": 1 }`. Retrying that submission returns the same acknowledgement.
A later submission changing `yes` to `unsure` adds an event and updates effective
state; the earlier event remains in history. Training uses effective state from
an identified snapshot, excluding `unsure` rather than treating it as `no`.

### Fields requiring explicit semantics

- Labels are `yes`, `no`, or `unsure`; absence means no submitted answer and
  never means `no`. Explicit deletion/withdrawal is deferred and would require
  a defined event type rather than overloading absence.
- Revision numbers are monotonic per annotator/species/window, allocated by the
  repository. They establish correction order without relying on client clocks.
- A batch is complete once every member has an effective answer from its assigned
  annotator. Existing answers must be considered when issuing/retrieving batches;
  completion does not prevent an explicit correction to a previously issued batch.
- Candidate set and classifier metadata are immutable. New scoring or selection
  settings produce a new set, retaining model provenance and prior annotations.
- Clip hashes and manifest hashes detect conflicting reimports. A changed clip
  asset must not silently replace media already referenced by an issued batch.

## HTTP contract

Preserve the existing frontend response shapes, including its current numeric
source `recordingId` where required; browser-facing window IDs remain opaque
strings. Document any necessary contract changes in both repositories.

| Method/path | Behavior |
|---|---|
| `GET /health` | Local readiness check, including initialized storage |
| `GET /api/species` | Canonical species codes and display names |
| `GET /api/species/{id}/references` | Target and comparison reference media; allow empty reference clips |
| `GET /api/species/{id}/candidates` | Stable `{batchId, clips}` with playable clip URLs |
| `POST /api/annotations` | Validated, durable `{submissionId, savedCount}` acknowledgement |
| `GET /api/submissions/{id}` | Resolve uncertain submission outcome; unknown ID returns 404 |
| `GET /media/clips/{clip_id}` | Serve only registered, imported clips |

Use consistent error bodies and distinguish malformed requests, unknown records,
conflicting retries, and storage failures. Generate OpenAPI from the server
models. Keep the API prefix separate from FastAPI's documentation route.

Bind to `127.0.0.1:8080`. Allow the explicit local frontend origins on port 8000
through CORS. Serve media by registered IDs; never expose arbitrary filesystem
paths or the full audio directory.

## Frontend work in the separate repository

Implement `ServerProvider` fetch calls, response validation, timeouts, URL encoding,
and error propagation. Switch configuration to server mode and use canonical
species codes. Preserve unsaved answers and retry IDs on failure. Connect existing
reference assets by canonical species ID; missing references must not prevent
candidate playback or saving. Handle empty candidate lists explicitly.

The current workspace permits writes in `C:\agile`; edits under
`C:\OsaAnurans` require the applicable filesystem approval when implementation
reaches that step. This plan does not change the frontend.

## Returning annotations to training

Export a versioned snapshot with source dataset/window identity, species,
annotator, current label, revision/event IDs, and snapshot timestamp. Initially
train from one explicitly selected annotator to avoid implicit conflict
resolution. Map `yes`/`no` to the existing classifier's label representation only
after inspecting that loader; exclude `unsure` and withdrawn binary labels.

Add a checked training bridge that joins the snapshot to existing local
embeddings in batches without silently mutating the original Hoplite annotation
tables. Define precedence for preexisting notebook labels versus new server
labels, and avoid placing the same window in both training and evaluation.
Record the annotation snapshot used by each subsequent classifier export.

## Implementation sequence

1. Define schemas, species mapping, repository protocol, and training-label rules.
2. Implement SQLite persistence, import/export commands, and submission handling.
3. Implement API routes and local media delivery with small fixtures.
4. Add notebook export helper/cell and import real boundary candidates.
5. Integrate the separate frontend provider and reference mapping.
6. Add the annotation-to-training bridge and run the complete local loop.
7. Document exact install/start/export/import commands, reset instructions, and
   measured results. Keep generated media and databases out of git.

## Test plan

Use temporary databases and short generated audio fixtures for automated tests;
reserve the research environment and real corpus for integration checks.

| Area | Meaningful checks |
|---|---|
| Candidate export | Exported IDs match selected Hoplite results; logits match direct calculation within stated tolerance; boundary ordering is correct; clip duration and offsets match source audio |
| Import | Unknown species, duplicate/conflicting IDs, invalid offsets, nonfinite scores, and missing/unsafe media paths fail without publishing partial data; identical reimport is safe |
| Batch retrieval | Species isolation, deterministic ordering, stable batch membership, at most 15 results, and empty queue behavior |
| Submission integrity | Invalid membership/metadata/labels reject the entire batch; no partial writes; exact retry returns original acknowledgement; changed retry returns 409 |
| Revisions | Corrections preserve history; annotators do not overwrite each other; `yes -> unsure` removes the previous effective binary training label |
| Recovery | Data and batches survive restart; timeout after commit followed by retry produces one effective revision; concurrent identical submissions cannot duplicate events |
| HTTP/media | Existing frontend response contract, expected error codes, allowed CORS origin, playable WAV, missing clip handling, and traversal rejection |
| Training bridge | Correct source-window joins and binary-label mapping; explicit existing-label precedence; unsure/conflicting labels handled as specified; no train/evaluation leakage |

### Manual end-to-end acceptance

1. Run candidate selection in `03_Classify` for a verified mapped species.
2. Export/import 15 real results and start the server and frontend locally.
3. Select the species and verify the displayed windows match the exported IDs.
4. Listen to clips and compare several against notebook playback.
5. Save yes/no/unsure labels, revise one answer, and inspect persisted history.
6. Interrupt the server during a save, restart, retry, and confirm one effective
   result without lost answers.
7. Export an annotation snapshot, load it for training, and generate a new
   candidate set with the new classifier/snapshot provenance.
8. Record fetch/save/playback timing and any UI friction. These measurements
   describe the local implementation only.

Completion means the frontend shows actual classifier-selected audio, saves
survive restart and retries, and those labels can be used in the next training
run. Google Cloud credentials are not required for any of these steps.

## Deferred BigQuery validation

Implement the same repository behavior against a real development dataset and
run the applicable contract tests. Explicitly resolve atomic submissions,
idempotency, concurrent writes, and effective-label semantics. Then measure
fetch/save latency, query bytes, training reads, and scoring at projected scale.
Retain the option to revise the storage design if BigQuery cannot meet the
application contract economically and reliably.

## Existing Hoplite compatibility review

See [HOPLITE_DATA_REVIEW.md](HOPLITE_DATA_REVIEW.md) for observed schema, counts,
label semantics, and import issues. Preserve source annotation IDs, recording
intervals, label types, and provenance when importing historical labels. Their
original timestamps/revision order are unknown. Historical provenance strings
are source attribution, not authenticated accounts. Keep that attribution
separate from a newly authenticated or locally configured annotator identity.

Window-based frontend annotations are a supported subset of Hoplite's more
general interval annotations. Import arbitrary historical intervals only with
an explicit overlap policy. The current 176 annotations all match one window
exactly, so no overlap conversion is required for this corpus.

For training, choose explicitly between a selected annotator and Hoplite's
existing positive-vote fraction across annotations. Do not claim identical
training behavior when switching policies; preserve the vote-fraction policy
when comparing old and new classifier results.
