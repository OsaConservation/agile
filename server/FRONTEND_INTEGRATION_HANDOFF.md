# Frontend integration handoff

Implement the local annotation server client in `C:\OsaAnurans`. The backend
already works in `C:\agile`; this task is to connect the existing annotation UI,
not build another UI or database adapter. Preserve unrelated user changes.

## Current state and intended result

- `docs/annotation-data.mjs`: `ServerProvider` is a skeleton. `DemoProvider` and
  `buildSubmission` already exist.
- `docs/annotation-config.js`: currently selects demo mode; its server base URL
  is already `http://127.0.0.1:8080/api`.
- `docs/annotate.js`: already loads providers, renders references/candidates,
  tracks dirty answers, builds submissions, checks acknowledgements, and keeps
  answers on save failure. Extend this flow rather than duplicating it.
- `docs/audio-window.mjs`: already bounds full-recording playback. Keep it.
- `docs/ANNOTATION_API.md` contains historical proposals and skeleton TODOs.
  Update it to the implemented contract below; some older statements are stale.

The result should let a user select DENEBR, listen to three reference examples
and 15 candidate windows, save yes/no/unsure answers, and load remaining work.
Other species can have references but no candidates; that is a valid empty state.

## Running locally

Backend, from `C:\agile` (check `/health` before starting another instance):

```powershell
.venv/Scripts/python.exe -m uvicorn server.app.main:create_app --factory --host 127.0.0.1 --port 8080
```

Frontend, from `C:\OsaAnurans`, using an available Python interpreter:

```powershell
python -m http.server 8000 --bind 127.0.0.1
```

Open `http://127.0.0.1:8000/docs/annotate.html`. Do not use `file://`.
Backend CORS allows `http://127.0.0.1:8000` and `http://localhost:8000`.
Different frontend ports require backend configuration changes.

Switch the frontend configuration to:

```javascript
window.annotationConfig = {
    provider: "server",
    baseUrl: "http://127.0.0.1:8080/api",
};
```

No Google credentials, browser database access, authentication headers, or new
runtime dependencies are needed. The server assigns one local annotator
(`local-development` by default); this is local attribution, not authentication.

## API contract

Paths below are relative to `baseUrl` unless explicitly stated otherwise.
Encode path identifiers with `encodeURIComponent`. Treat IDs as opaque strings.
Do not use scientific names as IDs: the canonical species code is `DENEBR`.

### Species: `GET /species`

Returns an array (not an envelope):

```json
[{"id":"DENEBR","scientificName":"Dendropsophus ebraccatus","commonName":"Hourglass treefrog"}]
```

The full catalog has 52 species. Use the returned names for display and the
returned `id` for requests and submissions. Do not maintain a second mapping.

### References: `GET /species/DENEBR/references`

```json
{
  "target": {
    "id": "DENEBR",
    "scientificName": "Dendropsophus ebraccatus",
    "commonName": "Hourglass treefrog",
    "clips": [{
      "audioUrl": "http://127.0.0.1:8080/media/references/0010.mp3",
      "spectrogramUrl": "http://127.0.0.1:8080/media/references/0011.webp",
      "startSeconds": null,
      "endSeconds": null
    }]
  },
  "comparisons": []
}
```

Asset filenames above are illustrative: always use returned URLs.
Comparisons contain the other species, including species with `clips: []`.
There is no `imageUrl` today; do not require one. `spectrogramUrl` can be null.
References play their whole file using native audio controls. Null playback
bounds are normal. Current local fixtures contain 19 clips across 12 species,
including three DENEBR clips, copied from the frontend's catalog and static assets.

### Candidates: `GET /species/DENEBR/candidates`

```json
{
  "batchId": "opaque-server-issued-context",
  "clips": [{
    "id": "osa-local:window:24898",
    "recordingId": 588,
    "offsets": [50.0, 55.0],
    "uri": "sw2/1Jsw2_20261008_000000.wav",
    "datasetName": "grillos",
    "audioUrl": "http://127.0.0.1:8080/media/recordings/osa-local:recording:588",
    "spectrogramUrl": null,
    "startSeconds": 50.0,
    "endSeconds": 55.0,
    "numAnnotations": 0
  }]
}
```

- At most 15 windows per response, unique by `id`. No scores, ranks, clip paths,
  hashes, or classifier metadata are provided or needed by the UI.
- `audioUrl` serves the full original WAV with HTTP byte-range support. Keep
  existing Play/Pause window controls: seek to `startSeconds`, stop at
  `endSeconds`, and replay from the start. Do not download/decode/trim whole WAVs.
  Playback bounds are approximate, not access restrictions on the full file.
- Candidate `spectrogramUrl` points to `/media/windows/{id}/spectrogram` once
  images are generated. Render it when present; handle null or missing images gracefully.
- `recordingId` is a numeric source recording ID; `id` is a string window ID.
  Preserve their types and all source metadata when saving.
- `numAnnotations` counts annotation rows for the window/species, including
  revisions. It is not a count of distinct reviewers.
- Empty queue: `{"batchId":null,"clips":[]}`. Show a normal no-work state.
- `batchId` is a species/annotator context, not a frozen assignment or cursor.
  Fetching again returns windows this annotator has not answered. Any saved
  label, including `unsure`, removes that window from their queue.

### Save: `POST /annotations`

Send JSON with `Content-Type: application/json`, using `buildSubmission`:

```json
{
  "submissionId": "client-generated-UUID",
  "speciesId": "DENEBR",
  "batchId": "the-returned-context",
  "createdAt": "2026-11-02T12:00:00.000Z",
  "annotations": [{
    "windowId": "osa-local:window:24898",
    "recordingId": 588,
    "offsets": [50.0, 55.0],
    "uri": "sw2/1Jsw2_20261008_000000.wav",
    "datasetName": "grillos",
    "label": "yes"
  }]
}
```

One submission contains 1–15 answers, no duplicate window IDs. Labels are exactly
`yes`, `no`, or `unsure`; omit unanswered clips. Echo the original source fields
unchanged. `createdAt` must include UTC; it is accepted for compatibility but the
server stores its own timestamp. Do not send annotator identity or extra fields.

Success is HTTP 200 with `{"submissionId":"client-generated-UUID","savedCount":1}`.
Only mark answers saved after validating that both acknowledgement fields match
the submitted snapshot. Saves are atomic: partial success is not a valid result.

The same submission ID and answers are safe to retry, including after restart.
Changed answers/source metadata/species/context under the same ID produce 409.
A new submission ID appends new annotation rows; the latest per person/window/
species is current. The backend does not overwrite audit history.

### Optional recovery: `GET /submissions/{submissionId}`

Returns the original acknowledgement for this annotator, or 404. Useful after an
ambiguous timeout. A 404 at that moment does not prove an earlier request cannot
still commit; retry the unchanged submission ID rather than creating a new one.
Basic integration may simply retry the unchanged POST.

### Health and errors

`GET http://127.0.0.1:8080/health` is outside `/api` and returns
`{"status":"ok","storage":"sqlite","schemaVersion":6}`.

API errors use `{"error":{"message":"..."}}`, sometimes with validation
`details`. Handle non-JSON responses too. Relevant statuses: 404 unknown entity,
409 submission conflict, 422 invalid request, 503 storage unavailable. Never
silently fall back to demo mode or treat an HTTP error as a successful save.

## Implementation requirements

1. Implement all four `ServerProvider` methods using a shared request helper.
   Normalize trailing slashes, encode IDs, check `response.ok`, parse JSON,
   validate response shapes, and return the existing provider shapes unchanged.
   Allow injected fetch for tests; use a finite timeout with AbortController and
   clear timeout resources on completion. Surface useful server/network errors.
2. Validate required types and invariants without introducing a framework:
   unique species/window IDs, clip lists, finite increasing playback bounds,
   numeric recording IDs, valid source fields, nonnegative integer counts,
   matching reference target, and valid acknowledgements. Accept null optional
   media fields and the empty-queue null batch ID. Do not coerce IDs to numbers.
3. Preserve existing dirty-answer and save-lock behavior. Unchanged failed saves
   must reuse their submission snapshot/ID. Do not automatically retry a write
   with a new ID. Keep answers on timeout, malformed JSON, or invalid ack.
4. Handle advancement explicitly. Prefer a Next/Load remaining action enabled
   after acknowledged saves, with unsaved-answer protection. Fetch candidates
   again rather than inventing a cursor. Do not discard unanswered windows or
   pending failed submissions during refresh. If refresh fails after a successful
   save, show that the save succeeded and offer a separate load retry; never
   present it as a failed save. Pause outgoing audio before replacing candidates.
5. Use readable species names in prompts rather than exposing `DENEBR` alone.
   Keep code as the actual selection value. Provide a clear way to select DENEBR
   for testing; the first catalog species may have no candidates.
6. Update API docs and replace the test asserting that ServerProvider is a
   skeleton. Keep demo mode selectable. Existing DemoProvider has user edits
   (`console.log` and `this.storage.clear()`) that cause a persistence test to
   fail; treat that as a pre-existing issue and do not silently change it as part
   of server integration. Explain separately if a fix is needed.

## Verification

Add focused Node tests with injected fetch for routes/encoding, base URL handling,
request JSON, successful responses, null media, empty queues, HTTP errors,
malformed responses, timeouts, and acknowledgement mismatches. Cover reuse of a
submission ID after failure and preservation of user answers. Retain the audio
window tests:

```powershell
node --test tests/annotation-data.test.mjs tests/audio-window.test.mjs
```

Browser acceptance against the real local server:

1. Select DENEBR. Confirm three target references and up to 15 candidates.
2. Play a reference and a candidate. Check mutual audio exclusion, candidate
   start/end/replay, spectrograms, and missing optional image handling.
3. Answer a few windows with all three labels, save, and confirm the exact ack.
4. Load remaining work: answered windows disappear; unanswered windows remain.
5. Stop the backend and attempt a save. Answers remain, retry feedback is clear,
   and the unchanged retry uses the same ID after the backend returns.
6. Confirm a species with no candidates shows the empty state, not an error.
7. Check browser console/network for unexpected errors or CORS failures.

Real browser saves persist in `server/local/annotations.sqlite` and consume this
annotator's queue. Do not delete the database to reset testing. For an isolated
test store, start the backend with `OSA_SERVER_LOCAL` pointing to a new directory,
then import fixtures before launch:

```powershell
# Run from C:\agile; choose a fresh path.
$env:OSA_SERVER_LOCAL = 'C:/agile/server/local/frontend-test'
.venv/Scripts/python.exe -m server.tools.import_candidates server/local/bundles/denebr-recordings/manifest.json
.venv/Scripts/python.exe -m server.tools.import_references C:/OsaAnurans
.venv/Scripts/python.exe -m uvicorn server.app.main:create_app --factory --host 127.0.0.1 --port 8080
```

Stop any existing port-8080 server first. Keep its data. Test labels stay in the
application SQLite store; the original Hoplite research database is unchanged.
No BigQuery, authentication, new backend endpoints, or training integration is
required for this frontend task.
