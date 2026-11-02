# Audio ingestion app

Stage 1 provides the processing core. Browser folder dropping and remote upload
are planned in [DESIGN.md](DESIGN.md), including data structures and API contracts.

From the repository root, with the existing research environment:

```powershell
.venv/Scripts/python.exe -m ingestion_app ./data/grillos --output ./ingestion_app/local/run-001
```

Optional: `--classifier path/to/classifier.pt` and `--count 15`.
The output directory must be new and outside the audio folder.

The pipeline recursively finds WAV files (including uppercase extensions), loads
the saved classifier's embedding configuration, and embeds into a fresh Hoplite
database. It selects up to 15 complete windows nearest zero per classifier species
and writes `bundles/<species>/manifest.json`, the classifier snapshot, and `run.json`.
The first embedding run may download the model. Keep the source recordings in
place and unchanged. Other audio formats are ignored in this version.

This command does not upload audio or import into the annotation database. To
test the existing local importer, configure `OSA_AUDIO_ROOTS` to include the
`datasetName: audioRoot` mapping printed in the run record, then run:

```powershell
.venv/Scripts/python.exe -m server.tools.import_candidates ./ingestion_app/local/run-001/bundles/DENEBR/manifest.json --generate-spectrograms
```

Preserve any other required audio-root mappings. A remote server needs its own
copy of the referenced recordings; metadata alone is insufficient.
The optional flag generates per-window WebP images from exact audio offsets and
stores their paths in SQL. Candidate responses then include `spectrogramUrl`.
Rendering requires the research environment.
Each new invocation creates a new source namespace. Transfer retries should reuse
the completed manifests; restarting the CLI is not a deduplicated retry.

```powershell
.venv/Scripts/python.exe -m unittest discover -s ingestion_app/tests -v
```
