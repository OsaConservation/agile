"""Stage one: a new folder run produces existing server-compatible manifests."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import uuid
import wave

from .classification import SavedClassifier, select_candidates
from .embedding import embed_folder


@dataclass(frozen=True)
class PipelineRequest:
    audio_dir: Path
    classifier_path: Path
    output_dir: Path
    count: int = 15


def run_pipeline(request: PipelineRequest, progress=print):
    """Write run.json and per-species manifest.json; never import or upload.

    Output must be new and outside the source folder. Originals must remain
    available and unchanged for later import/playback. Failed runs are retained.
    """
    from server.app.settings import Settings
    from server.tools.export_candidates import export_candidates

    audio_dir = request.audio_dir.resolve()
    output = request.output_dir.resolve()
    classifier_path = request.classifier_path.resolve()
    if not audio_dir.is_dir() or request.count < 1:
        raise ValueError('Choose an existing audio folder and a positive candidate count')
    if output == audio_dir or output.is_relative_to(audio_dir):
        raise ValueError('Output directory must be outside the audio folder')
    classifier_bytes = classifier_path.read_bytes()
    classifier = SavedClassifier.from_bytes(classifier_bytes)
    known = {item['id'] for item in json.loads(Settings().species_file.read_text(encoding='utf8'))}
    if not set(classifier.classes) <= known:
        raise ValueError('Classifier contains species absent from server configuration')

    progress('Checking WAV files')
    files = sorted(p for p in audio_dir.rglob('*') if p.is_file() and p.suffix.lower() == '.wav')
    if not files:
        raise ValueError('No WAV files found in folder or subfolders')

    durations = {}
    for path in files:
        if not path.resolve().is_relative_to(audio_dir):
            raise ValueError(f'Audio link escapes folder: {path}')
        relative = path.relative_to(audio_dir).as_posix()
        if ':' in relative or '\\' in relative:
            raise ValueError(f'Unsupported recording filename: {relative}')
        with wave.open(str(path), 'rb') as source:
            if not source.getnframes() or not source.getframerate():
                raise ValueError(f'Empty WAV: {relative}')
            durations[relative] = (source.getnframes() / source.getframerate(), 1 / source.getframerate())

    output.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    dataset = f'ingest_{run_id}'
    state = {
        'schemaVersion': 1,
        'runId': run_id,
        'sourceDatasetId': dataset,
        'datasetName': dataset,
        'audioRoot': str(audio_dir),
        'fileCount': len(files),
        'createdAt': datetime.now(timezone.utc).isoformat(),
        'status': 'embedding',
        'manifests': {},
    }

    def save():
        temporary = output / 'run.json.tmp'
        temporary.write_text(json.dumps(state, indent=2), encoding='utf8')
        temporary.replace(output / 'run.json')

    save()
    db = None
    try:
        # Snapshot the exact artifact; later edits to the training file cannot
        # change the weights/provenance halfway through this run.
        snapshot = output / 'classifier.json'
        snapshot.write_bytes(classifier_bytes)

        progress('Creating embeddings')
        db = embed_folder(audio_dir, output / 'embeddings', dataset, classifier.model_config)

        # Hoplite can pad final windows. Only complete source intervals can be
        # served by the annotation server without extending beyond the WAV.
        ids = []
        represented = set()
        for wid in db.match_window_ids():
            window = db.get_window(wid)
            recording = db.get_recording(window.recording_id)
            duration, tolerance = durations[recording.filename]
            start, end = window.offsets
            if 0 <= start < end <= duration + tolerance:
                ids.append(int(wid))
                represented.add(recording.filename)

        state['eligibleWindowCount'] = len(ids)
        state['filesWithoutEligibleWindows'] = sorted(set(durations) - represented)
        if not ids:
            raise ValueError('No complete audio windows were embedded')
        state['status'] = 'classifying'
        save()

        progress('Selecting closest windows to zero for each species')
        selections = select_candidates(db, ids, classifier, count=request.count)
        for column, species in enumerate(classifier.classes):
            # Reuse the existing exporter through its search-results interface.
            results = SimpleNamespace(search_results=[
                SimpleNamespace(window_id=wid) for wid in selections[species]
            ])
            manifest = export_candidates(
                results=results, db=db, weights=classifier.weights[:, column],
                bias=classifier.biases[column], output_dir=output / 'bundles' / species,
                species_id=species, source_dataset_id=dataset,
                embedding_model=classifier.model_config, count=request.count,
                classifier_path=snapshot, classifier_run_id=run_id,
            )
            state['manifests'][species] = manifest.relative_to(output).as_posix()

        state['status'] = 'ready_for_transfer'
        save()
        progress(f'Candidate manifests ready: {output / "run.json"}')
        return state
    except BaseException as exc:
        state.update(status='failed', error=str(exc))
        save()
        raise
    finally:
        if db is not None:
            db.db.close()
