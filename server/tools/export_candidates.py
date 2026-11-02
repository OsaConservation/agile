"""Hoplite adapter, called after boundary selection in 03_Classify.

Research dependencies are imported only inside the exporter, never by HTTP code.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import uuid


def export_candidates(
    *,
    results,
    db,
    weights,
    bias,
    output_dir,
    species_id,
    source_dataset_id,
    embedding_model,
    target_score=0.0,
    count=15,
    classifier_path=None,
    training_snapshot_id=None,
    candidate_set_id=None,
    classifier_run_id=None,
):
    """Write selected source windows as a metadata-only manifest."""
    import numpy as np
    from server.app.models import CandidateBundle
    from server.app.settings import Settings

    if count < 1:
        raise ValueError('count must be positive')

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError('Export output directory must be new')

    source_ids = [int(result.window_id) for result in results.search_results]
    if len(set(source_ids)) != len(source_ids) or not source_ids:
        raise ValueError('Search results must be nonempty with unique IDs')

    # Hoplite targeted sort_score is -abs(logit-target), not a signed logit.
    # Score from the source embeddings rather than trying to invert it.
    weights = np.asarray(weights)
    scored_windows = [
        (window_id, float(np.dot(np.asarray(db.get_embedding(window_id)), weights) + bias))
        for window_id in source_ids
    ]
    scored_windows.sort(key=lambda result: (abs(result[1] - target_score), result[0]))

    # A supplied artifact hash must describe the actual weights used above.
    artifact_hash = None
    if classifier_path:
        import base64

        content = Path(classifier_path).read_bytes()
        classifier_config = json.loads(content)
        column = classifier_config['classes'].index(species_id)
        saved_weights = np.frombuffer(
            base64.b64decode(classifier_config['beta']), dtype=np.float32
        ).reshape(-1, len(classifier_config['classes']))[:, column]
        saved_bias = np.frombuffer(
            base64.b64decode(classifier_config['beta_bias']), dtype=np.float32
        )[column]
        if (
            not np.array_equal(saved_weights, np.float32(weights))
            or saved_bias != np.float32(bias)
        ):
            raise ValueError(
                'Classifier artifact does not match the scoring weights/bias; '
                'save the current classifier first'
            )
        if isinstance(embedding_model, dict) and classifier_config['model_config'] != embedding_model:
            raise ValueError(
                'Classifier artifact embedding configuration differs from export metadata'
            )
        artifact_hash = hashlib.sha256(content).hexdigest()

    manifest = dict(
        schemaVersion=2,
        candidateSetId=candidate_set_id or str(uuid.uuid4()),
        speciesId=species_id,
        sourceDatasetId=source_dataset_id,
        createdAt=datetime.now(timezone.utc).isoformat(),
        classifierRun=dict(
            id=classifier_run_id or str(uuid.uuid4()),
            artifactSha256=artifact_hash,
            embeddingModel=embedding_model,
            trainingSnapshotId=training_snapshot_id,
        ),
        targetScore=float(target_score),
        selectionMode='boundary',
        candidates=[],
    )

    # Fail unknown mappings before publishing any output.
    species = json.loads(Settings().species_file.read_text(encoding='utf8'))
    if species_id not in {s['id'] for s in species}:
        raise ValueError(f'Unknown canonical species: {species_id}')

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix='export-') as staging:
        staging = Path(staging)
        for rank, (window_id, logit) in enumerate(scored_windows[:count]):
            window = db.get_window(window_id)
            recording = db.get_recording(window.recording_id)
            deployment = db.get_deployment(recording.deployment_id)
            if deployment is None:
                raise ValueError(f'Missing deployment for recording {recording.id}')

            start, end = map(float, window.offsets)
            manifest['candidates'].append(dict(
                sourceWindowId=window_id,
                recordingId=int(recording.id),
                offsets=[start, end],
                uri=recording.filename,
                datasetName=deployment.project,
                logit=logit,
                boundaryDistance=abs(logit - target_score),
                rank=rank,
                deployment=dict(
                    sourceDeploymentId=int(deployment.id),
                    name=deployment.name,
                    project=deployment.project,
                    latitude=deployment.latitude,
                    longitude=deployment.longitude,
                ),
            ))

        bundle = CandidateBundle.model_validate(manifest)
        (staging / 'manifest.json').write_text(
            bundle.model_dump_json(indent=2), encoding='utf8'
        )

        # Rename within the same filesystem only after the complete bundle exists.
        staging.rename(output_dir)

    return output_dir / 'manifest.json'


def main():
    """Optional standalone selection using the saved Hoplite JSON classifier."""
    import argparse
    import base64
    import numpy as np
    from ml_collections import config_dict
    from perch_hoplite.db import brutalism, score_functions, sqlite_usearch_impl

    parser = argparse.ArgumentParser(
        description='Export real boundary candidates from a read-only Hoplite database.'
    )
    parser.add_argument('--source-db', type=Path, default=Path('content/osa_db'))
    parser.add_argument(
        '--classifier', type=Path, default=Path('content/osa_db/agile_classifier.pt')
    )
    parser.add_argument(
        '--source-dataset-id', required=True,
        help='Stable namespace reused for every export of this corpus',
    )
    parser.add_argument('--species', default='DENEBR')
    parser.add_argument('--count', type=int, default=15)
    parser.add_argument('--target-score', type=float, default=0.0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')

    # Hoplite saves this .pt file as JSON with base64 float32 weights (not pickle).
    classifier_config = json.loads(args.classifier.read_text())
    column = classifier_config['classes'].index(args.species)
    weights = np.frombuffer(
        base64.b64decode(classifier_config['beta']), dtype=np.float32
    ).reshape(-1, len(classifier_config['classes']))[:, column]
    bias = np.frombuffer(base64.b64decode(classifier_config['beta_bias']), dtype=np.float32)[column]

    db = sqlite_usearch_impl.SQLiteUSearchDB.create(str(args.source_db), readonly=True)
    try:
        metadata = config_dict.ConfigDict(db.get_metadata(None))
        if classifier_config['model_config'] != metadata.model_config.to_dict():
            raise ValueError('Classifier embedding configuration does not match source database')

        results = brutalism.threaded_brute_search(
            db, weights, args.count,
            score_fn=score_functions.get_score_fn(
                'dot', bias=bias, target_score=args.target_score
            ),
            sample_size=1_000_000,
        )
        path = export_candidates(
            results=results,
            db=db,
            weights=weights,
            bias=bias,
            output_dir=args.output,
            species_id=args.species,
            source_dataset_id=args.source_dataset_id,
            embedding_model=classifier_config['model_config'],
            target_score=args.target_score,
            count=args.count,
            classifier_path=args.classifier,
        )
        print(path)
    finally:
        db.db.close()


if __name__ == '__main__':
    main()
