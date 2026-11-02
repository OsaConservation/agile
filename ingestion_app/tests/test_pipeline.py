import base64
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import numpy as np

from ingestion_app.classification import SavedClassifier, select_candidates
from ingestion_app.pipeline import PipelineRequest, run_pipeline
from server.app.services import AnnotationService
from server.app.settings import Settings
from server.app.sqlite_repository import SQLiteRepository


class Source:
    def __init__(self, dataset):
        self.dataset = dataset
        self.closed = False
        self.db = SimpleNamespace(close=self.close)

    def close(self):
        self.closed = True

    def match_window_ids(self):
        return [4, 3, 2, 1]

    def get_embedding(self, wid):
        return np.array({1: [-2., 1.], 2: [1., -2.], 3: [-1., 0.], 4: [0., 0.]}[wid], dtype=np.float32)

    def get_embeddings_batch(self, ids):
        return np.stack([self.get_embedding(wid) for wid in ids])

    def get_window(self, wid):
        return SimpleNamespace(recording_id=7, offsets=[wid - 1, wid])

    def get_recording(self, rid):
        return SimpleNamespace(id=7, filename='nested/recording.WAV', deployment_id=8)

    def get_deployment(self, did):
        return SimpleNamespace(id=8, name='nested', project=self.dataset, latitude=None, longitude=None)


class PipelineTests(unittest.TestCase):
    def test_boundary_selection_across_batches_and_species(self):
        classifier = SavedClassifier(('DENEBR', 'SCIBOU'), {}, np.eye(2, dtype=np.float32), np.zeros(2))
        source = Source('test')
        for batch_size in (1, 2, 20):
            result = select_candidates(source, [3, 2, 1], classifier, count=2, batch_size=batch_size)
            self.assertEqual(result, {'DENEBR': [2, 3], 'SCIBOU': [3, 1]})
        self.assertEqual(select_candidates(source, [], classifier), {'DENEBR': [], 'SCIBOU': []})

    def test_nonfinite_embeddings_rejected(self):
        classifier = SavedClassifier(('DENEBR',), {}, np.ones((2, 1)), np.zeros(1))
        source = SimpleNamespace(get_embeddings_batch=lambda _: np.array([[np.nan, 0.]]))
        with self.assertRaises(ValueError):
            select_candidates(source, [1], classifier)

    def write_artifact(self, path):
        data = dict(classes=['DENEBR', 'SCIBOU'], model_config={'embedding_dim': 2},
                    beta=base64.b64encode(np.eye(2, dtype=np.float32).tobytes()).decode(),
                    beta_bias=base64.b64encode(np.zeros(2, dtype=np.float32).tobytes()).decode())
        path.write_text(json.dumps(data), encoding='utf8')
        return data

    def test_classifier_dimensions_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.json'
            data = self.write_artifact(path)
            self.assertEqual(SavedClassifier.load(path).classes, ('DENEBR', 'SCIBOU'))
            self.assertEqual(SavedClassifier.from_bytes(path.read_bytes()).classes, ('DENEBR', 'SCIBOU'))
            data['model_config']['embedding_dim'] = 3
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                SavedClassifier.load(path)

    def test_pipeline_exports_importable_bundles_excludes_padded_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / 'audio'
            (audio / 'nested').mkdir(parents=True)
            with wave.open(str(audio / 'nested/recording.WAV'), 'wb') as source:
                source.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
                source.writeframes(b'\0\0' * 24000)
            artifact = root / 'classifier.pt'
            self.write_artifact(artifact)
            sources = []

            def fake_embed(audio_dir, db_path, dataset, config):
                sources.append(Source(dataset))
                return sources[-1]

            output = root / 'run'
            with patch('ingestion_app.pipeline.embed_folder', side_effect=fake_embed):
                result = run_pipeline(PipelineRequest(audio, artifact, output, 2), progress=lambda _: None)
            self.assertTrue(sources[0].closed)
            self.assertEqual(result['status'], 'ready_for_transfer')
            self.assertEqual(result['eligibleWindowCount'], 3)
            settings = Settings(local_dir=root / 'server', audio_roots={result['datasetName']: audio})
            repository = SQLiteRepository(settings.local_dir)
            repository.initialize()
            service = AnnotationService(repository, settings)
            for species, relative in result['manifests'].items():
                manifest = output / relative
                bundle = json.loads(manifest.read_text())
                self.assertNotIn(4, [c['sourceWindowId'] for c in bundle['candidates']])
                self.assertEqual(service.import_bundle(manifest)['candidateCount'], 2)
                service.import_bundle(manifest)
                self.assertEqual(len(repository.get_review_batch(species, 'tester', 15)['clips']), 2)
            with self.assertRaises(FileExistsError):
                run_pipeline(PipelineRequest(audio, artifact, output), progress=lambda _: None)

    def test_embedding_failure_persists_failed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / 'audio'
            audio.mkdir()
            with wave.open(str(audio / 'test.wav'), 'wb') as source:
                source.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
                source.writeframes(b'\0\0' * 8000)
            artifact = root / 'model.json'
            self.write_artifact(artifact)
            output = root / 'run'
            with patch('ingestion_app.pipeline.embed_folder', side_effect=RuntimeError('model unavailable')):
                with self.assertRaisesRegex(RuntimeError, 'model unavailable'):
                    run_pipeline(PipelineRequest(audio, artifact, output), progress=lambda _: None)
            self.assertEqual(json.loads((output / 'run.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
