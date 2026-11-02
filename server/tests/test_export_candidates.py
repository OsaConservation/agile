import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import wave

import numpy as np

from server.app.services import AnnotationService
from server.app.settings import Settings
from server.app.sqlite_repository import SQLiteRepository
from server.tools.export_candidates import export_candidates


class FakeSource:
    def get_embedding(self, wid):
        return {1:np.array([-2.,0.]), 2:np.array([1.,0.]), 3:np.array([-1.,0.])}[wid]

    def get_window(self, wid):
        return SimpleNamespace(recording_id=7, offsets=[wid, wid+1])

    def get_recording(self, rid):
        return SimpleNamespace(id=rid, filename='site/recording.wav', deployment_id=4)

    def get_deployment(self, did):
        return SimpleNamespace(id=did, name='site', project='grillos', latitude=None, longitude=None)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def loader(self, filename, offset):
        self.calls.append((filename, offset))
        return np.linspace(-0.5,0.5,8000, dtype=np.float32)

    def export(self, **changes):
        args = dict(results=SimpleNamespace(search_results=[SimpleNamespace(window_id=i, sort_score=-100) for i in [3,1,2]]),
            db=FakeSource(), weights=np.array([1.,0.]), bias=0., audio_loader=self.loader,
            output_dir=self.root/'bundle', species_id='DENEBR', source_dataset_id='test',
            embedding_model='fixture', sample_rate=8000, count=15)
        return export_candidates(**(args | changes))

    def test_signed_logits_tie_order_audio_offsets_and_import(self):
        path = self.export()
        bundle = json.loads(path.read_text())
        self.assertEqual([c['sourceWindowId'] for c in bundle['candidates']], [2,3,1])
        self.assertEqual([c['logit'] for c in bundle['candidates']], [1.,-1.,-2.])
        self.assertEqual(self.calls, [('site/recording.wav',2.),('site/recording.wav',3.),('site/recording.wav',1.)])
        for c in bundle['candidates']:
            with wave.open(str(path.parent/c['clip'])) as wav:
                self.assertEqual(wav.getnframes(),8000)
                self.assertEqual(wav.getsampwidth(),2)
        settings = Settings(local_dir=self.root/'store')
        repository = SQLiteRepository(settings.local_dir)
        repository.initialize()
        self.assertTrue(AnnotationService(repository,settings).import_bundle(path)['imported'])
        self.assertEqual(len(repository.get_review_batch('DENEBR','test',15)['clips']),3)

    def test_nonzero_target_and_count(self):
        bundle = json.loads(self.export(target_score=-1.5, count=2).read_text())
        self.assertEqual([c['sourceWindowId'] for c in bundle['candidates']],[1,3])
        self.assertEqual([c['boundaryDistance'] for c in bundle['candidates']],[0.5,0.5])

    def test_failed_export_does_not_publish_bundle(self):
        with self.assertRaises(ValueError):
            self.export(audio_loader=lambda *_:np.zeros(4000))
        self.assertFalse((self.root/'bundle').exists())
        self.assertFalse(list(self.root.glob('export-*')))

    def test_unknown_species_and_existing_output_rejected(self):
        with self.assertRaises(ValueError): self.export(species_id='UNKNOWN')
        self.assertEqual(self.calls,[])
        self.export()
        with self.assertRaises(ValueError): self.export()

    def test_artifact_hash_attests_actual_scoring_weights(self):
        import base64
        import hashlib
        artifact = self.root/'classifier.pt'
        cfg = dict(classes=['DENEBR'], model_config={'name':'fixture'},
            beta=base64.b64encode(np.array([1.,0.],dtype=np.float32).tobytes()).decode(),
            beta_bias=base64.b64encode(np.array([0.],dtype=np.float32).tobytes()).decode())
        artifact.write_text(json.dumps(cfg))
        with self.assertRaises(ValueError):
            self.export(classifier_path=artifact,bias=1.)
        self.assertFalse((self.root/'bundle').exists())
        manifest = json.loads(self.export(classifier_path=artifact,embedding_model=cfg['model_config']).read_text())
        self.assertEqual(manifest['classifierRun']['artifactSha256'],hashlib.sha256(artifact.read_bytes()).hexdigest())
