"""Opt-in acceptance: OSA_REAL_BUNDLE=path/to/manifest.json.

Uses the real read-only Hoplite corpus and temporary annotation storage. Test
labels are synthetic and are never written into the normal application store.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('OSA_REAL_BUNDLE'), 'Set OSA_REAL_BUNDLE for real-corpus acceptance')
class RealCorpusAcceptance(unittest.TestCase):
    def test_actual_scores_clips_saves_correction_and_restart(self):
        import base64
        import numpy as np
        import soundfile as sf
        from fastapi.testclient import TestClient
        from ml_collections import config_dict
        from perch_hoplite.agile import audio_loader, source_info
        from perch_hoplite.db import sqlite_usearch_impl
        from server.app.main import create_app
        from server.app.services import AnnotationService
        from server.app.settings import Settings
        from server.app.sqlite_repository import SQLiteRepository

        path = Path(os.environ['OSA_REAL_BUNDLE'])
        manifest = json.loads(path.read_text(encoding='utf8'))
        classifier_path = Path('content/osa_db/agile_classifier.pt')
        cfg = json.loads(classifier_path.read_text())
        self.assertEqual(hashlib.sha256(classifier_path.read_bytes()).hexdigest(), manifest['classifierRun']['artifactSha256'])
        col = cfg['classes'].index(manifest['speciesId'])
        weights = np.frombuffer(base64.b64decode(cfg['beta']), dtype=np.float32).reshape(-1,len(cfg['classes']))[:,col]
        bias = np.frombuffer(base64.b64decode(cfg['beta_bias']),dtype=np.float32)[col]
        db = sqlite_usearch_impl.SQLiteUSearchDB.create('content/osa_db',readonly=True)
        try:
            metadata = config_dict.ConfigDict(db.get_metadata(None))
            model = metadata.model_config.model_config
            loader = audio_loader.make_filepath_loader(source_info.AudioSources.from_config_dict(config=metadata.audio_sources),
                sample_rate_hz=model.sample_rate,window_size_s=model.window_size_s)
            for c in manifest['candidates']:
                wid = c['sourceWindowId']
                window = db.get_window(wid)
                recording = db.get_recording(window.recording_id)
                self.assertEqual(list(window.offsets),c['offsets'])
                self.assertEqual(recording.id,c['recordingId'])
                self.assertEqual(recording.filename,c['uri'])
                self.assertAlmostEqual(float(np.dot(db.get_embedding(wid),weights)+bias),c['logit'],places=6)
                clip, rate = sf.read(path.parent/c['clip'])
                source = loader(recording.filename,window.offsets[0])
                self.assertEqual(rate,model.sample_rate)
                np.testing.assert_allclose(clip,source,atol=1/32768,rtol=0)
        finally:
            db.db.close()
        with tempfile.TemporaryDirectory() as local:
            settings = Settings(local_dir=Path(local))
            repo = SQLiteRepository(settings.local_dir)
            repo.initialize()
            service = AnnotationService(repo,settings)
            self.assertTrue(service.import_bundle(path)['imported'])
            self.assertFalse(service.import_bundle(path)['imported'])
            with TestClient(create_app(settings)) as client:
                batch = client.get(f'/api/species/{manifest["speciesId"]}/candidates').json()
                self.assertEqual(len(batch['clips']),min(15,len(manifest['candidates'])))
                expected = [f'{manifest["sourceDatasetId"]}:window:{c["sourceWindowId"]}' for c in manifest['candidates'][:15]]
                self.assertEqual([r['id'] for r in batch['clips']],expected)
                for r in batch['clips']:
                    self.assertEqual(client.get(r['audioUrl']).status_code,200)
                payload = dict(submissionId='synthetic-acceptance', speciesId=manifest['speciesId'],batchId=batch['batchId'],
                    createdAt='2026-11-02T00:00:00Z',annotations=[dict(windowId=r['id'],label=['yes','no','unsure'][i%3],
                    **{k:r[k] for k in ['recordingId','offsets','uri','datasetName']}) for i,r in enumerate(batch['clips'])])
                ack = client.post('/api/annotations',json=payload)
                self.assertEqual(ack.status_code,200,ack.text)
            with TestClient(create_app(settings)) as restarted:
                self.assertEqual(restarted.get(f'/api/species/{manifest["speciesId"]}/candidates').json(),batch)
                self.assertEqual(restarted.post('/api/annotations',json=payload).json(),ack.json())
                correction = dict(payload,submissionId='synthetic-correction',annotations=[dict(payload['annotations'][0],label='unsure')])
                self.assertEqual(restarted.post('/api/annotations',json=correction).status_code,200)
            with repo.connection() as conn:
                self.assertEqual(conn.execute('SELECT count(*) FROM annotation_events').fetchone()[0],len(batch['clips'])+1)
                self.assertEqual(conn.execute('SELECT label,revision FROM effective_annotations WHERE window_id=?',(expected[0],)).fetchone()[:],('unsure',2))
