"""Repository/HTTP contract tests; no research runtime or source database writes."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import wave

from fastapi.testclient import TestClient
from pydantic import ValidationError

from server.app.main import create_app
from server.app.models import Submission
from server.app.repository import AppError
from server.app.services import AnnotationService
from server.app.settings import Settings
from server.app.sqlite_repository import SQLiteRepository


def make_bundle(root, count=3, set_id='set-one', species='DENEBR'):
    root.mkdir(parents=True, exist_ok=True)
    with wave.open(str(root / 'clip.wav'), 'wb') as wav:
        wav.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
        wav.writeframes(b'\0\0' * 8000)
    manifest = dict(schemaVersion=1, candidateSetId=set_id, speciesId=species,
        sourceDatasetId='test-corpus', createdAt='2026-11-02T00:00:00Z',
        classifierRun=dict(id='run-one', artifactSha256=None, embeddingModel='fixture', trainingSnapshotId=None),
        targetScore=0.0, selectionMode='boundary', candidates=[
            dict(sourceWindowId=i+1, recordingId=10, offsets=[float(i), float(i+1)], uri='site/audio.wav', datasetName='grillos',
                 logit=(i+1)*0.1, boundaryDistance=(i+1)*0.1, rank=i, clip='clip.wav') for i in range(count)])
    path = root / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf8')
    return path, manifest


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = Settings(local_dir=self.root / 'store')
        self.repo = SQLiteRepository(self.settings.local_dir)
        self.repo.initialize()
        self.service = AnnotationService(self.repo, self.settings)
        self.manifest_path, self.manifest = make_bundle(self.root / 'bundle')
        self.service.import_bundle(self.manifest_path)
        self.client = TestClient(create_app(self.settings, self.repo))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def batch(self):
        response = self.client.get('/api/species/DENEBR/candidates')
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def payload(self, batch=None, submission_id='submission-1'):
        batch = batch or self.batch()
        return dict(submissionId=submission_id, speciesId='DENEBR', batchId=batch['batchId'],
            createdAt='2026-11-02T00:00:00Z', annotations=[
                dict(windowId=r['id'], **{k: r[k] for k in ['recordingId','offsets','uri','datasetName']}, label='yes')
                for r in batch['clips']])

    def count(self, table):
        with self.repo.connection() as conn:
            return conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0]

    def test_frontend_contract_audio_cors_and_empty_species(self):
        self.assertEqual(self.client.get('/health').json()['status'], 'ok')
        species = self.client.get('/api/species').json()
        self.assertIn('DENEBR', {s['id'] for s in species})
        refs = self.client.get('/api/species/DENEBR/references').json()
        self.assertEqual(refs['target']['id'], 'DENEBR')
        self.assertEqual(refs['target']['clips'], [])
        self.assertNotIn('examples', refs['target'])
        self.assertTrue(all('clips' in species for species in refs['comparisons']))
        self.assertNotIn('DENEBR', {s['id'] for s in refs['comparisons']})
        batch = self.batch()
        self.assertNotIn('recordings', batch)
        self.assertEqual(len(batch['clips']), 3)
        self.assertEqual(batch, self.batch())
        self.assertEqual(batch['clips'][0]['offsets'], [0.0, 1.0])
        audio = self.client.get(batch['clips'][0]['audioUrl'])
        self.assertEqual(audio.status_code, 200)
        self.assertTrue(audio.content.startswith(b'RIFF'))
        self.assertEqual(audio.headers['content-type'], 'audio/wav')
        partial = self.client.get(batch['clips'][0]['audioUrl'], headers={'Range': 'bytes=0-43'})
        self.assertEqual(partial.status_code, 206)
        self.assertEqual(len(partial.content), 44)
        for origin, allowed in [('http://127.0.0.1:8000', True), ('http://localhost:8000', True), ('http://evil.example', False)]:
            response = self.client.options('/api/annotations', headers={'Origin':origin, 'Access-Control-Request-Method':'POST', 'Access-Control-Request-Headers':'Content-Type'})
            self.assertEqual(response.headers.get('access-control-allow-origin') == origin, allowed)
        self.assertEqual(self.client.get('/api/species/UNKNOWN/candidates').status_code, 404)
        self.assertEqual(self.client.get('/api/species/BOAROS/candidates').json(), {'batchId':None, 'clips':[]})
        self.assertEqual(self.client.get('/media/clips/missing').status_code, 404)
        self.assertEqual(self.client.get('/media/clips/%2e%2e%2fannotations.sqlite').status_code, 404)
        self.assertIn('/api/annotations', self.client.get('/openapi.json').json()['paths'])
        paths = self.client.get('/openapi.json').json()['paths']
        self.assertIn('/api/species/{species_id}/candidates', paths)
        self.assertNotIn('/api/species/{species_id}/recordings', paths)

    def test_retry_reorder_conflict_and_restart(self):
        batch = self.batch()
        payload = self.payload(batch)
        first = self.client.post('/api/annotations', json=payload)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json(), {'submissionId':'submission-1', 'savedCount':3})
        payload['annotations'].reverse()
        self.assertEqual(self.client.post('/api/annotations', json=payload).json(), first.json())
        self.assertEqual(self.count('annotation_events'), 3)
        payload['annotations'][0]['label'] = 'no'
        self.assertEqual(self.client.post('/api/annotations', json=payload).status_code, 409)
        self.assertEqual(self.client.get('/api/submissions/submission-1').json(), first.json())
        self.assertEqual(self.client.get('/api/submissions/missing').status_code, 404)
        with TestClient(create_app(self.settings)) as restarted:
            self.assertEqual(restarted.get('/api/species/DENEBR/candidates').json(), batch)
            payload['annotations'][0]['label'] = 'yes'
            self.assertEqual(restarted.post('/api/annotations', json=payload).json(), first.json())
        self.assertEqual(self.count('annotation_events'), 3)

    def test_atomic_validation_and_partial_completion(self):
        payload = self.payload()
        payload['annotations'][-1]['uri'] = 'forged.wav'
        self.assertEqual(self.client.post('/api/annotations', json=payload).status_code, 422)
        self.assertEqual(self.count('submissions'), 0)
        self.assertEqual(self.count('annotation_events'), 0)
        payload = self.payload()
        payload['annotations'] = payload['annotations'][:1]
        self.assertEqual(self.client.post('/api/annotations', json=payload).status_code, 200)
        with self.repo.connection() as conn:
            self.assertIsNone(conn.execute('SELECT completed_at FROM batches').fetchone()[0])
        self.assertEqual(len(self.batch()['clips']), 3)
        complete = self.payload(submission_id='complete')
        complete['annotations'][1]['label'] = 'unsure'
        self.assertEqual(self.client.post('/api/annotations', json=complete).status_code, 200)
        self.assertEqual(self.count('annotation_events'), 3)
        with self.repo.connection() as conn:
            self.assertIsNotNone(conn.execute('SELECT completed_at FROM batches').fetchone()[0])

    def test_corrections_and_annotator_isolation(self):
        original = self.payload()
        self.client.post('/api/annotations', json=original)
        correction = copy.deepcopy(original)
        correction['submissionId'] = 'correction'
        correction['annotations'] = correction['annotations'][:1]
        correction['annotations'][0]['label'] = 'unsure'
        self.assertEqual(self.client.post('/api/annotations', json=correction).status_code, 200)
        with self.repo.connection() as conn:
            row = conn.execute('SELECT label,revision FROM effective_annotations WHERE window_id=?', (correction['annotations'][0]['windowId'],)).fetchone()
            self.assertEqual(tuple(row), ('unsure',2))
        other = Settings(local_dir=self.settings.local_dir, annotator='another-annotator')
        with TestClient(create_app(other)) as client:
            self.assertEqual(client.get('/api/submissions/submission-1').status_code, 404)
            self.assertEqual(client.post('/api/annotations', json=correction).status_code, 409)
            batch = client.get('/api/species/DENEBR/candidates').json()
            foreign = self.payload(batch, 'other')
            foreign['annotations'][0]['label'] = 'no'
            self.assertEqual(client.post('/api/annotations', json=foreign).status_code, 200)
        self.assertEqual(self.count('effective_annotations'), 6)
        self.assertEqual(self.count('annotation_events'), 7)

    def test_concurrent_timeout_retry_has_one_commit(self):
        payload = Submission.model_validate(self.payload())
        # Ignore the first result, simulating a lost acknowledgement after commit.
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: self.repo.save_submission(payload, self.settings.annotator), range(12)))
        self.assertTrue(all(r == results[0] for r in results))
        self.assertEqual(self.count('submissions'), 1)
        self.assertEqual(self.count('annotation_events'), 3)

    def test_concurrent_batch_assignment_stays_stable(self):
        with ThreadPoolExecutor(max_workers=6) as pool:
            batches = list(pool.map(lambda _: self.repo.get_review_batch('DENEBR', 'concurrent', 15), range(12)))
        self.assertTrue(all(b == batches[0] for b in batches))
        self.assertEqual(self.count('batches'), 1)

    def test_invalid_submission_shapes(self):
        for change in ['duplicate','invalid-label','naive-time','unknown-window','wrong-species','bad-offset']:
            with self.subTest(change=change):
                payload = self.payload()
                if change == 'duplicate': payload['annotations'].append(payload['annotations'][0])
                if change == 'invalid-label': payload['annotations'][0]['label'] = 'maybe'
                if change == 'naive-time': payload['createdAt'] = '2026-11-02T00:00:00'
                if change == 'unknown-window': payload['annotations'][0]['windowId'] = 'missing'
                if change == 'wrong-species': payload['speciesId'] = 'BOAROS'
                if change == 'bad-offset': payload['annotations'][0]['offsets'] = [2,1]
                self.assertIn(self.client.post('/api/annotations',json=payload).status_code, [404,422])
        self.assertEqual(self.count('submissions'),0)

    def test_import_idempotency_conflicts_and_identity_validation(self):
        self.assertFalse(self.service.import_bundle(self.manifest_path)['imported'])
        changed = copy.deepcopy(self.manifest)
        changed['createdAt'] = '2026-11-03T00:00:00Z'
        self.manifest_path.write_text(json.dumps(changed))
        with self.assertRaises(AppError) as error:
            self.service.import_bundle(self.manifest_path)
        self.assertEqual(error.exception.status,409)
        changed['candidateSetId'] = 'second-set'
        changed['candidates'][1]['offsets'] = [4,5]
        self.manifest_path.write_text(json.dumps(changed))
        with self.assertRaises(AppError): self.service.import_bundle(self.manifest_path)
        self.assertEqual(self.count('candidate_sets'),1)
        self.assertEqual(self.count('windows'),3)

    def test_import_rejects_invalid_bundles_without_partial_publish(self):
        for change in ['missing','traversal','absolute','duplicate','unknown-species','nan','distance','rank','duration','recording-conflict']:
            with self.subTest(change=change):
                manifest = copy.deepcopy(self.manifest)
                manifest['candidateSetId'] = change
                c = manifest['candidates'][0]
                if change == 'missing': c['clip'] = 'missing.wav'
                if change == 'traversal': c['clip'] = '../bundle/clip.wav'
                if change == 'absolute': c['clip'] = str(self.root / 'bundle/clip.wav')
                if change == 'duplicate': manifest['candidates'][1]['sourceWindowId'] = c['sourceWindowId']
                if change == 'unknown-species': manifest['speciesId'] = 'UNKNOWN'
                if change == 'nan': c['logit'] = float('nan')
                if change == 'distance': c['boundaryDistance'] = 0
                if change == 'rank': c['rank'] = 3
                if change == 'duration': c['offsets'] = [0,2]
                if change == 'recording-conflict': c['uri'] = 'conflicting.wav'
                self.manifest_path.write_text(json.dumps(manifest))
                with self.assertRaises((AppError,ValidationError)):
                    self.service.import_bundle(self.manifest_path)
                self.assertEqual(self.count('candidate_sets'),1)

    def test_corrupt_media_and_missing_registered_file(self):
        (self.manifest_path.parent / 'clip.wav').write_bytes(b'not a wav')
        with self.assertRaises(AppError): self.service.import_bundle(self.manifest_path)
        batch = self.batch()
        path = next((self.settings.local_dir / 'clips').glob('*.wav'))
        path.unlink()
        self.assertEqual(self.client.get(batch['clips'][0]['audioUrl']).status_code,404)

    def test_maximum_batch_and_import_does_not_replace_membership(self):
        path, manifest = make_bundle(self.root/'large', count=20, set_id='large', species='BOAROS')
        self.service.import_bundle(path)
        batch = self.client.get('/api/species/BOAROS/candidates').json()
        self.assertEqual(len(batch['clips']),15)
        before = self.batch()
        manifest['speciesId'] = 'DENEBR'
        manifest['candidateSetId'] = 'new-denebr'
        path.write_text(json.dumps(manifest))
        self.service.import_bundle(path)
        self.assertEqual(before,self.batch())

    def test_future_schema_rejected_without_reset(self):
        with self.repo.connection(True) as conn: conn.execute('PRAGMA user_version=99')
        with self.assertRaises(RuntimeError): self.repo.initialize()
        self.assertEqual(self.count('windows'),3)

    def test_classifier_run_provenance_is_immutable(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['candidateSetId'] = 'different-set'
        manifest['classifierRun']['embeddingModel'] = 'different-model'
        self.manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(AppError) as error:
            self.service.import_bundle(self.manifest_path)
        self.assertEqual(error.exception.status,409)
        self.assertEqual(self.count('candidate_sets'),1)

    def test_concurrent_initialization_and_foreign_keys(self):
        store = self.root/'fresh'
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: SQLiteRepository(store).initialize(),range(8)))
        repository = SQLiteRepository(store)
        repository.health()
        with repository.connection() as conn:
            self.assertEqual(conn.execute('PRAGMA foreign_keys').fetchone()[0],1)
            self.assertEqual(conn.execute('PRAGMA user_version').fetchone()[0],1)
            self.assertEqual(conn.execute('PRAGMA foreign_key_check').fetchall(),[])

    def test_transaction_failure_rolls_back_events_and_acknowledgement(self):
        payload = self.payload()
        # Force a failure after the submission and first answer have been inserted.
        with self.repo.connection(True) as conn:
            conn.execute("CREATE TRIGGER fail_second BEFORE INSERT ON annotation_events WHEN NEW.window_id='test-corpus:window:2' BEGIN SELECT RAISE(ABORT,'simulated write failure'); END")
        with self.assertLogs('server.app.main', level='ERROR'):
            response = self.client.post('/api/annotations',json=payload)
        self.assertEqual(response.status_code,503)
        self.assertEqual(self.count('submissions'),0)
        self.assertEqual(self.count('annotation_events'),0)
        self.assertEqual(self.count('effective_annotations'),0)
        with self.repo.connection(True) as conn: conn.execute('DROP TRIGGER fail_second')
        self.assertEqual(self.client.post('/api/annotations',json=payload).status_code,200)


if __name__ == '__main__':
    unittest.main()
