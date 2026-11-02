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
    audio = root.parent / 'audio/site/audio.wav'
    audio.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(audio), 'wb') as wav:
        wav.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
        wav.writeframes(b'\0\0' * 8000 * max(count, 3))
    manifest = dict(schemaVersion=2, candidateSetId=set_id, speciesId=species,
        sourceDatasetId='test-corpus', createdAt='2026-11-02T00:00:00Z',
        classifierRun=dict(id='run-one', artifactSha256=None, embeddingModel='fixture', trainingSnapshotId=None),
        targetScore=0.0, selectionMode='boundary', candidates=[
            dict(sourceWindowId=i+1, recordingId=10, offsets=[float(i), float(i+1)], uri='site/audio.wav', datasetName='grillos',
                 logit=(i+1)*0.1, boundaryDistance=(i+1)*0.1, rank=i) for i in range(count)])
    path = root / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf8')
    return path, manifest


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = Settings(local_dir=self.root / 'store', audio_roots={'grillos': self.root/'audio'})
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
        self.assertEqual(self.client.get('/media/recordings/missing').status_code, 404)
        self.assertEqual(self.client.get('/media/recordings/%2e%2e%2fannotations.sqlite').status_code, 404)
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
        self.assertEqual(self.count('annotations'), 3)
        payload['annotations'][0]['label'] = 'no'
        self.assertEqual(self.client.post('/api/annotations', json=payload).status_code, 409)
        self.assertEqual(self.client.get('/api/submissions/submission-1').json(), first.json())
        self.assertEqual(self.client.get('/api/submissions/missing').status_code, 404)
        with TestClient(create_app(self.settings)) as restarted:
            self.assertEqual(restarted.get('/api/species/DENEBR/candidates').json()['clips'], [])
            payload['annotations'][0]['label'] = 'yes'
            self.assertEqual(restarted.post('/api/annotations', json=payload).json(), first.json())
        self.assertEqual(self.count('annotations'), 3)

    def test_atomic_validation_and_partial_completion(self):
        payload = self.payload()
        payload['annotations'][-1]['uri'] = 'forged.wav'
        self.assertEqual(self.client.post('/api/annotations', json=payload).status_code, 422)
        self.assertEqual(self.count('annotations'), 0)
        payload = self.payload()
        payload['annotations'] = payload['annotations'][:1]
        self.assertEqual(self.client.post('/api/annotations', json=payload).status_code, 200)
        self.assertEqual(len(self.batch()['clips']), 2)
        complete = self.payload(submission_id='complete')
        complete['annotations'][0]['label'] = 'unsure'
        self.assertEqual(self.client.post('/api/annotations', json=complete).status_code, 200)
        self.assertEqual(self.count('annotations'), 3)
        self.assertEqual(self.batch()['clips'], [])

    def test_corrections_and_annotator_isolation(self):
        original = self.payload()
        self.client.post('/api/annotations', json=original)
        correction = copy.deepcopy(original)
        correction['submissionId'] = 'correction'
        correction['annotations'] = correction['annotations'][:1]
        correction['annotations'][0]['label'] = 'unsure'
        self.assertEqual(self.client.post('/api/annotations', json=correction).status_code, 200)
        with self.repo.connection() as conn:
            row = conn.execute('SELECT label FROM annotations WHERE window_id=? ORDER BY rowid DESC LIMIT 1', (correction['annotations'][0]['windowId'],)).fetchone()
            self.assertEqual(tuple(row), ('unsure',))
        other = Settings(local_dir=self.settings.local_dir, annotator='another-annotator', audio_roots=self.settings.audio_roots)
        with TestClient(create_app(other)) as client:
            self.assertEqual(client.get('/api/submissions/submission-1').status_code, 404)
            self.assertEqual(client.post('/api/annotations', json=correction).status_code, 409)
            batch = client.get('/api/species/DENEBR/candidates').json()
            foreign = self.payload(batch, 'other')
            foreign['annotations'][0]['label'] = 'no'
            self.assertEqual(client.post('/api/annotations', json=foreign).status_code, 200)
        self.assertEqual(self.count('annotations'), 7)

    def test_concurrent_timeout_retry_has_one_commit(self):
        payload = Submission.model_validate(self.payload())
        # Ignore the first result, simulating a lost acknowledgement after commit.
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: self.repo.save_submission(payload, self.settings.annotator), range(12)))
        self.assertTrue(all(r == results[0] for r in results))
        self.assertEqual(self.count('annotations'), 3)

    def test_concurrent_review_queries_return_same_unreviewed_windows(self):
        with ThreadPoolExecutor(max_workers=6) as pool:
            batches = list(pool.map(lambda _: self.repo.get_review_batch('DENEBR', 'concurrent', 15), range(12)))
        self.assertTrue(all(b == batches[0] for b in batches))
        self.assertEqual(self.count('annotations'), 0)

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
        self.assertEqual(self.count('annotations'),0)

    def test_import_idempotency_conflicts_and_identity_validation(self):
        self.assertFalse(self.service.import_bundle(self.manifest_path)['imported'])
        changed = copy.deepcopy(self.manifest)
        # Export metadata does not create another candidate for the same window/species.
        changed['createdAt'] = '2026-11-03T00:00:00Z'
        changed['candidateSetId'] = 'second-set'
        changed['classifierRun']['id'] = 'new-run'
        self.manifest_path.write_text(json.dumps(changed))
        self.assertFalse(self.service.import_bundle(self.manifest_path)['imported'])
        changed['candidates'][1]['offsets'] = [0, 1]
        self.manifest_path.write_text(json.dumps(changed))
        with self.assertRaises(AppError):
            self.service.import_bundle(self.manifest_path)
        self.assertEqual(self.count('candidates'), 3)
        self.assertEqual(self.count('windows'), 3)

    def test_import_rejects_invalid_bundles_without_partial_publish(self):
        for change in ['missing','traversal','absolute','duplicate','unknown-species','nan','distance','rank','duration','recording-conflict']:
            with self.subTest(change=change):
                manifest = copy.deepcopy(self.manifest)
                manifest['candidateSetId'] = change
                c = manifest['candidates'][0]
                if change == 'missing': c['uri'] = 'missing.wav'
                if change == 'traversal': c['uri'] = '../outside.wav'
                if change == 'absolute': c['uri'] = str(self.root / 'outside.wav')
                if change == 'duplicate': manifest['candidates'][1]['sourceWindowId'] = c['sourceWindowId']
                if change == 'unknown-species': manifest['speciesId'] = 'UNKNOWN'
                if change == 'nan': c['logit'] = float('nan')
                if change == 'distance': c['boundaryDistance'] = 0
                if change == 'rank': c['rank'] = 3
                if change == 'duration': c['offsets'] = [0,999]
                if change == 'recording-conflict': c['uri'] = 'conflicting.wav'
                self.manifest_path.write_text(json.dumps(manifest))
                with self.assertRaises((AppError,ValidationError)):
                    self.service.import_bundle(self.manifest_path)
                self.assertEqual(self.count('candidates'),3)

    def test_corrupt_recording_and_missing_registered_file(self):
        audio = self.root/'audio/site/audio.wav'
        audio.write_bytes(b'not a wav')
        with self.assertRaises(AppError):
            self.service.import_bundle(self.manifest_path)
        batch = self.batch()
        audio.unlink()
        self.assertEqual(self.client.get(batch['clips'][0]['audioUrl']).status_code, 404)

    def test_recording_path_resolution_rejects_unsafe_paths_and_unknown_roots(self):
        for uri in ['../outside.wav', '/absolute.wav', 'C:/outside.wav', 'site\\audio.wav']:
            with self.subTest(uri=uri):
                with self.assertRaises(AppError) as error:
                    self.service.recording_path({'dataset_name': 'grillos', 'file_path': uri})
                self.assertEqual(error.exception.status, 422)
        with self.assertRaises(AppError):
            self.service.recording_path({'dataset_name': 'unconfigured', 'file_path': 'audio.wav'})

    def test_recording_url_shared_by_windows_and_range_uses_full_source(self):
        batch = self.batch()
        urls = {clip['audioUrl'] for clip in batch['clips']}
        self.assertEqual(len(urls), 1)
        for clip in batch['clips']:
            self.assertEqual([clip['startSeconds'], clip['endSeconds']], clip['offsets'])
            self.assertNotIn('clipId', clip)
        source = (self.root/'audio/site/audio.wav').read_bytes()
        self.assertEqual(self.client.get(next(iter(urls))).content, source)
        response = self.client.get(next(iter(urls)), headers={'Range': 'bytes=100-199'})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.content, source[100:200])
        with self.repo.connection() as conn:
            fields = {row[1] for row in conn.execute('PRAGMA table_info(windows)')}
        self.assertEqual(fields, {'id', 'recording_id', 'source_window_id',
                                  'start_seconds', 'end_seconds', 'spectrogram_path'})
        self.assertFalse((self.settings.local_dir/'clips').exists())

    def test_maximum_batch_and_new_candidates_follow_existing_windows(self):
        path, manifest = make_bundle(self.root/'large', count=20, set_id='large', species='BOAROS')
        self.service.import_bundle(path)
        batch = self.client.get('/api/species/BOAROS/candidates').json()
        self.assertEqual(len(batch['clips']),15)
        before = self.batch()
        manifest['speciesId'] = 'DENEBR'
        manifest['candidateSetId'] = 'new-denebr'
        path.write_text(json.dumps(manifest))
        self.service.import_bundle(path)
        after = self.batch()
        self.assertEqual(before['clips'], after['clips'][:len(before['clips'])])
        self.assertEqual(len(after['clips']), 15)

    def test_only_four_tables_and_latest_answer_history(self):
        with self.repo.connection() as conn:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
        self.assertEqual(tables, {'recordings', 'windows', 'annotations', 'candidates'})
        payload = self.payload()
        self.assertEqual(self.client.post('/api/annotations', json=payload).status_code, 200)
        payload['submissionId'] = 'fresh-save-same-labels'
        self.assertEqual(self.client.post('/api/annotations', json=payload).status_code, 200)
        self.assertEqual(self.count('annotations'), 6)
        with self.repo.connection() as conn:
            latest = conn.execute('''
                SELECT a.* FROM annotations a WHERE NOT EXISTS (
                    SELECT 1 FROM annotations newer
                    WHERE newer.window_id=a.window_id AND newer.species=a.species
                    AND newer.annotator=a.annotator AND newer.rowid>a.rowid
                )
            ''').fetchall()
        self.assertEqual(len(latest), 3)
        self.assertTrue(all(row['submission_id'] == payload['submissionId'] for row in latest))

    def test_refresh_advances_to_remaining_candidates(self):
        path, _ = make_bundle(self.root/'queue', count=20, species='BOAROS', set_id='queue')
        self.service.import_bundle(path)
        batch = self.client.get('/api/species/BOAROS/candidates').json()
        payload = self.payload(batch)
        payload['speciesId'] = 'BOAROS'
        self.assertEqual(self.client.post('/api/annotations', json=payload).status_code, 200)
        remaining = self.client.get('/api/species/BOAROS/candidates').json()
        self.assertEqual(len(remaining['clips']), 5)
        self.assertTrue(set(r['id'] for r in batch['clips']).isdisjoint(
            r['id'] for r in remaining['clips']
        ))
        # The old context still permits corrections after the queue advances.
        payload['submissionId'] = 'correct-old-page'
        payload['annotations'] = [dict(payload['annotations'][0], label='no')]
        self.assertEqual(self.client.post('/api/annotations', json=payload).status_code, 200)

    def test_minimal_recording_and_annotation_columns_and_server_timestamp(self):
        with self.repo.connection() as conn:
            recording_columns = {row[1] for row in conn.execute('PRAGMA table_info(recordings)')}
            annotation_columns = {row[1] for row in conn.execute('PRAGMA table_info(annotations)')}
        self.assertEqual(recording_columns, {'id', 'source_recording_id', 'file_path', 'dataset_name'})
        self.assertEqual(annotation_columns, {'id', 'window_id', 'species', 'annotator',
                                              'label', 'submission_id', 'created_at'})
        payload = self.payload()
        self.assertEqual(self.client.post('/api/annotations', json=payload).status_code, 200)
        with self.repo.connection() as conn:
            timestamp = conn.execute('SELECT created_at FROM annotations LIMIT 1').fetchone()[0]
        from datetime import datetime, timezone
        timestamp = datetime.fromisoformat(timestamp)
        self.assertEqual(timestamp.utcoffset().total_seconds(), 0)
        self.assertLess(abs((datetime.now(timezone.utc) - timestamp).total_seconds()), 30)
        # Client time is not persisted or used to identify answer retries.
        payload['createdAt'] = '2020-01-01T00:00:00Z'
        self.assertEqual(self.client.post('/api/annotations', json=payload).status_code, 200)
        self.assertEqual(self.count('annotations'), 3)
        for change in ['missing-answer', 'extra-answer', 'metadata']:
            changed = copy.deepcopy(payload)
            if change == 'missing-answer': changed['annotations'].pop()
            if change == 'extra-answer':
                changed['annotations'].append(dict(changed['annotations'][0], windowId='unknown'))
            if change == 'metadata': changed['annotations'][0]['uri'] = 'forged.wav'
            self.assertEqual(self.client.post('/api/annotations', json=changed).status_code, 409)

    def test_old_schema_rejected_without_reset(self):
        with self.repo.connection(True) as conn:
            conn.execute('PRAGMA user_version=1')
        with self.assertRaisesRegex(RuntimeError, 'fresh store'):
            self.repo.initialize()
        self.assertEqual(self.count('windows'), 3)

    def test_future_schema_rejected_without_reset(self):
        with self.repo.connection(True) as conn: conn.execute('PRAGMA user_version=99')
        with self.assertRaises(RuntimeError): self.repo.initialize()
        self.assertEqual(self.count('windows'),3)

    def test_minimal_candidate_columns_and_computed_annotation_count(self):
        with self.repo.connection() as conn:
            fields = {row[1] for row in conn.execute('PRAGMA table_info(candidates)')}
        self.assertEqual(fields, {'id', 'window_id', 'species'})
        batch = self.batch()
        self.assertTrue(all(clip['numAnnotations'] == 0 for clip in batch['clips']))
        payload = self.payload(batch)
        payload['annotations'] = payload['annotations'][:1]
        self.assertEqual(self.client.post('/api/annotations', json=payload).status_code, 200)
        payload['submissionId'] = 'correction'
        payload['annotations'][0]['label'] = 'no'
        self.assertEqual(self.client.post('/api/annotations', json=payload).status_code, 200)
        other = self.repo.get_review_batch('DENEBR', 'someone-else', 15)
        counts = {clip['id']: clip['numAnnotations'] for clip in other['clips']}
        self.assertEqual(counts[payload['annotations'][0]['windowId']], 2)
        self.assertEqual(sorted(counts.values()), [0, 0, 2])
        # A different species' labels must not increase this candidate's count.
        manifest = copy.deepcopy(self.manifest)
        manifest['speciesId'] = 'BOAROS'
        self.manifest_path.write_text(json.dumps(manifest))
        self.service.import_bundle(self.manifest_path)
        self.assertTrue(all(clip['numAnnotations'] == 0 for clip in
                            self.repo.get_review_batch('BOAROS', 'someone-else', 15)['clips']))

    def test_concurrent_initialization_and_foreign_keys(self):
        store = self.root/'fresh'
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: SQLiteRepository(store).initialize(),range(8)))
        repository = SQLiteRepository(store)
        repository.health()
        with repository.connection() as conn:
            self.assertEqual(conn.execute('PRAGMA foreign_keys').fetchone()[0],1)
            self.assertEqual(conn.execute('PRAGMA user_version').fetchone()[0],6)
            self.assertEqual(conn.execute('PRAGMA foreign_key_check').fetchall(),[])

    def test_transaction_failure_rolls_back_answers_and_retry_metadata(self):
        payload = self.payload()
        # Force a failure after the first answer has been inserted.
        with self.repo.connection(True) as conn:
            conn.execute("CREATE TRIGGER fail_second BEFORE INSERT ON annotations WHEN NEW.window_id='test-corpus:window:2' BEGIN SELECT RAISE(ABORT,'simulated write failure'); END")
        with self.assertLogs('server.app.main', level='ERROR'):
            response = self.client.post('/api/annotations',json=payload)
        self.assertEqual(response.status_code,503)
        self.assertEqual(self.count('annotations'),0)
        with self.repo.connection(True) as conn: conn.execute('DROP TRIGGER fail_second')
        self.assertEqual(self.client.post('/api/annotations',json=payload).status_code,200)


if __name__ == '__main__':
    unittest.main()
