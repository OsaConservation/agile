from pathlib import Path
import sqlite3
import tempfile
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from server.app.main import create_app
from server.app.services import AnnotationService
from server.app.settings import Settings
from server.app.sqlite_repository import SQLiteRepository
from server.tests.test_server import make_bundle
from server.tools.generate_spectrograms import generate_spectrograms


class SpectrogramTests(unittest.TestCase):
    def test_v5_migration_preserves_existing_annotations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            conn = sqlite3.connect(root / 'annotations.sqlite')
            try:
                conn.executescript((Path(__file__).parents[1] / 'app/migrations/001_initial.sql').read_text())
                conn.execute("INSERT INTO recordings VALUES('r',1,'audio.wav','fixture')")
                conn.execute("INSERT INTO windows VALUES('w','r',1,0,1)")
                conn.execute("INSERT INTO candidates VALUES('c','w','DENEBR')")
                conn.execute("INSERT INTO annotations VALUES('a','w','DENEBR','person','yes','submission','2026-11-02T00:00:00Z')")
                conn.commit()
            finally:
                conn.close()
            repository = SQLiteRepository(root)
            repository.initialize()
            repository.initialize()
            with repository.connection() as conn:
                self.assertEqual(conn.execute('SELECT label FROM annotations').fetchone()[0], 'yes')
                self.assertEqual(conn.execute('SELECT count(*) FROM candidates').fetchone()[0], 1)
                self.assertIsNone(conn.execute('SELECT spectrogram_path FROM windows').fetchone()[0])

    def test_migrate_generate_serve_retry_and_path_safety(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(local_dir=root / 'store', audio_roots={'grillos': root / 'audio'})
            settings.local_dir.mkdir()
            # Build a genuine v5 database, then exercise the additive migration.
            with sqlite3.connect(settings.local_dir / 'annotations.sqlite') as conn:
                conn.executescript((Path(__file__).parents[1] / 'app/migrations/001_initial.sql').read_text())
            conn.close()
            repository = SQLiteRepository(settings.local_dir)
            repository.initialize()
            service = AnnotationService(repository, settings)
            manifest, _ = make_bundle(root / 'bundle', count=2)
            service.import_bundle(manifest)
            with TestClient(create_app(settings, repository)) as client:
                before = client.get('/api/species/DENEBR/candidates').json()['clips']
                self.assertIsNone(before[0]['spectrogramUrl'])
                result = generate_spectrograms(settings, repository)
                self.assertEqual(result, {'generated': 2, 'skipped': 0})
                self.assertEqual(generate_spectrograms(settings, repository), {'generated': 0, 'skipped': 2})
                # Reimport must not treat the new optional column as an identity conflict.
                service.import_bundle(manifest)
                clips = client.get('/api/species/DENEBR/candidates').json()['clips']
                self.assertNotEqual(clips[0]['spectrogramUrl'], clips[1]['spectrogramUrl'])
                self.assertNotIn('spectrogramPath', clips[0])
                response = client.get(clips[0]['spectrogramUrl'])
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers['content-type'], 'image/webp')
                path = service.get_window_spectrogram(clips[0]['id'])
                with Image.open(path) as image:
                    self.assertEqual(image.size, (1000, 300))
                path.unlink()
                self.assertEqual(client.get(clips[0]['spectrogramUrl']).status_code, 404)
                self.assertEqual(generate_spectrograms(settings, repository)['generated'], 1)
                repository.set_spectrogram(clips[0]['id'], '../annotations.sqlite')
                self.assertEqual(client.get(clips[0]['spectrogramUrl']).status_code, 404)
                self.assertEqual(client.get('/media/windows/unknown/spectrogram').status_code, 404)
                self.assertEqual(client.get('/health').json()['schemaVersion'], 6)
                with repository.connection() as conn:
                    self.assertEqual(conn.execute('SELECT count(*) FROM candidates').fetchone()[0], 2)


if __name__ == '__main__':
    unittest.main()
