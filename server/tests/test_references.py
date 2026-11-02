"""Reference import and HTTP playback use isolated fixtures."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from server.app.main import create_app
from server.app.references import ReferenceCatalog
from server.app.settings import Settings
from server.tools.import_references import import_references


class ReferenceTests(unittest.TestCase):
    def test_unreadable_catalog_is_not_treated_as_empty(self):
        with patch.object(Path, 'read_text', side_effect=PermissionError('Access denied')):
            with self.assertRaises(PermissionError):
                ReferenceCatalog(Path('references'))

    def test_failed_import_removes_staging_and_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frontend = root / 'frontend'
            frontend.mkdir()
            (frontend / 'specieslist.csv').write_text(
                'code\tdisplay\tscientific_name\taudio_files\tspec_files\n'
                'DENEBR\t1\tDendropsophus ebraccatus\tmissing.mp3\tmissing.webp\n',
                encoding='utf8',
            )
            settings = Settings(local_dir=root / 'store')
            output = settings.local_dir / 'references'
            with self.assertRaises(FileNotFoundError):
                import_references(frontend, output, settings.species_file)
            self.assertFalse(output.exists())
            self.assertEqual(list(settings.local_dir.glob('.references-*')), [])

    def test_import_playback_missing_assets_and_candidate_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frontend = root / 'frontend'
            audio = frontend / 'docs/static/Dendropsophus ebraccatus/audio'
            audio.mkdir(parents=True)
            (audio / 'call.mp3').write_bytes(b'ID3test-audio')
            (frontend / 'specieslist.csv').write_text(
                'code\tdisplay\tscientific_name\taudio_files\tspec_files\n'
                'DENEBR\t1\tDendropsophus ebraccatus\tcall.mp3\t\n', encoding='utf8')
            settings = Settings(local_dir=root / 'store')
            result = import_references(frontend, settings.local_dir / 'references', settings.species_file)
            self.assertEqual(result, {'species': 1, 'clips': 1})
            with TestClient(create_app(settings)) as client:
                response = client.get('/api/species/DENEBR/references')
                self.assertEqual(response.status_code, 200)
                clip = response.json()['target']['clips'][0]
                self.assertIsNone(clip['startSeconds'])
                self.assertIsNone(clip['spectrogramUrl'])
                media = client.get(clip['audioUrl'], headers={'Range': 'bytes=0-2'})
                self.assertEqual(media.status_code, 206)
                self.assertEqual(media.content, b'ID3')
                self.assertEqual(media.headers['content-type'], 'audio/mpeg')
                self.assertEqual(client.get('/media/references/manifest.json').status_code, 404)
                self.assertEqual(client.get('/api/species/DENEBR/candidates').json()['clips'], [])
                (settings.local_dir / 'references/0000.mp3').unlink()
                self.assertEqual(client.get(clip['audioUrl']).status_code, 404)
            with self.assertRaises(ValueError):
                import_references(frontend, settings.local_dir / 'references', settings.species_file)
