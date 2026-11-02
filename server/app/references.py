"""Optional local reference fixtures, separate from review and annotation storage."""

import json
from pathlib import Path

from .repository import AppError


class ReferenceCatalog:
    def __init__(self, root):
        self.root = Path(root).resolve()
        manifest = self.root / 'manifest.json'
        # exists() can hide access-denied errors on Windows. A missing optional
        # catalog is valid; an unreadable catalog must fail visibly at startup.
        try:
            content = manifest.read_text(encoding='utf8')
        except FileNotFoundError:
            self.species = {}
        else:
            self.species = json.loads(content)
        self.assets = {}
        for clips in self.species.values():
            for clip in clips:
                for filename in clip.values():
                    if filename:
                        path = (self.root / filename).resolve()
                        if not path.is_relative_to(self.root):
                            raise ValueError('Reference asset escapes reference directory')
                        self.assets[filename] = path

    def clips(self, species_id, request):
        return [
            {key: str(request.url_for('reference_asset', asset_id=value)) if value else None
             for key, value in clip.items()}
            for clip in self.species.get(species_id, [])
        ]

    def asset_path(self, asset_id):
        # Only manifest-listed files are public, never arbitrary filesystem paths.
        path = self.assets.get(asset_id)
        if path is None or not path.is_file():
            raise AppError(404, 'Reference asset unavailable')
        return path
