"""Run from repository root: python -m server.tools.import_candidates bundle/manifest.json."""

import argparse
import json
from pathlib import Path

from server.app.services import AnnotationService
from server.app.settings import Settings
from server.app.sqlite_repository import SQLiteRepository


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--local-dir', type=Path)
    parser.add_argument('--generate-spectrograms', action='store_true',
                        help='Generate missing window images after import (requires research environment)')
    args = parser.parse_args()

    settings = Settings(**({'local_dir': args.local_dir} if args.local_dir else {}))
    repository = SQLiteRepository(settings.local_dir)
    repository.initialize()
    print(json.dumps(
        AnnotationService(repository, settings).import_bundle(args.manifest),
        indent=2,
    ))
    if args.generate_spectrograms:
        from server.tools.generate_spectrograms import generate_spectrograms
        print(json.dumps(generate_spectrograms(settings, repository)))


if __name__ == '__main__':
    main()
