"""Copy reference audio/spectrogram fixtures from an OsaAnurans checkout."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import uuid

from server.app.settings import Settings


def import_references(frontend_root, output, species_file):
    frontend_root, output = Path(frontend_root), Path(output)
    if output.exists():
        raise ValueError('Output already exists; choose a new reference directory')
    known = {s['id']: s['scientificName'] for s in json.loads(Path(species_file).read_text(encoding='utf8'))}
    static = (frontend_root / 'docs/static').resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # A regular sibling directory inherits the server media permissions. Windows
    # TemporaryDirectory ACLs can otherwise make the published catalog unreadable.
    staging = output.parent / f'.references-{uuid.uuid4().hex}'
    staging.mkdir()
    try:
        manifest = {}
        asset_count = 0

        def copy_asset(source):
            nonlocal asset_count
            source = source.resolve()
            if not source.is_relative_to(static):
                raise ValueError('Reference path escapes frontend static directory')
            if not source.is_file():
                raise FileNotFoundError(source)
            name = f'{asset_count:04d}{source.suffix.lower()}'
            shutil.copyfile(source, staging / name)
            asset_count += 1
            return name

        with (frontend_root / 'specieslist.csv').open(encoding='utf-8-sig', newline='') as source:
            for row in csv.DictReader(source, delimiter='\t'):
                code = row['code']
                if row['display'] != '1' or code not in known:
                    continue
                if known[code] != row['scientific_name']:
                    raise ValueError(f'Species mapping mismatch: {code}')
                audio = next(csv.reader([row['audio_files']]))
                specs = next(csv.reader([row['spec_files']]))
                root = static / row['scientific_name']
                manifest[code] = [
                    {'audioUrl': copy_asset(root / 'audio' / filename),
                     'spectrogramUrl': copy_asset(root / 'img/spec' / specs[i])
                     if i < len(specs) and specs[i] else None}
                    for i, filename in enumerate(audio) if filename
                ]
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
        staging.rename(output)
    finally:
        # The path is a newly allocated sibling; never remove existing output.
        if staging.exists():
            shutil.rmtree(staging)
    return {'species': len(manifest), 'clips': sum(map(len, manifest.values()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frontend_root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    settings = Settings()
    result = import_references(
        args.frontend_root,
        args.output or settings.local_dir / 'references',
        settings.species_file,
    )
    print(json.dumps(result))


if __name__ == '__main__':
    main()
