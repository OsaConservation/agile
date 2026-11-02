"""Application validation and candidate-bundle media publication."""

import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import os
import tempfile
import wave

from .models import CandidateBundle, Species
from .repository import AppError, Repository
from .serialization import digest


class AnnotationService:
    def __init__(self, repository: Repository, settings):
        self.repository = repository
        self.settings = settings
        species = [
            Species.model_validate(s)
            for s in json.loads(settings.species_file.read_text(encoding='utf8'))
        ]
        self.species = {s.id: s for s in species}
        if len(self.species) != len(species):
            raise ValueError('Duplicate species codes in configuration')

    def require_species(self, species_id):
        if species_id not in self.species:
            raise AppError(404, 'Unknown species')
        return self.species[species_id]

    def get_batch(self, species_id):
        self.require_species(species_id)
        return self.repository.get_review_batch(
            species_id, self.settings.annotator, self.settings.batch_size
        )

    def save(self, submission):
        self.require_species(submission.speciesId)
        return self.repository.save_submission(submission, self.settings.annotator)

    def import_bundle(self, manifest_path):
        """Validate a bundle, install complete clips, then publish its records."""
        manifest_path = Path(manifest_path).resolve()
        bundle = CandidateBundle.model_validate_json(
            manifest_path.read_text(encoding='utf8')
        )
        self.require_species(bundle.speciesId)
        root = manifest_path.parent
        media = {}

        # Validate all media before publishing either files or database records.
        for candidate in bundle.candidates:
            name = candidate.clip
            relative = PurePosixPath(name)
            if (
                '\\' in name
                or ':' in name
                or relative.is_absolute()
                or '..' in relative.parts
            ):
                raise AppError(422, 'Clip must be a safe relative POSIX path')
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise AppError(422, f'Missing or unsafe clip: {name}')

            content = path.read_bytes()
            sha = hashlib.sha256(content).hexdigest()

            # Validate the same bytes that will be hashed and installed, even if
            # someone changes the source file while the import is running.
            try:
                with wave.open(io.BytesIO(content), 'rb') as wav:
                    rate, frames = wav.getframerate(), wav.getnframes()
                    if (
                        wav.getcomptype() != 'NONE'
                        or wav.getnchannels() not in (1, 2)
                        or wav.getsampwidth() != 2
                    ):
                        raise ValueError('Expected mono/stereo PCM16 WAV')
                    if (
                        rate <= 0
                        or frames <= 0
                        or len(wav.readframes(frames)) != frames * wav.getnchannels() * 2
                    ):
                        raise ValueError('Truncated or empty WAV')
                    # Allow one sample of rounding when comparing clip duration.
                    if not math.isclose(
                        frames / rate,
                        candidate.offsets[1] - candidate.offsets[0],
                        abs_tol=1 / rate,
                        rel_tol=0,
                    ):
                        raise ValueError('WAV duration differs from window offsets')
            except (wave.Error, EOFError, ValueError) as exc:
                raise AppError(422, f'Invalid clip {name}: {exc}') from exc

            if name in media and media[name]['sha256'] != sha:
                raise AppError(409, 'Clip changed during import')
            media[name] = dict(
                id=sha,
                path=f'clips/{sha}.wav',
                sha256=sha,
                sample_rate=rate,
                frames=frames,
                content=content,
            )

        # Include media hashes so a manifest cannot be retried with altered audio.
        manifest_hash = digest({
            'manifest': bundle.model_dump(mode='json'),
            'clips': {k: v['sha256'] for k, v in media.items()},
        })
        local = self.settings.local_dir.resolve()
        clips_dir = local / 'clips'
        clips_dir.mkdir(parents=True, exist_ok=True)

        # Content-addressed files are complete before the SQL transaction publishes
        # them. A failed transaction can leave harmless, unregistered orphan files.
        with tempfile.TemporaryDirectory(dir=local, prefix='import-') as staging:
            for clip in media.values():
                destination = local / clip['path']
                if (
                    destination.exists()
                    and hashlib.sha256(destination.read_bytes()).hexdigest()
                    == clip['sha256']
                ):
                    continue

                staged = Path(staging) / (clip['id'] + '.wav')
                staged.write_bytes(clip.pop('content'))
                os.replace(staged, destination)

        imported = self.repository.import_candidate_set(bundle, manifest_hash, media)
        return {
            'candidateSetId': bundle.candidateSetId,
            'imported': imported,
            'candidateCount': len(bundle.candidates),
        }
