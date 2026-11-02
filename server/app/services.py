"""Application validation and source recording access."""

import json
from pathlib import Path, PurePosixPath
import wave

from .models import CandidateBundle, Species
from .repository import AppError, Repository


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

    def recording_path(self, recording):
        """Resolve a registered relative filename within its configured audio root."""
        root = self.settings.audio_roots.get(recording['dataset_name'])
        if root is None:
            raise AppError(422, 'No audio root configured for this recording project')
        root = Path(root).resolve()
        relative = PurePosixPath(recording['file_path'])
        if (relative.is_absolute() or '..' in relative.parts
                or ':' in recording['file_path'] or '\\' in recording['file_path']):
            raise AppError(422, 'Recording path must be a safe relative POSIX path')
        path = (root / str(relative)).resolve()
        if not path.is_relative_to(root):
            raise AppError(422, 'Recording path escapes its configured audio root')
        if not path.is_file():
            raise AppError(404, 'Recording file unavailable')
        return path

    def get_recording_audio(self, recording_id):
        return self.recording_path(self.repository.get_recording(recording_id))

    def import_bundle(self, manifest_path):
        """Validate source recording availability and bounds, then import metadata."""
        bundle = CandidateBundle.model_validate_json(
            Path(manifest_path).read_text(encoding='utf8')
        )
        self.require_species(bundle.speciesId)
        durations = {}
        for candidate in bundle.candidates:
            key = (candidate.datasetName, candidate.uri)
            if key not in durations:
                path = self.recording_path({
                    'dataset_name': candidate.datasetName, 'file_path': candidate.uri,
                })
                try:
                    with wave.open(str(path), 'rb') as wav:
                        rate, frames = wav.getframerate(), wav.getnframes()
                        if rate <= 0 or frames <= 0:
                            raise ValueError('Empty WAV')
                        durations[key] = (frames / rate, 1 / rate)
                except (wave.Error, EOFError, ValueError) as exc:
                    raise AppError(422, f'Invalid source recording: {exc}') from exc
            duration, tolerance = durations[key]
            if candidate.offsets[1] > duration + tolerance:
                raise AppError(422, 'Window extends beyond the source recording')

        imported = self.repository.import_candidate_set(bundle)
        return {
            'candidateSetId': bundle.candidateSetId,
            'imported': imported,
            'candidateCount': len(bundle.candidates),
        }
