from pathlib import Path
from typing import Protocol

from .models import CandidateBundle, Submission


class AppError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(message)


class Repository(Protocol):
    """Each mutating operation is atomic; retries never duplicate events.

    Issued batch membership is immutable. Labels belong to annotator/species/window,
    independently of scoring sets. Adapters must enforce these guarantees themselves.
    """

    def initialize(self) -> None: ...

    def health(self) -> None: ...

    def import_candidate_set(
        self, bundle: CandidateBundle, manifest_hash: str, clips: dict
    ) -> bool: ...

    def get_review_batch(self, species: str, annotator: str, size: int) -> dict: ...

    def save_submission(self, submission: Submission, annotator: str) -> dict: ...

    def get_submission(self, submission_id: str, annotator: str) -> dict: ...

    def get_clip(self, clip_id: str) -> Path: ...
