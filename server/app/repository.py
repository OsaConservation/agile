from typing import Protocol

from .models import CandidateBundle, Submission


class AppError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(message)


class Repository(Protocol):
    """Atomic imports and saves, individual annotation history, query-based reviews.

    A batch ID identifies a species/annotator review context, not stored membership.
    Labels belong to annotator/species/window independently of classifier runs.
    """

    def initialize(self) -> None: ...

    def health(self) -> None: ...

    def import_candidate_set(
        self, bundle: CandidateBundle
    ) -> bool: ...

    def get_review_batch(self, species: str, annotator: str, size: int) -> dict: ...

    def save_submission(self, submission: Submission, annotator: str) -> dict: ...

    def get_submission(self, submission_id: str, annotator: str) -> dict: ...

    def get_recording(self, recording_id: str) -> dict: ...
