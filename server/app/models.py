"""Versioned bundle and HTTP contracts. Source IDs are not server IDs."""

from datetime import datetime
from typing import Annotated, Literal
import math

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=250)]


class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class Species(Model):
    id: Identifier
    scientificName: str
    commonName: str


class WindowMetadata(Model):
    """Source fields echoed by the client and checked against stored metadata."""

    recordingId: Annotated[int, Field(strict=True, ge=0)]
    offsets: tuple[float, float]
    uri: Identifier
    datasetName: Identifier

    @model_validator(mode='after')
    def valid_offsets(self):
        if not 0 <= self.offsets[0] < self.offsets[1]:
            raise ValueError('Offsets must satisfy 0 <= start < end')
        return self


class Candidate(WindowMetadata):
    sourceWindowId: Annotated[int, Field(strict=True, ge=0)]
    logit: float
    boundaryDistance: Annotated[float, Field(ge=0)]
    rank: Annotated[int, Field(strict=True, ge=0)]
    deployment: dict | None = None


class ClassifierRun(Model):
    id: Identifier
    artifactSha256: Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')] | None
    embeddingModel: Annotated[str, Field(min_length=1, max_length=16000)] | dict
    trainingSnapshotId: Identifier | None


class CandidateBundle(Model):
    """Immutable boundary-selection output, including scoring provenance."""

    schemaVersion: Literal[2]
    candidateSetId: Identifier
    speciesId: Identifier
    sourceDatasetId: Annotated[str, Field(pattern=r'^[A-Za-z0-9_-]{1,100}$')]
    createdAt: datetime
    classifierRun: ClassifierRun
    targetScore: float
    selectionMode: Literal['boundary']
    candidates: Annotated[list[Candidate], Field(min_length=1)]

    @model_validator(mode='after')
    def consistent_candidates(self):
        require_utc(self.createdAt)

        ids = [candidate.sourceWindowId for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate source window IDs')
        # Resolve equal distances consistently so every import keeps the same order.
        ordered = sorted(
            self.candidates, key=lambda candidate: (candidate.boundaryDistance, candidate.sourceWindowId)
        )
        if (
            [candidate.sourceWindowId for candidate in ordered] != ids
            or [candidate.rank for candidate in ordered] != list(range(len(ids)))
        ):
            raise ValueError(
                'Candidates must have contiguous ranks ordered by distance, '
                'then source window ID'
            )

        recordings = {}
        for candidate in self.candidates:
            if not math.isclose(
                candidate.boundaryDistance,
                abs(candidate.logit - self.targetScore),
                rel_tol=1e-6,
                abs_tol=1e-7,
            ):
                raise ValueError('Boundary distance does not match logit')

            metadata = (candidate.uri, candidate.datasetName, candidate.deployment)
            if candidate.recordingId in recordings and recordings[candidate.recordingId] != metadata:
                raise ValueError('Conflicting source recording metadata')
            recordings[candidate.recordingId] = metadata
        return self


def require_utc(value):
    if value.tzinfo is None or value.utcoffset().total_seconds() != 0:
        raise ValueError('Timestamp must include UTC timezone')


class Answer(WindowMetadata):
    windowId: Identifier
    label: Literal['yes', 'no', 'unsure']


class Submission(Model):
    """One atomic save of individual annotations; submissionId identifies retries."""

    submissionId: Identifier
    speciesId: Identifier
    batchId: Identifier
    createdAt: datetime
    annotations: Annotated[list[Answer], Field(min_length=1, max_length=15)]

    @model_validator(mode='after')
    def valid_submission(self):
        require_utc(self.createdAt)
        if len({a.windowId for a in self.annotations}) != len(self.annotations):
            raise ValueError('Duplicate annotation window IDs')
        return self


class Acknowledgement(Model):
    submissionId: str
    savedCount: int


class AudioClip(Model):
    """Playable media shared by reference clips and clips awaiting review."""

    audioUrl: str
    spectrogramUrl: str | None = None
    # Null for standalone reference audio; full recordings specify playback bounds.
    startSeconds: float | None = None
    endSeconds: float | None = None


class CandidateClip(WindowMetadata, AudioClip):
    """A review clip with source metadata needed to validate an annotation."""

    id: str
    numAnnotations: Annotated[int, Field(ge=0)] = 0


class BatchResponse(Model):
    # Identifies a species/annotator review context, not a persisted assignment.
    batchId: str | None
    clips: list[CandidateClip]


class ReferenceSpecies(Species):
    clips: list[AudioClip] = Field(default_factory=list)


class References(Model):
    target: ReferenceSpecies
    comparisons: list[ReferenceSpecies]


class Recording(Model):
    """Original audio file and its source recording ID/project."""

    id: Identifier
    source_recording_id: int
    file_path: str
    dataset_name: str


class Window(Model):
    """Subsection of a recording; source_window_id links back to Hoplite embeddings."""

    id: Identifier
    recording_id: Identifier
    source_window_id: int
    start_seconds: Annotated[float, Field(ge=0)]
    end_seconds: float
    spectrogram_path: str | None = None

    @model_validator(mode='after')
    def valid_interval(self):
        if self.end_seconds <= self.start_seconds:
            raise ValueError('Window end must be after its start')
        return self


class Annotation(Model):
    """One answer with a server UTC timestamp; the latest inserted answer is current."""

    id: Identifier
    window_id: Identifier
    species: Identifier
    annotator: Identifier
    label: Literal['yes', 'no', 'unsure']
    submission_id: Identifier
    created_at: datetime


class CandidateRecord(Model):
    """A window awaiting review for a target species."""

    id: Identifier
    window_id: Identifier
    species: Identifier
