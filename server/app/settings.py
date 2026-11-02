"""Local configuration; defaults do not depend on the shell's working directory."""

from dataclasses import dataclass, field
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    local_dir: Path = field(
        default_factory=lambda: Path(os.environ.get('OSA_SERVER_LOCAL', ROOT / 'local'))
    )
    species_file: Path = ROOT / 'config' / 'species.json'
    annotator: str = field(
        default_factory=lambda: os.environ.get('OSA_ANNOTATOR', 'local-development')
    )
    batch_size: int = 15
    origins: tuple[str, ...] = ('http://127.0.0.1:8000', 'http://localhost:8000')

    def __post_init__(self):
        if not self.annotator.strip() or not 1 <= self.batch_size <= 15:
            raise ValueError('Annotator must be nonempty and batch size must be 1..15')
