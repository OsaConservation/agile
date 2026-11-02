"""Run from repository root: python -m ingestion_app --help."""

import argparse
from pathlib import Path

from .pipeline import PipelineRequest, run_pipeline


def main():
    parser = argparse.ArgumentParser(description='Embed a WAV folder and export boundary candidates.')
    parser.add_argument('audio_dir', type=Path)
    parser.add_argument('--classifier', type=Path, default=Path('content/osa_db/agile_classifier.pt'))
    parser.add_argument('--output', type=Path, required=True, help='New run directory outside the audio folder')
    parser.add_argument('--count', type=int, default=15)
    args = parser.parse_args()
    run_pipeline(PipelineRequest(args.audio_dir, args.classifier, args.output, args.count))


if __name__ == '__main__':
    main()
