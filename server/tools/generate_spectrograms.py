"""Generate exact-window spectrograms for imported candidates, once per window."""

import argparse
import hashlib
from pathlib import Path
import uuid

from server.app.services import AnnotationService
from server.app.repository import AppError
from server.app.settings import Settings
from server.app.sqlite_repository import SQLiteRepository


def generate_spectrograms(settings, repository):
    # Research dependencies stay outside the HTTP server.
    import matplotlib

    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    import numpy as np
    import soundfile as sf

    service = AnnotationService(repository, settings)
    root = settings.local_dir / 'spectrograms'
    root.mkdir(parents=True, exist_ok=True)
    generated = skipped = 0
    for window in repository.list_windows():
        # A shared window only needs one image, regardless of target species.
        if window['spectrogram_path']:
            try:
                service.get_window_spectrogram(window['id'])
                skipped += 1
                continue
            except AppError as exc:
                if exc.status != 404:
                    raise

        recording = repository.get_recording(window['recording_id'])
        source = service.recording_path(recording)
        with sf.SoundFile(source) as audio:
            start = round(window['start_seconds'] * audio.samplerate)
            end = round(window['end_seconds'] * audio.samplerate)
            audio.seek(start)
            samples = audio.read(end - start, dtype='float32', always_2d=True).mean(axis=1)
            rate = audio.samplerate
        if len(samples) != end - start or len(samples) < 2:
            raise ValueError(f'Incomplete window audio: {window["id"]}')

        figure = Figure(figsize=(10, 3), dpi=100)
        axis = figure.subplots()
        nfft = min(1024, len(samples))
        with np.errstate(divide='ignore'):
            axis.specgram(samples, NFFT=nfft, Fs=rate, noverlap=nfft // 2,
                          cmap='magma', vmin=-100, vmax=0)
        axis.set_xlabel('Time within window (seconds)')
        axis.set_ylabel('Frequency (Hz)')
        figure.tight_layout()

        # Publish the image before its SQL path, so readers never see a partial file.
        name = hashlib.sha256(window['id'].encode()).hexdigest() + '.webp'
        temporary = root / f'{uuid.uuid4().hex}.tmp'
        try:
            figure.savefig(temporary, format='webp')
            temporary.replace(root / name)
            repository.set_spectrogram(window['id'], name)
        finally:
            temporary.unlink(missing_ok=True)
            figure.clear()
        generated += 1
    return {'generated': generated, 'skipped': skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-dir', type=Path)
    args = parser.parse_args()
    settings = Settings(**({'local_dir': args.local_dir} if args.local_dir else {}))
    repository = SQLiteRepository(settings.local_dir)
    repository.initialize()
    print(generate_spectrograms(settings, repository))


if __name__ == '__main__':
    main()
