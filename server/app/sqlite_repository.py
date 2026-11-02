"""SQLite adapter. Short-lived connections; transactions serialize imports and saves."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import time

from .repository import AppError
from .serialization import digest


def now():
    return datetime.now(timezone.utc).isoformat()


class SQLiteRepository:
    def __init__(self, local_dir: Path):
        self.local_dir = Path(local_dir).resolve()
        self.database = self.local_dir / 'annotations.sqlite'

    @contextmanager
    def connection(self, write=False):
        """Commit or roll back one operation, then always release its connection."""
        conn = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')

        try:
            # Reserve the writer before checking retries or source identities.
            conn.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self):
        self.local_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.database, timeout=30, isolation_level=None)

        try:
            # Changing journal mode can return BUSY immediately during parallel startup.
            for attempt in range(10):
                try:
                    conn.execute('PRAGMA journal_mode=WAL')
                    break
                except sqlite3.OperationalError as exc:
                    if exc.sqlite_errorcode != sqlite3.SQLITE_BUSY or attempt == 9:
                        raise
                    time.sleep(0.05 * (attempt + 1))
            script = (Path(__file__).parent / 'migrations/001_initial.sql').read_text()

            # Lock before checking the version so concurrent startups cannot both
            # apply the migration. Avoid executescript, which commits implicitly.
            conn.execute('BEGIN EXCLUSIVE')
            version = conn.execute('PRAGMA user_version').fetchone()[0]

            if version == 0:
                statement = ''
                for line in script.splitlines(keepends=True):
                    statement += line
                    if sqlite3.complete_statement(statement):
                        conn.execute(statement)
                        statement = ''
                if statement.strip():
                    raise RuntimeError('Incomplete schema statement')
            elif version not in (5, 6):
                raise RuntimeError(
                    f'Unsupported database schema version {version}. '
                    'The four-table schema requires a fresh store; back up the old store first.'
                )
            if version in (0, 5):
                # Add derived media metadata without rebuilding source/answer rows.
                migration = (Path(__file__).parent / 'migrations/002_spectrograms.sql').read_text()
                for statement in migration.split(';'):
                    if statement.strip():
                        conn.execute(statement)
            conn.commit()
        finally:
            conn.close()

    def health(self):
        with self.connection() as conn:
            conn.execute('SELECT 1 FROM candidates LIMIT 1').fetchall()

    def import_candidate_set(self, bundle):
        """Import recording/window identities and add previously unseen candidates."""
        inserted = 0
        with self.connection(True) as conn:
            for candidate in bundle.candidates:
                window_id = f'{bundle.sourceDatasetId}:window:{candidate.sourceWindowId}'
                recording_id = f'{bundle.sourceDatasetId}:recording:{candidate.recordingId}'
                recording = (
                    recording_id, candidate.recordingId, candidate.uri, candidate.datasetName,
                )
                self._insert_source(conn, 'recordings', recording)
                # Derived images are not part of immutable source identity.
                values = (window_id, recording_id, candidate.sourceWindowId, *candidate.offsets)
                previous = conn.execute(
                    'SELECT id,recording_id,source_window_id,start_seconds,end_seconds FROM windows WHERE id=?',
                    (window_id,),
                ).fetchone()
                if previous and tuple(previous) != values:
                    raise AppError(409, f'Conflicting source identity: {window_id}')
                if not previous:
                    conn.execute(
                        '''INSERT INTO windows(
                            id,recording_id,source_window_id,start_seconds,end_seconds
                        ) VALUES(?,?,?,?,?)''',
                        values,
                    )
                cursor = conn.execute(
                    'INSERT INTO candidates VALUES(?,?,?) ON CONFLICT(window_id,species) DO NOTHING',
                    (digest([window_id, bundle.speciesId]), window_id, bundle.speciesId),
                )
                inserted += cursor.rowcount
        return inserted > 0

    @staticmethod
    def _insert_source(conn, table, values):
        # Table names are internal constants, never supplied by the client.
        previous = conn.execute(f'SELECT * FROM {table} WHERE id=?', (values[0],)).fetchone()
        if previous:
            if tuple(previous) != values:
                raise AppError(409, f'Conflicting source identity: {values[0]}')
            return
        placeholders = ','.join('?' for _ in values)
        conn.execute(f'INSERT INTO {table} VALUES({placeholders})', values)

    @staticmethod
    def _batch_id(species, annotator):
        # A query context, not a persisted assignment or authentication token.
        return digest([species, annotator])

    @staticmethod
    def _metadata(row):
        return dict(
            recordingId=row['source_recording_id'],
            offsets=[row['start_seconds'], row['end_seconds']],
            uri=row['file_path'], datasetName=row['dataset_name'],
        )

    def get_review_batch(self, species, annotator, size):
        """Return unreviewed windows for this species; refresh advances the list."""
        with self.connection() as conn:
            rows = conn.execute(
                '''
                SELECT w.*,r.source_recording_id,r.file_path,r.dataset_name,
                    (SELECT count(*) FROM annotations a
                     WHERE a.window_id=w.id AND a.species=c.species) AS num_annotations
                FROM candidates c
                JOIN windows w ON w.id=c.window_id
                JOIN recordings r ON r.id=w.recording_id
                WHERE c.species=? AND NOT EXISTS (
                    SELECT 1 FROM annotations a
                    WHERE a.window_id=w.id AND a.species=c.species AND a.annotator=?
                )
                ORDER BY c.rowid LIMIT ?
                ''',
                (species, annotator, size),
            ).fetchall()
            if not rows:
                return {'batchId': None, 'clips': []}
            chosen = rows[:size]
            return {
                'batchId': self._batch_id(species, annotator),
                'clips': [dict(id=row['id'], **self._metadata(row), mediaRecordingId=row['recording_id'],
                               spectrogramPath=row['spectrogram_path'], numAnnotations=row['num_annotations'])
                          for row in chosen],
            }

    def save_submission(self, submission, annotator):
        """Save individual answers atomically; repeated requests create no new rows."""
        with self.connection(True) as conn:
            previous = conn.execute(
                '''
                SELECT a.*,w.start_seconds,w.end_seconds,
                       r.source_recording_id,r.file_path,r.dataset_name
                FROM annotations a JOIN windows w ON w.id=a.window_id
                JOIN recordings r ON r.id=w.recording_id
                WHERE a.submission_id=? ORDER BY a.window_id
                ''',
                (submission.submissionId,),
            ).fetchall()
            valid_context = submission.batchId == self._batch_id(submission.speciesId, annotator)
            if previous:
                # Compare actual saved answers, avoiding stored hashes or duplicate counts.
                saved = [dict(windowId=row['window_id'], label=row['label'],
                              **self._metadata(row)) for row in previous]
                requested = sorted(
                    [answer.model_dump(mode='json') for answer in submission.annotations],
                    key=lambda answer: answer['windowId'],
                )
                same_owner_and_species = all(
                    row['annotator'] == annotator and row['species'] == submission.speciesId
                    for row in previous
                )
                if not valid_context or not same_owner_and_species or saved != requested:
                    raise AppError(409, 'Submission ID already has different answers or owner')
                return {'submissionId': submission.submissionId, 'savedCount': len(previous)}

            if not valid_context:
                raise AppError(404, 'Unknown review context for this species and annotator')

            for answer in submission.annotations:
                row = conn.execute(
                    '''
                    SELECT w.*,r.source_recording_id,r.file_path,r.dataset_name
                    FROM windows w JOIN recordings r ON r.id=w.recording_id
                    JOIN candidates c ON c.window_id=w.id
                    WHERE w.id=? AND c.species=?
                    ''',
                    (answer.windowId, submission.speciesId),
                ).fetchone()
                metadata = answer.model_dump(
                    include={'recordingId', 'offsets', 'uri', 'datasetName'}, mode='json'
                )
                if not row or self._metadata(row) != metadata:
                    raise AppError(422, 'Annotation is not a candidate for this species or metadata differs')

            created_at = now()
            for answer in submission.annotations:
                conn.execute(
                    'INSERT INTO annotations VALUES(?,?,?,?,?,?,?)',
                    (
                        digest([submission.submissionId, answer.windowId]), answer.windowId,
                        submission.speciesId, annotator, answer.label,
                        submission.submissionId, created_at,
                    ),
                )
            return {'submissionId': submission.submissionId,
                    'savedCount': len(submission.annotations)}

    def get_submission(self, submission_id, annotator):
        with self.connection() as conn:
            row = conn.execute(
                '''
                SELECT submission_id,count(*) AS saved_count FROM annotations
                WHERE submission_id=? AND annotator=? GROUP BY submission_id
                ''',
                (submission_id, annotator),
            ).fetchone()
            if not row:
                raise AppError(404, 'Unknown submission')
            return {'submissionId': row['submission_id'], 'savedCount': row['saved_count']}

    def get_recording(self, recording_id):
        """Return only registered recording metadata; paths are resolved by the service."""
        with self.connection() as conn:
            row = conn.execute('SELECT * FROM recordings WHERE id=?', (recording_id,)).fetchone()
        if not row:
            raise AppError(404, 'Unknown recording')
        return dict(row)

    def get_window(self, window_id):
        with self.connection() as conn:
            row = conn.execute('SELECT * FROM windows WHERE id=?', (window_id,)).fetchone()
        if not row:
            raise AppError(404, 'Unknown window')
        return dict(row)

    def list_windows(self):
        with self.connection() as conn:
            return [dict(row) for row in conn.execute('SELECT * FROM windows ORDER BY id')]

    def set_spectrogram(self, window_id, path):
        with self.connection(True) as conn:
            cursor = conn.execute(
                'UPDATE windows SET spectrogram_path=? WHERE id=?', (path, window_id)
            )
            if cursor.rowcount != 1:
                raise AppError(404, 'Unknown window')
