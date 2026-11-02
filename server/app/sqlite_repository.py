"""SQLite adapter. Short-lived connections; transactions serialize assignment/saves."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid

from .repository import AppError
from .serialization import canonical, digest


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
            # Reserve the writer before reading state used to assign IDs/revisions.
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
            conn.execute('PRAGMA journal_mode=WAL')
            script = (Path(__file__).parent / 'migrations/001_initial.sql').read_text()

            # Lock before checking the version so concurrent startups cannot both
            # apply the migration. Avoid executescript, which commits implicitly.
            conn.execute('BEGIN EXCLUSIVE')
            version = conn.execute('PRAGMA user_version').fetchone()[0]

            if version == 0:
                for statement in script.split(';'):
                    if statement.strip():
                        conn.execute(statement)
            elif version != 1:
                raise RuntimeError(f'Unsupported database schema version {version}')
            conn.commit()
        finally:
            conn.close()

    def health(self):
        with self.connection() as conn:
            conn.execute('SELECT 1 FROM candidate_sets LIMIT 1').fetchall()

    def import_candidate_set(self, bundle, manifest_hash, clips):
        """Publish one complete set, preserving immutable source and model IDs."""
        with self.connection(True) as conn:
            existing = conn.execute(
                'SELECT manifest_hash FROM candidate_sets WHERE id=?',
                (bundle.candidateSetId,),
            ).fetchone()
            if existing:
                if existing[0] != manifest_hash:
                    raise AppError(409, 'Candidate set ID already has different content')
                return False

            # A run ID identifies the model provenance across selection sets.
            for row in conn.execute('SELECT manifest FROM candidate_sets'):
                previous_run = json.loads(row[0])['classifierRun']
                if (
                    previous_run['id'] == bundle.classifierRun.id
                    and previous_run != bundle.classifierRun.model_dump(mode='json')
                ):
                    raise AppError(409, 'Classifier run ID already has different provenance')

            conn.execute(
                'INSERT INTO candidate_sets VALUES(?,?,?,?,?)',
                (
                    bundle.candidateSetId,
                    bundle.speciesId,
                    manifest_hash,
                    canonical(bundle.model_dump(mode='json')),
                    now(),
                ),
            )

            for candidate in bundle.candidates:
                window_id = f'{bundle.sourceDatasetId}:window:{candidate.sourceWindowId}'
                recording_id = f'{bundle.sourceDatasetId}:recording:{candidate.recordingId}'
                window_metadata = candidate.model_dump(
                    include={'recordingId', 'offsets', 'uri', 'datasetName'},
                    mode='json',
                )
                recording = canonical(
                    {k: v for k, v in window_metadata.items() if k != 'offsets'}
                    | {'deployment': candidate.deployment}
                )

                # Reusing a source ID is safe only when its metadata agrees.
                for table, identity, metadata in [
                    ('recordings', recording_id, recording),
                    ('windows', window_id, canonical(window_metadata)),
                ]:
                    existing = conn.execute(
                        f'SELECT metadata FROM {table} WHERE id=?', (identity,)
                    ).fetchone()
                    if existing and existing[0] != metadata:
                        raise AppError(409, f'Conflicting source identity: {identity}')
                    if not existing:
                        if table == 'windows':
                            conn.execute(
                                'INSERT INTO windows VALUES(?,?,?)',
                                (identity, recording_id, metadata),
                            )
                        else:
                            conn.execute(
                                'INSERT INTO recordings VALUES(?,?)',
                                (identity, metadata),
                            )

                clip = clips[candidate.clip]
                conn.execute(
                    'INSERT OR IGNORE INTO clips VALUES(?,?,?,?,?)',
                    (
                        clip['id'], clip['path'], clip['sha256'],
                        clip['sample_rate'], clip['frames'],
                    ),
                )
                conn.execute(
                    'INSERT INTO candidates VALUES(?,?,?,?)',
                    (bundle.candidateSetId, window_id, candidate.rank, clip['id']),
                )

            return True

    def get_review_batch(self, species, annotator, size):
        """Reuse the issued batch, or assign unreviewed windows from one set."""
        with self.connection(True) as conn:
            batch = conn.execute(
                '''
                SELECT * FROM batches
                WHERE species=? AND annotator=?
                ORDER BY rowid DESC LIMIT 1
                ''',
                (species, annotator),
            ).fetchone()

            # Completed batches remain stable until an explicit next operation
            # is introduced along with frontend paging.
            if not batch:
                rows = conn.execute(
                    '''
                    SELECT c.* FROM candidates c
                    JOIN candidate_sets s ON s.id=c.set_id
                    WHERE s.species=? AND NOT EXISTS (
                        SELECT 1 FROM effective_annotations e
                        WHERE e.window_id=c.window_id AND e.species=? AND e.annotator=?
                    )
                    ORDER BY s.rowid,c.rank
                    ''',
                    (species, species, annotator),
                ).fetchall()
                if not rows:
                    return {'batchId': None, 'clips': []}

                set_id = rows[0]['set_id']
                chosen = [row for row in rows if row['set_id'] == set_id][:size]
                batch_id = str(uuid.uuid4())

                conn.execute(
                    'INSERT INTO batches VALUES(?,?,?,?,?,NULL)',
                    (batch_id, species, annotator, set_id, now()),
                )
                conn.executemany(
                    'INSERT INTO batch_members VALUES(?,?,?,?)',
                    [(batch_id, row['window_id'], row['rank'], row['clip_id']) for row in chosen],
                )
            else:
                batch_id = batch['id']

            records = conn.execute(
                '''
                SELECT w.id,w.metadata,m.clip_id FROM batch_members m
                JOIN windows w ON w.id=m.window_id
                WHERE m.batch_id=? ORDER BY m.rank
                ''',
                (batch_id,),
            ).fetchall()
            return {
                'batchId': batch_id,
                'clips': [
                    dict(id=row['id'], **json.loads(row['metadata']), clipId=row['clip_id'])
                    for row in records
                ],
            }

    def save_submission(self, submission, annotator):
        """Validate, record history, and update current labels in one transaction."""
        payload = submission.model_dump(mode='json')
        # Answer order is irrelevant when recognizing a retry of the same payload.
        payload['annotations'].sort(key=lambda a: a['windowId'])
        payload_hash = digest(payload)

        with self.connection(True) as conn:
            # Resolve a retry before allocating any new events or revisions.
            existing = conn.execute(
                'SELECT * FROM submissions WHERE id=?', (submission.submissionId,)
            ).fetchone()
            if existing:
                if existing['annotator'] != annotator or existing['payload_hash'] != payload_hash:
                    raise AppError(409, 'Submission ID already has different content or owner')
                return json.loads(existing['acknowledgement'])

            batch = conn.execute(
                'SELECT * FROM batches WHERE id=? AND species=? AND annotator=?',
                (submission.batchId, submission.speciesId, annotator),
            ).fetchone()
            if not batch:
                raise AppError(404, 'Unknown batch for this species and annotator')

            members = {
                row['id']: json.loads(row['metadata'])
                for row in conn.execute(
                    '''
                    SELECT w.* FROM windows w
                    JOIN batch_members m ON m.window_id=w.id
                    WHERE m.batch_id=?
                    ''',
                    (submission.batchId,),
                )
            }
            # Check every answer against trusted metadata before writing any.
            for answer in submission.annotations:
                metadata = answer.model_dump(
                    include={'recordingId', 'offsets', 'uri', 'datasetName'},
                    mode='json',
                )
                if members.get(answer.windowId) != metadata:
                    raise AppError(422, 'Annotation is outside the batch or source metadata differs')

            timestamp = now()
            acknowledgement = {
                'submissionId': submission.submissionId,
                'savedCount': len(submission.annotations),
            }
            conn.execute(
                'INSERT INTO submissions VALUES(?,?,?,?,?,?,?)',
                (
                    submission.submissionId, submission.batchId, annotator,
                    payload_hash, canonical(payload), canonical(acknowledgement), timestamp,
                ),
            )

            for answer in submission.annotations:
                key = (answer.windowId, submission.speciesId, annotator)
                current = conn.execute(
                    '''
                    SELECT * FROM effective_annotations
                    WHERE window_id=? AND species=? AND annotator=?
                    ''',
                    key,
                ).fetchone()
                # A fresh submission ID need not create history for an unchanged answer.
                if current and current['label'] == answer.label:
                    continue

                revision = current['revision'] + 1 if current else 1
                event = conn.execute(
                    '''
                    INSERT INTO annotation_events (
                        submission_id,window_id,species,annotator,label,revision,server_time
                    ) VALUES(?,?,?,?,?,?,?)
                    ''',
                    (submission.submissionId, *key, answer.label, revision, timestamp),
                )
                conn.execute(
                    '''
                    INSERT INTO effective_annotations VALUES(?,?,?,?,?,?)
                    ON CONFLICT(window_id,species,annotator) DO UPDATE SET
                        event_id=excluded.event_id,
                        label=excluded.label,
                        revision=excluded.revision
                    ''',
                    (*key, event.lastrowid, answer.label, revision),
                )

            # Partial saves stay open; any effective label, including unsure,
            # counts as reviewed when deciding whether the batch is complete.
            remaining = conn.execute(
                '''
                SELECT count(*) FROM batch_members m
                WHERE batch_id=? AND NOT EXISTS (
                    SELECT 1 FROM effective_annotations e
                    WHERE e.window_id=m.window_id AND e.species=? AND e.annotator=?
                )
                ''',
                (submission.batchId, submission.speciesId, annotator),
            ).fetchone()[0]
            if not remaining:
                conn.execute(
                    'UPDATE batches SET completed_at=COALESCE(completed_at,?) WHERE id=?',
                    (timestamp, submission.batchId),
                )

            return acknowledgement

    def get_submission(self, submission_id, annotator):
        with self.connection() as conn:
            row = conn.execute(
                'SELECT acknowledgement FROM submissions WHERE id=? AND annotator=?',
                (submission_id, annotator),
            ).fetchone()
            if not row:
                raise AppError(404, 'Unknown submission')
            return json.loads(row[0])

    def get_clip(self, clip_id):
        with self.connection() as conn:
            row = conn.execute(
                'SELECT path FROM clips WHERE id=?', (clip_id,)
            ).fetchone()
        if not row:
            raise AppError(404, 'Unknown clip')

        # Resolve before checking containment, including any filesystem symlinks.
        path = (self.local_dir / row[0]).resolve()
        if not path.is_relative_to(self.local_dir / 'clips') or not path.is_file():
            raise AppError(404, 'Clip file unavailable')
        return path
