"""Candidate batches with species-bound keyset pagination."""

import base64
import binascii
from datetime import datetime, timezone
import json

from google.cloud import bigquery

import media
import settings
from settings import DATASET_ID, MAX_CANDIDATES, PROJECT_ID


def encode_cursor(species_id, cutoff, row):
    # A continuation token, not an authorization credential. No private data.
    data = {'v': 1, 'species_id': species_id, 'cutoff': cutoff.isoformat(),
            'created_at': row.created_at.isoformat(), 'candidate_id': row.candidate_id}
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip('=')


def decode_cursor(token, species_id):
    try:
        if not token or len(token) > 2048:
            raise ValueError()
        raw = base64.b64decode(token + '=' * (-len(token) % 4), altchars=b'-_', validate=True)
        data = json.loads(raw)
        if (not isinstance(data, dict) or type(data.get('v')) is not int
                or data['v'] != 1 or data['species_id'] != species_id):
            raise ValueError()
        cutoff = datetime.fromisoformat(data['cutoff'])
        created_at = datetime.fromisoformat(data['created_at'])
        if cutoff.utcoffset() is None or created_at.utcoffset() is None:
            raise ValueError()
        # BigQuery TIMESTAMP uses UTC. Conversion also rejects offset timestamps
        # whose UTC value would lie outside Python/BigQuery's supported years.
        cutoff = cutoff.astimezone(timezone.utc)
        created_at = created_at.astimezone(timezone.utc)
        candidate_id = data['candidate_id']
        if (created_at > cutoff or cutoff > datetime.now(timezone.utc)
                or not isinstance(candidate_id, str) or not 1 <= len(candidate_id) <= 256
                or not candidate_id.isprintable() or candidate_id != candidate_id.strip()):
            raise ValueError()
        return cutoff, created_at, candidate_id
    except (ValueError, KeyError, TypeError, AttributeError, binascii.Error, UnicodeError,
            RecursionError, OverflowError):
        raise ValueError('Invalid cursor for this species') from None


def parse_candidate_request(args):
    for key in ('species_id', 'count', 'cursor'):
        if len(args.getlist(key)) > 1:
            raise ValueError(f'Supply {key} only once')
    species_id = args.get('species_id', '').strip()
    if not 1 <= len(species_id) <= 128 or not species_id.isprintable():
        raise ValueError('Provide one nonempty species_id (maximum 128 characters)')
    count_text = args.get('count', '15')
    if (not count_text.isascii() or not count_text.isdecimal()
            or len(count_text) > 2 or not 1 <= int(count_text) <= MAX_CANDIDATES):
        raise ValueError('count must be an integer between 1 and 20')
    if 'cursor' in args:
        cutoff, after_time, after_id = decode_cursor(args['cursor'], species_id)
    else:
        cutoff, after_time, after_id = datetime.now(timezone.utc), None, None
    return species_id, int(count_text), cutoff, after_time, after_id


def _read_candidate_rows(species_id, count, cutoff, after_time, after_id):
    query = f'''
        SELECT c.candidate_id, c.window_id, c.query_species_id, c.created_at, c.score,
               w.start_seconds, w.end_seconds, w.spectrogram_gcs_uri,
               r.recording_id, r.audio_gcs_uri
        FROM `{PROJECT_ID}.{DATASET_ID}.candidates` c
        JOIN `{PROJECT_ID}.{DATASET_ID}.windows` w USING (window_id)
        JOIN `{PROJECT_ID}.{DATASET_ID}.recordings` r USING (recording_id)
        WHERE c.query_species_id = @species_id AND c.created_at <= @cutoff
          AND (@after_time IS NULL OR c.created_at > @after_time
               OR (c.created_at = @after_time AND c.candidate_id > @after_id))
        ORDER BY c.created_at, c.candidate_id
        LIMIT @fetch_count
    '''
    return settings.read_rows(query, [
        bigquery.ScalarQueryParameter('species_id', 'STRING', species_id),
        bigquery.ScalarQueryParameter('cutoff', 'TIMESTAMP', cutoff),
        bigquery.ScalarQueryParameter('after_time', 'TIMESTAMP', after_time),
        bigquery.ScalarQueryParameter('after_id', 'STRING', after_id),
        # Fetch one extra row to detect whether another page exists.
        bigquery.ScalarQueryParameter('fetch_count', 'INT64', count + 1),
    ])


def _candidate_payload(row, urls):
    return {
        'candidate_id': row.candidate_id, 'window_id': row.window_id,
        'query_species_id': row.query_species_id, 'score': row.score,
        'created_at': row.created_at.isoformat(), 'recording_id': row.recording_id,
        'audio_url': urls.url(row.audio_gcs_uri),
        'start_seconds': row.start_seconds, 'end_seconds': row.end_seconds,
        'spectrogram_url': urls.url(row.spectrogram_gcs_uri),
    }


def get_candidates(species_id, count, cutoff, after_time, after_id):
    rows = _read_candidate_rows(species_id, count, cutoff, after_time, after_id)
    page = rows[:count]
    next_cursor = encode_cursor(species_id, cutoff, page[-1]) if len(rows) > count else None
    urls = media.MediaUrls()
    return {
        'candidates': [_candidate_payload(row, urls) for row in page],
        'next_cursor': next_cursor,
        'urls_expire_at': urls.expires_at.isoformat(),
    }
