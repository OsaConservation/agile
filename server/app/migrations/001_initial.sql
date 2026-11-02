-- Original audio files. Source IDs remain scoped to their dataset.
CREATE TABLE recordings (
    id TEXT PRIMARY KEY,
    source_recording_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    UNIQUE(dataset_name, source_recording_id)
);

-- An interval within the original recording; no separate audio asset.
CREATE TABLE windows (
    id TEXT PRIMARY KEY,
    recording_id TEXT NOT NULL REFERENCES recordings(id),
    source_window_id INTEGER NOT NULL,
    start_seconds REAL NOT NULL CHECK(start_seconds >= 0),
    end_seconds REAL NOT NULL CHECK(end_seconds > start_seconds),
    UNIQUE(recording_id, start_seconds, end_seconds)
);

-- One review candidate per window and target species.
CREATE TABLE candidates (
    id TEXT PRIMARY KEY,
    window_id TEXT NOT NULL REFERENCES windows(id),
    species TEXT NOT NULL,
    UNIQUE(window_id, species)
);

-- Each accepted answer is a row. Corrections append another answer.
-- A submission ID groups answers and identifies retries; created_at is server UTC.
CREATE TABLE annotations (
    id TEXT PRIMARY KEY,
    window_id TEXT NOT NULL REFERENCES windows(id),
    species TEXT NOT NULL,
    annotator TEXT NOT NULL,
    label TEXT NOT NULL CHECK(label IN ('yes','no','unsure')),
    submission_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(submission_id, window_id)
);

-- Speed up species queues, per-window annotation lookups, and save retries.
CREATE INDEX candidate_queue ON candidates(species);
CREATE INDEX annotation_latest ON annotations(window_id, species, annotator);
CREATE INDEX annotation_submission ON annotations(submission_id);
PRAGMA user_version=5;
