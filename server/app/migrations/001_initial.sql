CREATE TABLE candidate_sets (
    id TEXT PRIMARY KEY,
    species TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    manifest TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE recordings (
    id TEXT PRIMARY KEY,
    metadata TEXT NOT NULL
);

CREATE TABLE windows (
    id TEXT PRIMARY KEY,
    recording_id TEXT NOT NULL REFERENCES recordings(id),
    metadata TEXT NOT NULL
);

CREATE TABLE clips (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    sample_rate INTEGER NOT NULL,
    frames INTEGER NOT NULL
);

CREATE TABLE candidates (
    set_id TEXT NOT NULL REFERENCES candidate_sets(id),
    window_id TEXT NOT NULL REFERENCES windows(id),
    rank INTEGER NOT NULL,
    clip_id TEXT NOT NULL REFERENCES clips(id),
    PRIMARY KEY(set_id,window_id),
    UNIQUE(set_id,rank)
);

CREATE TABLE batches (
    id TEXT PRIMARY KEY,
    species TEXT NOT NULL,
    annotator TEXT NOT NULL,
    set_id TEXT NOT NULL REFERENCES candidate_sets(id),
    created_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE(id,species,annotator)
);

CREATE TABLE batch_members (
    batch_id TEXT NOT NULL REFERENCES batches(id),
    window_id TEXT NOT NULL REFERENCES windows(id),
    rank INTEGER NOT NULL,
    clip_id TEXT NOT NULL REFERENCES clips(id),
    PRIMARY KEY(batch_id,window_id)
);

CREATE TABLE submissions (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES batches(id),
    annotator TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    payload TEXT NOT NULL,
    acknowledgement TEXT NOT NULL,
    server_time TEXT NOT NULL
);

CREATE TABLE annotation_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id TEXT NOT NULL REFERENCES submissions(id),
    window_id TEXT NOT NULL REFERENCES windows(id),
    species TEXT NOT NULL,
    annotator TEXT NOT NULL,
    label TEXT NOT NULL CHECK(label IN ('yes','no','unsure')),
    revision INTEGER NOT NULL,
    server_time TEXT NOT NULL,
    UNIQUE(window_id,species,annotator,revision)
);

CREATE TABLE effective_annotations (
    window_id TEXT NOT NULL REFERENCES windows(id),
    species TEXT NOT NULL,
    annotator TEXT NOT NULL,
    event_id INTEGER NOT NULL REFERENCES annotation_events(id),
    label TEXT NOT NULL,
    revision INTEGER NOT NULL,
    PRIMARY KEY(window_id,species,annotator)
);

CREATE INDEX batch_lookup ON batches(species,annotator,created_at);

CREATE INDEX set_species ON candidate_sets(species);

PRAGMA user_version=1;
