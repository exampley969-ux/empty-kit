-- Optional: the service creates all of this itself on start (IF NOT EXISTS).
-- Run in Supabase > SQL Editor only if you prefer to create it by hand.

CREATE TABLE IF NOT EXISTS cameras (
    camera_id    TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    source_uri   TEXT NOT NULL,
    frame_width  INTEGER,
    frame_height INTEGER,
    orientation  TEXT NOT NULL DEFAULT 'landscape'
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    camera_id     TEXT REFERENCES cameras(camera_id) ON DELETE SET NULL,
    camera_uri    TEXT,
    state         TEXT NOT NULL,
    started_at    DOUBLE PRECISION,
    ended_at      DOUBLE PRECISION,
    frame_count   INTEGER NOT NULL DEFAULT 0,
    error_message TEXT
);
CREATE TABLE IF NOT EXISTS entities (
    entity_id   TEXT PRIMARY KEY,
    session_id  TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
    entity_type TEXT NOT NULL,
    first_seen  DOUBLE PRECISION,
    last_seen   DOUBLE PRECISION,
    last_state  TEXT,
    last_x      DOUBLE PRECISION,
    last_y      DOUBLE PRECISION,
    confidence  DOUBLE PRECISION
);
CREATE TABLE IF NOT EXISTS zones (
    zone_id      TEXT PRIMARY KEY,
    camera_id    TEXT REFERENCES cameras(camera_id) ON DELETE CASCADE,
    name         TEXT NOT NULL,
    polygon_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    event_id      TEXT PRIMARY KEY,
    session_id    TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
    event_type    TEXT NOT NULL,
    entity_type   TEXT,
    zone_id       TEXT,
    start_time    DOUBLE PRECISION NOT NULL,
    end_time      DOUBLE PRECISION,
    confidence    DOUBLE PRECISION NOT NULL,
    state         TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    created_at    DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_events_start  ON events(start_time);
CREATE INDEX IF NOT EXISTS idx_events_type   ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_zone   ON events(zone_id);
CREATE INDEX IF NOT EXISTS idx_events_state  ON events(state);
CREATE TABLE IF NOT EXISTS event_entities (
    event_id  TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL,
    PRIMARY KEY (event_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_event_entities_entity ON event_entities(entity_id);
CREATE TABLE IF NOT EXISTS rules (
    rule_id         TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    definition_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id   TEXT PRIMARY KEY,
    rule_id    TEXT NOT NULL,
    rule_name  TEXT,
    event_id   TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    message    TEXT NOT NULL,
    confidence DOUBLE PRECISION,
    created_at DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at);
CREATE TABLE IF NOT EXISTS evidence (
    event_id              TEXT PRIMARY KEY REFERENCES events(event_id) ON DELETE CASCADE,
    timestamp             DOUBLE PRECISION NOT NULL,
    confidence            DOUBLE PRECISION,
    entity_location_json  TEXT,
    zone_geometry_json    TEXT,
    frame_snapshot_path   TEXT
);
CREATE TABLE IF NOT EXISTS system_state (
    key        TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at DOUBLE PRECISION
);
CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL
);
ALTER TABLE cameras ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE zones ENABLE ROW LEVEL SECURITY;
ALTER TABLE events ENABLE ROW LEVEL SECURITY;
ALTER TABLE event_entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE evidence ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
