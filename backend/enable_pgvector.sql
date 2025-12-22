-- Enable pgvector extension in Neon
-- Run this in Neon's SQL Editor

CREATE EXTENSION IF NOT EXISTS vector;

-- Verify it's enabled
SELECT * FROM pg_extension WHERE extname = 'vector';

