-- Quick script to enable pgvector in Neon
-- Copy and paste this into Neon's SQL Editor

-- Enable the extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify it's working
SELECT 
    extname as "Extension Name",
    extversion as "Version"
FROM pg_extension 
WHERE extname = 'vector';

-- If you see a row with "vector" and a version number, you're good to go!

