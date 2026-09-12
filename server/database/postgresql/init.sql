CREATE EXTENSION IF NOT EXISTS vchord CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_tokenizer CASCADE;
CREATE EXTENSION IF NOT EXISTS vchord_bm25 CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE OR REPLACE FUNCTION max_sim(document vector[], query vector[]) RETURNS double precision AS $$
    WITH
        queries AS (
            SELECT row_number() OVER () AS query_number, * FROM (SELECT unnest(query) AS query)
        ),
        documents AS (
            SELECT unnest(document) AS document
        ),
        similarities AS (
            SELECT query_number, 1 - (document <=> query) AS similarity FROM queries CROSS JOIN documents
        ),
        max_similarities AS (
            SELECT MAX(similarity) AS max_similarity FROM similarities GROUP BY query_number
        )
    SELECT SUM(max_similarity) FROM max_similarities
$$ LANGUAGE SQL;
