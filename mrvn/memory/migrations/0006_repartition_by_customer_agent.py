# Destructive migration: repartition EmbeddingChunk by (customer_id, agent_id).
#
# Drops the old agent_id-only LIST-partitioned table and recreates it with a
# composite (customer_id, agent_id) partition key.  Existing rows are lost;
# operators must re-index after applying this migration.
#
# Sentinel partition: rows whose customer_id is the nil UUID
# (00000000-0000-0000-0000-000000000000) fall into the DEFAULT partition.

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("agents", "0007_project_context"),
        ("memory", "0005_session_project_context_customer"),
    ]

    operations = [
        # 1. Tell Django's schema state about the new customer_id field.
        #    The actual column is created by the RunSQL below (table recreated).
        migrations.AddField(
            model_name="embeddingchunk",
            name="customer_id",
            field=models.UUIDField(
                default="00000000-0000-0000-0000-000000000000",
                help_text="Customer ID for tenant isolation (denormalized)",
            ),
        ),
        # 2. Drop the old partitioned table (and all its partitions / indexes).
        migrations.RunSQL(
            sql="DROP TABLE IF EXISTS memory_embeddingchunk CASCADE;",
            reverse_sql="",  # Recreated by the next statement on reverse
        ),
        # 3. Recreate with composite (customer_id, agent_id) partition key.
        migrations.RunSQL(
            sql="""
            CREATE TABLE memory_embeddingchunk (
                id BIGSERIAL,
                customer_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000',
                agent_id BIGINT NOT NULL REFERENCES agents_agent(id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
                source VARCHAR(20) NOT NULL DEFAULT 'message',
                source_id BIGINT NOT NULL,
                text TEXT NOT NULL,
                start_line INTEGER NOT NULL DEFAULT 0,
                end_line INTEGER NOT NULL DEFAULT 0,
                embedding vector(384) NOT NULL,
                embedding_model VARCHAR(100) NOT NULL DEFAULT 'all-MiniLM-L6-v2',
                content_hash VARCHAR(64) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                PRIMARY KEY (id, customer_id, agent_id)
            ) PARTITION BY LIST (customer_id, agent_id);
            """,
            reverse_sql="DROP TABLE IF EXISTS memory_embeddingchunk CASCADE;",
        ),
        # 4. Default partition — catches sentinel UUID and any unpartitioned rows.
        migrations.RunSQL(
            sql="""
            CREATE TABLE memory_embeddingchunk_default
            PARTITION OF memory_embeddingchunk DEFAULT;
            """,
            reverse_sql="DROP TABLE IF EXISTS memory_embeddingchunk_default;",
        ),
        # 5. Recreate indexes.
        migrations.RunSQL(
            sql="""
            CREATE INDEX memory_embeddingchunk_customer_agent_idx
            ON memory_embeddingchunk (customer_id, agent_id);
            """,
            reverse_sql="DROP INDEX IF EXISTS memory_embeddingchunk_customer_agent_idx;",
        ),
        migrations.RunSQL(
            sql="""
            CREATE INDEX memory_embeddingchunk_content_hash_idx
            ON memory_embeddingchunk (content_hash);
            """,
            reverse_sql="DROP INDEX IF EXISTS memory_embeddingchunk_content_hash_idx;",
        ),
        migrations.RunSQL(
            sql="""
            CREATE INDEX memory_embe_agent_source_idx
            ON memory_embeddingchunk (agent_id, source, source_id);
            """,
            reverse_sql="DROP INDEX IF EXISTS memory_embe_agent_source_idx;",
        ),
        migrations.RunSQL(
            sql="""
            CREATE INDEX memory_embe_agent_hash_idx
            ON memory_embeddingchunk (agent_id, content_hash);
            """,
            reverse_sql="DROP INDEX IF EXISTS memory_embe_agent_hash_idx;",
        ),
        # 6. HNSW index (m=24, ef_construction=128) for fast ANN search.
        migrations.RunSQL(
            sql="""
            CREATE INDEX embedding_chunk_hnsw_idx
            ON memory_embeddingchunk
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 24, ef_construction = 128);
            """,
            reverse_sql="DROP INDEX IF EXISTS embedding_chunk_hnsw_idx;",
        ),
    ]
