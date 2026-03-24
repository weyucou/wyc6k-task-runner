# Rename memory models for clarity and repartition SessionEmbeddingChunk by customer_id.
#
# Model renames:
#   Message             → SessionMessage
#   ConversationSummary → SessionSummary
#   EmbeddingChunk      → SessionEmbeddingChunk
#
# Partition change:
#   Old: LIST partitioned by agent_id
#   New: LIST partitioned by customer_id
#
# Design:
#   customer_id is the tenant partition key.
#   agent is accessed via session (session_id FK replaces agent_id FK).
#
# SessionEmbeddingChunk is dropped and recreated (destructive) because
# psqlextra does not support ALTER TABLE ... SET PARTITION KEY.
# customer_id is required (NOT NULL, no default) — must be supplied at write time.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("memory", "0005_session_project_context_customer"),
    ]

    operations = [
        # 1. Rename regular tables via Django RenameModel (generates ALTER TABLE ... RENAME).
        migrations.RenameModel(
            old_name="Message",
            new_name="SessionMessage",
        ),
        migrations.RenameModel(
            old_name="ConversationSummary",
            new_name="SessionSummary",
        ),
        migrations.AlterModelOptions(
            name="sessionsummary",
            options={"ordering": ["-created_datetime"], "verbose_name_plural": "Session Summaries"},
        ),
        # 2. Rename EmbeddingChunk → SessionEmbeddingChunk, repartition by customer_id,
        #    and replace agent_id FK with session_id FK.
        #    SeparateDatabaseAndState: Django state sees the final model shape;
        #    the database drops the old table and creates the new one via RunSQL.
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.DeleteModel("EmbeddingChunk"),
                migrations.CreateModel(
                    name="SessionEmbeddingChunk",
                    fields=[
                        ("id", models.BigAutoField(primary_key=True, serialize=False)),
                        (
                            "customer_id",
                            models.UUIDField(help_text="Customer ID for tenant isolation (partition key)"),
                        ),
                        (
                            "session",
                            models.ForeignKey(
                                on_delete=models.deletion.CASCADE,
                                related_name="embedding_chunks",
                                to="memory.session",
                            ),
                        ),
                        (
                            "source",
                            models.CharField(
                                choices=[("message", "Message"), ("summary", "Summary"), ("file", "File")],
                                default="message",
                                max_length=20,
                            ),
                        ),
                        ("source_id", models.BigIntegerField(help_text="ID of the source message, summary, or file")),
                        ("text", models.TextField()),
                        ("start_line", models.IntegerField(default=0)),
                        ("end_line", models.IntegerField(default=0)),
                        (
                            "embedding",
                            models.Field(),  # VectorField — handled by psqlextra/pgvector
                        ),
                        (
                            "embedding_model",
                            models.CharField(
                                default="all-MiniLM-L6-v2",
                                help_text="Model used to generate embedding",
                                max_length=100,
                            ),
                        ),
                        (
                            "content_hash",
                            models.CharField(
                                db_index=True,
                                help_text="SHA256 hash for deduplication",
                                max_length=64,
                            ),
                        ),
                        ("created_at", models.DateTimeField(auto_now_add=True)),
                        ("updated_at", models.DateTimeField(auto_now=True)),
                    ],
                ),
            ],
            database_operations=[
                # Drop the old agent_id-partitioned table and all its partitions/indexes.
                migrations.RunSQL(
                    sql="DROP TABLE IF EXISTS memory_embeddingchunk CASCADE;",
                    reverse_sql="",
                ),
                # Recreate as memory_sessionembeddingchunk partitioned by customer_id,
                # with session_id FK replacing agent_id.
                migrations.RunSQL(
                    sql="""
                    CREATE TABLE memory_sessionembeddingchunk (
                        id BIGSERIAL,
                        customer_id UUID NOT NULL,
                        session_id BIGINT NOT NULL REFERENCES memory_session(id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
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
                        PRIMARY KEY (id, customer_id)
                    ) PARTITION BY LIST (customer_id);
                    """,
                    reverse_sql="DROP TABLE IF EXISTS memory_sessionembeddingchunk CASCADE;",
                ),
                # Indexes.
                migrations.RunSQL(
                    sql="CREATE INDEX memory_sessionembeddingchunk_customer_idx ON memory_sessionembeddingchunk (customer_id);",
                    reverse_sql="DROP INDEX IF EXISTS memory_sessionembeddingchunk_customer_idx;",
                ),
                migrations.RunSQL(
                    sql="CREATE INDEX memory_sessionembeddingchunk_content_hash_idx ON memory_sessionembeddingchunk (content_hash);",
                    reverse_sql="DROP INDEX IF EXISTS memory_sessionembeddingchunk_content_hash_idx;",
                ),
                migrations.RunSQL(
                    sql="CREATE INDEX memory_sessionembeddingchunk_session_source_idx ON memory_sessionembeddingchunk (session_id, source, source_id);",
                    reverse_sql="DROP INDEX IF EXISTS memory_sessionembeddingchunk_session_source_idx;",
                ),
                migrations.RunSQL(
                    sql="CREATE INDEX memory_sessionembeddingchunk_session_hash_idx ON memory_sessionembeddingchunk (session_id, content_hash);",
                    reverse_sql="DROP INDEX IF EXISTS memory_sessionembeddingchunk_session_hash_idx;",
                ),
                # HNSW index (m=24, ef_construction=128) for fast ANN search.
                migrations.RunSQL(
                    sql="""
                    CREATE INDEX sessionembeddingchunk_hnsw_idx
                    ON memory_sessionembeddingchunk
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 24, ef_construction = 128);
                    """,
                    reverse_sql="DROP INDEX IF EXISTS sessionembeddingchunk_hnsw_idx;",
                ),
            ],
        ),
    ]
