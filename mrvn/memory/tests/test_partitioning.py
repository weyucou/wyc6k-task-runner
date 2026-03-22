"""Tests for memory partitioning functionality."""

from unittest.mock import MagicMock
from uuid import UUID

from accounts.models import Customer
from agents.models import Agent
from django.db import connection
from django.test import TestCase, TransactionTestCase

from memory.models import SENTINEL_CUSTOMER_ID, EmbeddingChunk
from memory.partitioning import (
    AgentListPartitioningStrategy,
    PostgresListPartition,
    _partition_name,
    get_partitioning_manager,
)


class PostgresListPartitionTests(TestCase):
    """Tests for PostgresListPartition class."""

    def test_name_returns_partition_name(self):
        """Test that name() returns the partition name."""
        partition = PostgresListPartition(name="abc123_1", values=[("abc123", 1)])
        self.assertEqual(partition.name(), "abc123_1")

    def test_deconstruct_returns_dict_with_values(self):
        """Test that deconstruct() returns dict with values."""
        customer_id = UUID("01959b3e-1234-7000-b000-123456789abc")
        partition = PostgresListPartition(
            name=_partition_name(customer_id, 42),
            values=[(customer_id, 42)],
        )
        result = partition.deconstruct()

        self.assertIn("values", result)
        self.assertEqual(result["values"], [(customer_id, 42)])

    def test_partition_with_multiple_values(self):
        """Test partition can hold multiple composite values."""
        cid1 = UUID("01959b3e-0000-7000-b000-000000000001")
        cid2 = UUID("01959b3e-0000-7000-b000-000000000002")
        partition = PostgresListPartition(
            name="multi",
            values=[(cid1, 1), (cid2, 1)],
        )
        self.assertEqual(partition.values, [(cid1, 1), (cid2, 1)])
        self.assertEqual(partition.deconstruct()["values"], [(cid1, 1), (cid2, 1)])

    def test_create_calls_schema_editor_composite(self):
        """Test that create() uses raw SQL for composite (tuple) values."""
        customer_id = UUID("01959b3e-1234-7000-b000-123456789abc")
        partition = PostgresListPartition(
            name=_partition_name(customer_id, 5),
            values=[(customer_id, 5)],
        )
        mock_model = MagicMock()
        mock_model._meta.db_table = "memory_embeddingchunk"
        mock_schema_editor = MagicMock()

        partition.create(model=mock_model, schema_editor=mock_schema_editor, comment="test")

        # Should call execute() with raw SQL (not add_list_partition) for tuple values
        mock_schema_editor.execute.assert_called_once()
        sql = mock_schema_editor.execute.call_args[0][0]
        self.assertIn("CREATE TABLE", sql)
        self.assertIn("FOR VALUES IN", sql)
        self.assertIn(str(customer_id), sql)

    def test_delete_calls_schema_editor(self):
        """Test that delete() calls schema_editor.delete_partition."""
        customer_id = UUID("01959b3e-1234-7000-b000-123456789abc")
        name = _partition_name(customer_id, 5)
        partition = PostgresListPartition(name=name, values=[(customer_id, 5)])
        mock_model = MagicMock()
        mock_schema_editor = MagicMock()

        partition.delete(model=mock_model, schema_editor=mock_schema_editor)

        mock_schema_editor.delete_partition.assert_called_once_with(mock_model, name)


class PartitionNameTests(TestCase):
    """Tests for the _partition_name helper."""

    def test_name_is_deterministic(self):
        """Same (customer_id, agent_id) always produces the same name."""
        customer_id = UUID("01959b3e-1234-7000-b000-123456789abc")
        self.assertEqual(_partition_name(customer_id, 7), _partition_name(customer_id, 7))

    def test_name_differs_by_customer(self):
        """Different customer_ids produce different names."""
        cid1 = UUID("01959b3e-0000-7000-b000-000000000001")
        cid2 = UUID("ffffffff-0000-7000-b000-000000000002")
        self.assertNotEqual(_partition_name(cid1, 1), _partition_name(cid2, 1))

    def test_name_differs_by_agent(self):
        """Different agent_ids produce different names."""
        customer_id = UUID("01959b3e-1234-7000-b000-123456789abc")
        self.assertNotEqual(_partition_name(customer_id, 1), _partition_name(customer_id, 2))

    def test_full_table_name_within_postgres_limit(self):
        """Full partition table name stays within PostgreSQL's 63-char identifier limit."""
        customer_id = UUID("01959b3e-1234-7000-b000-123456789abc")
        # Use a large agent_id to test the worst case
        name = _partition_name(customer_id, 9_999_999_999_999_999_999)
        full_name = f"memory_embeddingchunk_{name}"
        self.assertLessEqual(len(full_name), 63, f"Name too long: {full_name!r}")


class AgentListPartitioningStrategyTests(TransactionTestCase):
    """Tests for AgentListPartitioningStrategy."""

    def _make_customer(self, name: str = "Test Customer") -> Customer:
        return Customer.objects.create(name=name)

    def _make_agent(self, name: str = "Agent") -> Agent:
        return Agent.objects.create(name=name, model_name="test-model")

    def test_to_create_yields_partitions_for_indexed_pairs(self):
        """to_create() yields partitions for each unique (customer_id, agent_id) pair."""
        from psqlextra.backend.schema import PostgresSchemaEditor  # noqa: PLC0415

        customer = self._make_customer()
        agent = self._make_agent()

        # Create partition and insert a chunk so the strategy can discover the pair
        with connection.schema_editor() as schema_editor:
            if isinstance(schema_editor, PostgresSchemaEditor):
                partition = PostgresListPartition(
                    name=_partition_name(customer.id, agent.id),
                    values=[(customer.id, agent.id)],
                )
                partition.create(model=EmbeddingChunk, schema_editor=schema_editor)

        EmbeddingChunk.objects.create(
            customer_id=customer.id,
            agent=agent,
            source="message",
            source_id=1,
            text="hello",
            embedding=[0.1] * 384,
            content_hash="hash1",
        )

        strategy = AgentListPartitioningStrategy()
        partitions = list(strategy.to_create())

        pair_names = [p.name() for p in partitions]
        self.assertIn(_partition_name(customer.id, agent.id), pair_names)

    def test_to_create_excludes_sentinel_customer(self):
        """to_create() does not yield partitions for the sentinel customer_id."""
        agent = self._make_agent()

        # Insert a chunk with sentinel customer_id (goes to default partition)
        EmbeddingChunk.objects.create(
            customer_id=SENTINEL_CUSTOMER_ID,
            agent=agent,
            source="message",
            source_id=1,
            text="sentinel chunk",
            embedding=[0.1] * 384,
            content_hash="sentinelhash",
        )

        strategy = AgentListPartitioningStrategy()
        partitions = list(strategy.to_create())

        sentinel_name = _partition_name(SENTINEL_CUSTOMER_ID, agent.id)
        partition_names = [p.name() for p in partitions]
        self.assertNotIn(sentinel_name, partition_names)

    def test_to_create_empty_when_no_non_sentinel_chunks(self):
        """to_create() yields nothing when there are no non-sentinel chunks."""
        Agent.objects.all().delete()

        strategy = AgentListPartitioningStrategy()
        partitions = list(strategy.to_create())

        self.assertEqual(len(partitions), 0)

    def test_to_delete_yields_nothing(self):
        """to_delete() is empty (no auto-deletion)."""
        strategy = AgentListPartitioningStrategy()
        self.assertEqual(list(strategy.to_delete()), [])


class GetPartitioningManagerTests(TestCase):
    """Tests for get_partitioning_manager function."""

    def test_returns_manager_with_embedding_chunk_config(self):
        """Test that manager is configured for EmbeddingChunk."""
        manager = get_partitioning_manager()

        self.assertIsNotNone(manager)
        self.assertTrue(hasattr(manager, "configs"))
        self.assertEqual(len(manager.configs), 1)

        config = manager.configs[0]
        self.assertEqual(config.model, EmbeddingChunk)
        self.assertIsInstance(config.strategy, AgentListPartitioningStrategy)


class PartitionIntegrationTests(TransactionTestCase):
    """Integration tests for partition creation/deletion."""

    def _partition_exists(self, partition_name: str) -> bool:
        """Check if a partition exists in the database."""
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.relname = %s
                    AND c.relispartition = true
                )
                """,
                [partition_name],
            )
            return cursor.fetchone()[0]

    def _list_partitions(self) -> list[str]:
        """List all partitions of memory_embeddingchunk table."""
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT c.relname
                FROM pg_inherits i
                JOIN pg_class c ON c.oid = i.inhrelid
                JOIN pg_class p ON p.oid = i.inhparent
                WHERE p.relname = 'memory_embeddingchunk'
                """,
            )
            return [row[0] for row in cursor.fetchall()]

    def _make_customer(self, name: str = "Test Customer") -> Customer:
        return Customer.objects.create(name=name)

    def _make_agent(self, name: str = "Test Agent") -> Agent:
        return Agent.objects.create(name=name, model_name="test-model")

    def test_partition_creation_via_strategy(self):
        """Partitions can be created for (customer_id, agent_id) pairs."""
        from psqlextra.backend.schema import PostgresSchemaEditor  # noqa: PLC0415

        customer = self._make_customer("Partition Test Customer")
        agent = self._make_agent("Partition Test Agent")

        name = _partition_name(customer.id, agent.id)
        partition_table = f"memory_embeddingchunk_{name}"

        with connection.schema_editor() as schema_editor:
            if isinstance(schema_editor, PostgresSchemaEditor):
                partition = PostgresListPartition(
                    name=name,
                    values=[(customer.id, agent.id)],
                )
                partition.create(model=EmbeddingChunk, schema_editor=schema_editor)

        self.assertTrue(
            self._partition_exists(partition_table),
            f"Partition {partition_table} should exist after creation",
        )

    def test_partition_deletion_via_strategy(self):
        """Partitions can be deleted via schema editor."""
        from psqlextra.backend.schema import PostgresSchemaEditor  # noqa: PLC0415

        customer = self._make_customer("Delete Customer")
        agent = self._make_agent("Delete Agent")
        name = _partition_name(customer.id, agent.id)
        partition_table = f"memory_embeddingchunk_{name}"

        with connection.schema_editor() as schema_editor:
            if isinstance(schema_editor, PostgresSchemaEditor):
                partition = PostgresListPartition(name=name, values=[(customer.id, agent.id)])
                partition.create(model=EmbeddingChunk, schema_editor=schema_editor)

        self.assertTrue(self._partition_exists(partition_table))

        with connection.schema_editor() as schema_editor:
            if isinstance(schema_editor, PostgresSchemaEditor):
                partition = PostgresListPartition(name=name, values=[(customer.id, agent.id)])
                partition.delete(model=EmbeddingChunk, schema_editor=schema_editor)

        self.assertFalse(
            self._partition_exists(partition_table),
            f"Partition {partition_table} should not exist after deletion",
        )

    def test_default_partition_exists(self):
        """The default partition exists from migrations."""
        partitions = self._list_partitions()
        self.assertIn("memory_embeddingchunk_default", partitions)

    def test_data_routes_to_correct_partition(self):
        """Data is routed to the correct (customer_id, agent_id) partition."""
        from psqlextra.backend.schema import PostgresSchemaEditor  # noqa: PLC0415

        customer = self._make_customer("Routing Customer")
        agent = self._make_agent("Routing Agent")
        name = _partition_name(customer.id, agent.id)

        with connection.schema_editor() as schema_editor:
            if isinstance(schema_editor, PostgresSchemaEditor):
                partition = PostgresListPartition(name=name, values=[(customer.id, agent.id)])
                partition.create(model=EmbeddingChunk, schema_editor=schema_editor)

        chunk = EmbeddingChunk.objects.create(
            customer_id=customer.id,
            agent=agent,
            source="message",
            source_id=1,
            text="Test embedding chunk",
            embedding=[0.1] * 384,
            content_hash="testhash123",
        )

        self.assertIsNotNone(chunk.id)
        self.assertEqual(chunk.customer_id, customer.id)
        self.assertEqual(chunk.agent_id, agent.id)

        retrieved = EmbeddingChunk.objects.get(id=chunk.id, customer_id=customer.id, agent_id=agent.id)
        self.assertEqual(retrieved.text, "Test embedding chunk")

    def test_data_routes_to_default_partition_for_sentinel(self):
        """Data with sentinel customer_id routes to the default partition."""
        agent = self._make_agent("Sentinel Agent")

        chunk = EmbeddingChunk.objects.create(
            customer_id=SENTINEL_CUSTOMER_ID,
            agent=agent,
            source="message",
            source_id=1,
            text="Test in default partition",
            embedding=[0.2] * 384,
            content_hash="defaulthash123",
        )

        self.assertIsNotNone(chunk.id)
        retrieved = EmbeddingChunk.objects.get(id=chunk.id, customer_id=SENTINEL_CUSTOMER_ID, agent_id=agent.id)
        self.assertEqual(retrieved.text, "Test in default partition")

    def test_cross_customer_isolation(self):
        """Agents in different customers cannot read each other's chunks."""
        from psqlextra.backend.schema import PostgresSchemaEditor  # noqa: PLC0415

        customer_a = self._make_customer("Customer A")
        customer_b = self._make_customer("Customer B")
        agent_a = self._make_agent("Agent A")
        agent_b = self._make_agent("Agent B")

        # Create partitions for both pairs
        with connection.schema_editor() as schema_editor:
            if isinstance(schema_editor, PostgresSchemaEditor):
                for cid, aid in [(customer_a.id, agent_a.id), (customer_b.id, agent_b.id)]:
                    PostgresListPartition(
                        name=_partition_name(cid, aid),
                        values=[(cid, aid)],
                    ).create(model=EmbeddingChunk, schema_editor=schema_editor)

        # Index one chunk per customer
        EmbeddingChunk.objects.create(
            customer_id=customer_a.id,
            agent=agent_a,
            source="message",
            source_id=10,
            text="Customer A secret",
            embedding=[0.1] * 384,
            content_hash="hash_a",
        )
        EmbeddingChunk.objects.create(
            customer_id=customer_b.id,
            agent=agent_b,
            source="message",
            source_id=20,
            text="Customer B secret",
            embedding=[0.2] * 384,
            content_hash="hash_b",
        )

        # Customer A query must not see Customer B's chunk
        a_chunks = list(EmbeddingChunk.objects.filter(customer_id=customer_a.id).values_list("text", flat=True))
        self.assertIn("Customer A secret", a_chunks)
        self.assertNotIn("Customer B secret", a_chunks)

        # Customer B query must not see Customer A's chunk
        b_chunks = list(EmbeddingChunk.objects.filter(customer_id=customer_b.id).values_list("text", flat=True))
        self.assertIn("Customer B secret", b_chunks)
        self.assertNotIn("Customer A secret", b_chunks)


class PartitioningManagerPlanTests(TransactionTestCase):
    """Tests for partitioning manager plan generation."""

    def _make_customer(self, name: str = "Plan Customer") -> Customer:
        return Customer.objects.create(name=name)

    def _make_agent(self, name: str = "Plan Agent") -> Agent:
        return Agent.objects.create(name=name, model_name="test")

    def test_manager_plan_includes_indexed_pairs(self):
        """Partitioning manager plan includes partitions for indexed (customer_id, agent_id) pairs."""
        from psqlextra.backend.schema import PostgresSchemaEditor  # noqa: PLC0415

        customer = self._make_customer("Plan Customer 1")
        agent = self._make_agent("Plan Agent 1")
        name = _partition_name(customer.id, agent.id)

        # Create partition and insert a chunk so the strategy discovers the pair
        with connection.schema_editor() as schema_editor:
            if isinstance(schema_editor, PostgresSchemaEditor):
                PostgresListPartition(name=name, values=[(customer.id, agent.id)]).create(
                    model=EmbeddingChunk, schema_editor=schema_editor
                )

        EmbeddingChunk.objects.create(
            customer_id=customer.id,
            agent=agent,
            source="message",
            source_id=1,
            text="plan test",
            embedding=[0.1] * 384,
            content_hash="planhash",
        )

        manager = get_partitioning_manager()
        plan = manager.plan()

        create_names = [p.name() for p in plan.creations]
        self.assertIn(name, create_names)

    def test_manager_plan_no_deletions(self):
        """Partitioning manager plan has no deletions (by design)."""
        manager = get_partitioning_manager()
        plan = manager.plan()
        self.assertEqual(len(plan.deletions), 0)
