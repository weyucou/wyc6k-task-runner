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


class PartitionNameTests(TestCase):
    """Tests for the _partition_name helper."""

    def test_format_is_hex_underscore_agent(self):
        """Partition name uses first 16 hex chars of customer UUID + agent_id."""
        customer_id = UUID("12345678-1234-1234-1234-123456789abc")
        name = _partition_name(customer_id, 42)
        self.assertEqual(name, "1234567812341234_42")

    def test_sentinel_customer_id(self):
        """Sentinel UUID produces all-zero hex prefix."""
        name = _partition_name(SENTINEL_CUSTOMER_ID, 1)
        self.assertEqual(name, "0000000000000000_1")

    def test_name_stays_within_63_chars_with_table_prefix(self):
        """Full partition table name (prefix + name) stays within 63 chars."""
        # Longest realistic agent_id (max bigint has 19 digits, but typical IDs are much shorter)
        customer_id = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
        name = _partition_name(customer_id, 9999999999)
        full_name = f"memory_embeddingchunk_{name}"
        self.assertLessEqual(len(full_name), 63)


class PostgresListPartitionTests(TestCase):
    """Tests for PostgresListPartition class."""

    def test_name_returns_partition_name(self):
        """Test that name() returns the partition name."""
        partition = PostgresListPartition(name="0000000000000000_1", values=[(SENTINEL_CUSTOMER_ID, 1)])
        self.assertEqual(partition.name(), "0000000000000000_1")

    def test_deconstruct_returns_dict_with_values(self):
        """Test that deconstruct() returns dict with values."""
        customer_id = UUID("12345678-0000-0000-0000-000000000000")
        partition = PostgresListPartition(
            name=_partition_name(customer_id, 42),
            values=[(customer_id, 42)],
        )
        result = partition.deconstruct()

        self.assertIn("values", result)
        self.assertEqual(result["values"], [(customer_id, 42)])

    def test_partition_with_multiple_values(self):
        """Test partition can hold multiple value tuples."""
        c1 = UUID("aaaaaaaa-0000-0000-0000-000000000000")
        c2 = UUID("bbbbbbbb-0000-0000-0000-000000000000")
        partition = PostgresListPartition(name="multi", values=[(c1, 1), (c2, 2)])
        self.assertEqual(len(partition.values), 2)

    def test_delete_calls_schema_editor(self):
        """Test that delete() calls schema_editor.delete_partition."""
        partition = PostgresListPartition(name="0000000000000000_5", values=[(SENTINEL_CUSTOMER_ID, 5)])
        mock_model = MagicMock()
        mock_schema_editor = MagicMock()

        partition.delete(model=mock_model, schema_editor=mock_schema_editor)

        mock_schema_editor.delete_partition.assert_called_once_with(mock_model, "0000000000000000_5")


class AgentListPartitioningStrategyTests(TransactionTestCase):
    """Tests for AgentListPartitioningStrategy."""

    def test_to_create_yields_partitions_for_agents(self):
        """Test that to_create() yields a partition for each agent."""
        agent1 = Agent.objects.create(name="Agent 1", model_name="test-model")
        agent2 = Agent.objects.create(name="Agent 2", model_name="test-model")

        strategy = AgentListPartitioningStrategy()
        partitions = list(strategy.to_create())

        partition_names = [p.name() for p in partitions]
        expected1 = _partition_name(SENTINEL_CUSTOMER_ID, agent1.id)
        expected2 = _partition_name(SENTINEL_CUSTOMER_ID, agent2.id)
        self.assertIn(expected1, partition_names)
        self.assertIn(expected2, partition_names)

    def test_to_create_values_are_tuples(self):
        """Test that partition values are (customer_id, agent_id) tuples."""
        agent = Agent.objects.create(name="Tuple Agent", model_name="test-model")

        strategy = AgentListPartitioningStrategy()
        partitions = list(strategy.to_create())

        self.assertEqual(len(partitions), 1)
        values = partitions[0].values
        self.assertEqual(len(values), 1)
        customer_id, agent_id = values[0]
        self.assertEqual(customer_id, SENTINEL_CUSTOMER_ID)
        self.assertEqual(agent_id, agent.id)

    def test_to_create_empty_when_no_agents(self):
        """Test that to_create() yields nothing when no agents exist."""
        Agent.objects.all().delete()

        strategy = AgentListPartitioningStrategy()
        partitions = list(strategy.to_create())

        self.assertEqual(len(partitions), 0)

    def test_to_delete_yields_nothing(self):
        """Test that to_delete() is empty (no auto-deletion)."""
        Agent.objects.create(name="Agent", model_name="test-model")

        strategy = AgentListPartitioningStrategy()
        partitions = list(strategy.to_delete())

        self.assertEqual(len(partitions), 0)


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

    def _make_agent(self, name: str = "Test Agent") -> Agent:
        return Agent.objects.create(name=name, model_name="test-model")

    def _make_partition(self, agent: Agent, customer_id: UUID = SENTINEL_CUSTOMER_ID) -> PostgresListPartition:
        from psqlextra.backend.schema import PostgresSchemaEditor  # noqa: PLC0415

        name = _partition_name(customer_id, agent.id)
        partition = PostgresListPartition(name=name, values=[(customer_id, agent.id)])

        with connection.schema_editor() as schema_editor:
            if isinstance(schema_editor, PostgresSchemaEditor):
                partition.create(model=EmbeddingChunk, schema_editor=schema_editor)

        return partition

    def test_partition_creation_via_strategy(self):
        """Test that composite-key partitions can be created via schema editor."""
        agent = self._make_agent("Partition Test Agent")
        partition = self._make_partition(agent)

        full_name = f"memory_embeddingchunk_{partition.name()}"
        self.assertTrue(
            self._partition_exists(full_name),
            f"Partition {full_name} should exist after creation",
        )

    def test_partition_deletion_via_strategy(self):
        """Test that partitions can be deleted via schema editor."""
        from psqlextra.backend.schema import PostgresSchemaEditor  # noqa: PLC0415

        agent = self._make_agent("Delete Partition Test")
        partition = self._make_partition(agent)
        full_name = f"memory_embeddingchunk_{partition.name()}"

        self.assertTrue(self._partition_exists(full_name))

        with connection.schema_editor() as schema_editor:
            if isinstance(schema_editor, PostgresSchemaEditor):
                partition.delete(model=EmbeddingChunk, schema_editor=schema_editor)

        self.assertFalse(
            self._partition_exists(full_name),
            f"Partition {full_name} should not exist after deletion",
        )

    def test_default_partition_exists(self):
        """Test that the default partition exists for unassigned rows."""
        partitions = self._list_partitions()
        self.assertIn("memory_embeddingchunk_default", partitions)

    def test_data_routes_to_correct_partition(self):
        """Test that data routes to the correct composite-key partition."""
        agent = self._make_agent("Data Routing Test")
        self._make_partition(agent)

        chunk = EmbeddingChunk.objects.create(
            agent=agent,
            customer_id=SENTINEL_CUSTOMER_ID,
            source="message",
            source_id=1,
            text="Test embedding chunk",
            embedding=[0.1] * 384,
            content_hash="testhash123",
        )

        self.assertIsNotNone(chunk.id)
        retrieved = EmbeddingChunk.objects.get(id=chunk.id, agent_id=agent.id, customer_id=SENTINEL_CUSTOMER_ID)
        self.assertEqual(retrieved.text, "Test embedding chunk")

    def test_data_routes_to_default_partition_for_sentinel(self):
        """Test that rows with sentinel customer_id go to the default partition."""
        agent = self._make_agent("Default Partition Test")

        # No explicit partition — data should fall into default
        chunk = EmbeddingChunk.objects.create(
            agent=agent,
            customer_id=SENTINEL_CUSTOMER_ID,
            source="message",
            source_id=1,
            text="Test in default partition",
            embedding=[0.2] * 384,
            content_hash="defaulthash123",
        )

        self.assertIsNotNone(chunk.id)
        retrieved = EmbeddingChunk.objects.get(id=chunk.id, agent_id=agent.id)
        self.assertEqual(retrieved.text, "Test in default partition")

    def test_cross_customer_isolation(self):
        """Test that agents in different customers cannot see each other's chunks."""
        customer_a = Customer.objects.create(name="Customer A", github_org="org-a")
        customer_b = Customer.objects.create(name="Customer B", github_org="org-b")

        agent_a = self._make_agent("Agent A")
        agent_b = self._make_agent("Agent B")

        self._make_partition(agent_a, customer_id=customer_a.id)
        self._make_partition(agent_b, customer_id=customer_b.id)

        EmbeddingChunk.objects.create(
            agent=agent_a,
            customer_id=customer_a.id,
            source="message",
            source_id=100,
            text="Customer A secret",
            embedding=[0.3] * 384,
            content_hash="hash_a",
        )
        EmbeddingChunk.objects.create(
            agent=agent_b,
            customer_id=customer_b.id,
            source="message",
            source_id=200,
            text="Customer B secret",
            embedding=[0.4] * 384,
            content_hash="hash_b",
        )

        # Customer A can only see their own chunks
        a_chunks = EmbeddingChunk.objects.filter(customer_id=customer_a.id)
        self.assertEqual(a_chunks.count(), 1)
        self.assertEqual(a_chunks.first().text, "Customer A secret")

        # Customer B can only see their own chunks
        b_chunks = EmbeddingChunk.objects.filter(customer_id=customer_b.id)
        self.assertEqual(b_chunks.count(), 1)
        self.assertEqual(b_chunks.first().text, "Customer B secret")


class PartitioningManagerPlanTests(TransactionTestCase):
    """Tests for partitioning manager plan generation."""

    def test_manager_plan_includes_new_agents(self):
        """Test that partitioning manager plan includes partitions for new agents."""
        agent1 = Agent.objects.create(name="Plan Agent 1", model_name="test")
        agent2 = Agent.objects.create(name="Plan Agent 2", model_name="test")

        manager = get_partitioning_manager()
        plan = manager.plan()

        create_names = [p.name() for p in plan.creations]

        expected1 = _partition_name(SENTINEL_CUSTOMER_ID, agent1.id)
        expected2 = _partition_name(SENTINEL_CUSTOMER_ID, agent2.id)
        self.assertIn(expected1, create_names)
        self.assertIn(expected2, create_names)

    def test_manager_plan_no_deletions(self):
        """Test that partitioning manager plan has no deletions (by design)."""
        Agent.objects.create(name="No Delete Agent", model_name="test")

        manager = get_partitioning_manager()
        plan = manager.plan()

        self.assertEqual(len(plan.deletions), 0)
