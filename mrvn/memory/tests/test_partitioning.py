"""Tests for memory partitioning functionality."""

from unittest.mock import MagicMock

from agents.models import Agent
from django.db import connection
from django.test import TestCase, TransactionTestCase

from memory.models import EmbeddingChunk
from memory.partitioning import (
    AgentListPartitioningStrategy,
    PostgresListPartition,
    get_partitioning_manager,
)


class PostgresListPartitionTests(TestCase):
    """Tests for PostgresListPartition class."""

    def test_name_returns_partition_name(self):
        """Test that name() returns the partition name."""
        partition = PostgresListPartition(name="agent_1", values=[1])
        self.assertEqual(partition.name(), "agent_1")

    def test_deconstruct_returns_dict_with_values(self):
        """Test that deconstruct() returns dict with values."""
        partition = PostgresListPartition(name="agent_42", values=[42])
        result = partition.deconstruct()

        self.assertIn("values", result)
        self.assertEqual(result["values"], [42])

    def test_partition_with_multiple_values(self):
        """Test partition can hold multiple values."""
        partition = PostgresListPartition(name="multi_agent", values=[1, 2, 3])
        self.assertEqual(partition.values, [1, 2, 3])
        self.assertEqual(partition.deconstruct()["values"], [1, 2, 3])

    def test_create_calls_schema_editor(self):
        """Test that create() calls schema_editor.add_list_partition."""
        partition = PostgresListPartition(name="agent_5", values=[5])
        mock_model = MagicMock()
        mock_schema_editor = MagicMock()

        partition.create(model=mock_model, schema_editor=mock_schema_editor, comment="test")

        mock_schema_editor.add_list_partition.assert_called_once_with(
            model=mock_model,
            name="agent_5",
            values=[5],
            comment="test",
        )

    def test_delete_calls_schema_editor(self):
        """Test that delete() calls schema_editor.delete_partition."""
        partition = PostgresListPartition(name="agent_5", values=[5])
        mock_model = MagicMock()
        mock_schema_editor = MagicMock()

        partition.delete(model=mock_model, schema_editor=mock_schema_editor)

        mock_schema_editor.delete_partition.assert_called_once_with(mock_model, "agent_5")


class AgentListPartitioningStrategyTests(TransactionTestCase):
    """Tests for AgentListPartitioningStrategy."""

    def test_to_create_yields_partitions_for_agents(self):
        """Test that to_create() yields partition for each agent."""
        agent1 = Agent.objects.create(
            name="Agent 1",
            model_name="test-model",
        )
        agent2 = Agent.objects.create(
            name="Agent 2",
            model_name="test-model",
        )

        strategy = AgentListPartitioningStrategy()
        partitions = list(strategy.to_create())

        # Should have partitions for both agents
        partition_names = [p.name() for p in partitions]
        self.assertIn(f"agent_{agent1.id}", partition_names)
        self.assertIn(f"agent_{agent2.id}", partition_names)

        # Verify partition values
        for partition in partitions:
            if partition.name() == f"agent_{agent1.id}":
                self.assertEqual(partition.values, [agent1.id])
            elif partition.name() == f"agent_{agent2.id}":
                self.assertEqual(partition.values, [agent2.id])

    def test_to_create_empty_when_no_agents(self):
        """Test that to_create() yields nothing when no agents exist."""
        # Ensure no agents exist
        Agent.objects.all().delete()

        strategy = AgentListPartitioningStrategy()
        partitions = list(strategy.to_create())

        self.assertEqual(len(partitions), 0)

    def test_to_delete_yields_nothing(self):
        """Test that to_delete() is empty (no auto-deletion)."""
        Agent.objects.create(
            name="Agent",
            model_name="test-model",
        )

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

    def test_partition_creation_via_strategy(self):
        """Test that partitions can be created for agents via schema editor."""
        from psqlextra.backend.schema import PostgresSchemaEditor  # noqa: PLC0415

        agent = Agent.objects.create(
            name="Partition Test Agent",
            model_name="test-model",
        )

        partition_name = f"memory_embeddingchunk_agent_{agent.id}"

        # Create partition using schema editor
        with connection.schema_editor() as schema_editor:
            if isinstance(schema_editor, PostgresSchemaEditor):
                partition = PostgresListPartition(
                    name=f"agent_{agent.id}",
                    values=[agent.id],
                )
                partition.create(
                    model=EmbeddingChunk,
                    schema_editor=schema_editor,
                )

        # Verify partition exists
        self.assertTrue(
            self._partition_exists(partition_name),
            f"Partition {partition_name} should exist after creation",
        )

    def test_partition_deletion_via_strategy(self):
        """Test that partitions can be deleted via schema editor."""
        from psqlextra.backend.schema import PostgresSchemaEditor  # noqa: PLC0415

        agent = Agent.objects.create(
            name="Delete Partition Test",
            model_name="test-model",
        )

        partition_name = f"memory_embeddingchunk_agent_{agent.id}"

        # First create the partition
        with connection.schema_editor() as schema_editor:
            if isinstance(schema_editor, PostgresSchemaEditor):
                partition = PostgresListPartition(
                    name=f"agent_{agent.id}",
                    values=[agent.id],
                )
                partition.create(model=EmbeddingChunk, schema_editor=schema_editor)

        # Verify it exists
        self.assertTrue(self._partition_exists(partition_name))

        # Now delete it
        with connection.schema_editor() as schema_editor:
            if isinstance(schema_editor, PostgresSchemaEditor):
                partition = PostgresListPartition(
                    name=f"agent_{agent.id}",
                    values=[agent.id],
                )
                partition.delete(model=EmbeddingChunk, schema_editor=schema_editor)

        # Verify partition is deleted
        self.assertFalse(
            self._partition_exists(partition_name),
            f"Partition {partition_name} should not exist after deletion",
        )

    def test_default_partition_exists(self):
        """Test that the default partition exists for unassigned agents."""
        # The default partition should exist from migrations
        partitions = self._list_partitions()
        self.assertIn("memory_embeddingchunk_default", partitions)

    def test_data_routes_to_correct_partition(self):
        """Test that data is routed to the correct agent partition."""
        from psqlextra.backend.schema import PostgresSchemaEditor  # noqa: PLC0415

        agent = Agent.objects.create(
            name="Data Routing Test",
            model_name="test-model",
        )

        # Create partition for this agent
        with connection.schema_editor() as schema_editor:
            if isinstance(schema_editor, PostgresSchemaEditor):
                partition = PostgresListPartition(
                    name=f"agent_{agent.id}",
                    values=[agent.id],
                )
                partition.create(model=EmbeddingChunk, schema_editor=schema_editor)

        # Insert data for this agent
        chunk = EmbeddingChunk.objects.create(
            agent=agent,
            source="message",
            source_id=1,
            text="Test embedding chunk",
            embedding=[0.1] * 384,  # 384-dimensional vector
            content_hash="testhash123",
        )

        # Verify data was inserted
        self.assertIsNotNone(chunk.id)
        self.assertEqual(chunk.agent_id, agent.id)

        # Query back and verify
        retrieved = EmbeddingChunk.objects.get(id=chunk.id, agent_id=agent.id)
        self.assertEqual(retrieved.text, "Test embedding chunk")

    def test_data_routes_to_default_partition_without_agent_partition(self):
        """Test that data routes to default partition when no agent partition exists."""
        agent = Agent.objects.create(
            name="Default Partition Test",
            model_name="test-model",
        )

        # Don't create a specific partition - data should go to default
        chunk = EmbeddingChunk.objects.create(
            agent=agent,
            source="message",
            source_id=1,
            text="Test in default partition",
            embedding=[0.2] * 384,
            content_hash="defaulthash123",
        )

        # Data should still be queryable
        self.assertIsNotNone(chunk.id)
        retrieved = EmbeddingChunk.objects.get(id=chunk.id, agent_id=agent.id)
        self.assertEqual(retrieved.text, "Test in default partition")


class PartitioningManagerPlanTests(TransactionTestCase):
    """Tests for partitioning manager plan generation."""

    def test_manager_plan_includes_new_agents(self):
        """Test that partitioning manager plan includes partitions for new agents."""
        # Create some agents
        agent1 = Agent.objects.create(name="Plan Agent 1", model_name="test")
        agent2 = Agent.objects.create(name="Plan Agent 2", model_name="test")

        manager = get_partitioning_manager()
        plan = manager.plan()

        # Plan should include partitions to create (creations is a list)
        create_names = [p.name() for p in plan.creations]

        self.assertIn(f"agent_{agent1.id}", create_names)
        self.assertIn(f"agent_{agent2.id}", create_names)

    def test_manager_plan_no_deletions(self):
        """Test that partitioning manager plan has no deletions (by design)."""
        Agent.objects.create(name="No Delete Agent", model_name="test")

        manager = get_partitioning_manager()
        plan = manager.plan()

        # Should have no deletions (deletions is a list)
        self.assertEqual(len(plan.deletions), 0)
