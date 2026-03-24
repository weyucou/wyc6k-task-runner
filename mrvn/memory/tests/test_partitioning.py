"""Tests for memory partitioning functionality."""

from unittest.mock import MagicMock
from uuid import UUID

from accounts.models import Customer
from django.db import connection
from django.test import TestCase, TransactionTestCase

from memory.models import SessionEmbeddingChunk
from memory.partitioning import (
    CustomerListPartitioningStrategy,
    PostgresListPartition,
    _partition_name,
    get_partitioning_manager,
)


class PartitionNameTests(TestCase):
    """Tests for the _partition_name helper."""

    def test_format_is_full_uuid_hex(self):
        """Partition name is the full UUID without dashes (32 hex chars)."""
        customer_id = UUID("12345678-1234-1234-1234-123456789abc")
        name = _partition_name(customer_id)
        self.assertEqual(name, "12345678123412341234123456789abc")

    def test_name_length_is_32_chars(self):
        """Partition name is always 32 characters."""
        customer_id = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
        name = _partition_name(customer_id)
        self.assertEqual(len(name), 32)

    def test_name_stays_within_63_chars_with_table_prefix(self):
        """Full partition table name (prefix + name) stays within 63 chars."""
        customer_id = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
        name = _partition_name(customer_id)
        full_name = f"memory_sessionembeddingchunk_{name}"
        self.assertLessEqual(len(full_name), 63)


class PostgresListPartitionTests(TestCase):
    """Tests for PostgresListPartition class."""

    def test_name_returns_partition_name(self):
        """Test that name() returns the partition name."""
        customer_id = UUID("12345678-1234-1234-1234-123456789abc")
        partition = PostgresListPartition(name=_partition_name(customer_id), values=[customer_id])
        self.assertEqual(partition.name(), "12345678123412341234123456789abc")

    def test_deconstruct_returns_dict_with_values(self):
        """Test that deconstruct() returns dict with values."""
        customer_id = UUID("12345678-0000-0000-0000-000000000000")
        partition = PostgresListPartition(
            name=_partition_name(customer_id),
            values=[customer_id],
        )
        result = partition.deconstruct()

        self.assertIn("values", result)
        self.assertEqual(result["values"], [customer_id])

    def test_partition_with_multiple_values(self):
        """Test partition can hold multiple values."""
        c1 = UUID("aaaaaaaa-0000-0000-0000-000000000000")
        c2 = UUID("bbbbbbbb-0000-0000-0000-000000000000")
        partition = PostgresListPartition(name="multi", values=[c1, c2])
        self.assertEqual(len(partition.values), 2)

    def test_delete_calls_schema_editor(self):
        """Test that delete() calls schema_editor.delete_partition."""
        customer_id = UUID("12345678-1234-1234-1234-123456789abc")
        name = _partition_name(customer_id)
        partition = PostgresListPartition(name=name, values=[customer_id])
        mock_model = MagicMock()
        mock_schema_editor = MagicMock()

        partition.delete(model=mock_model, schema_editor=mock_schema_editor)

        mock_schema_editor.delete_partition.assert_called_once_with(mock_model, name)


class CustomerListPartitioningStrategyTests(TransactionTestCase):
    """Tests for CustomerListPartitioningStrategy."""

    def test_to_create_yields_partitions_for_customers(self):
        """Test that to_create() yields a partition for each customer."""
        customer1 = Customer.objects.create(name="Customer 1", github_org="org-1")
        customer2 = Customer.objects.create(name="Customer 2", github_org="org-2")

        strategy = CustomerListPartitioningStrategy()
        partitions = list(strategy.to_create())

        partition_names = [p.name() for p in partitions]
        self.assertIn(_partition_name(customer1.id), partition_names)
        self.assertIn(_partition_name(customer2.id), partition_names)

    def test_to_create_values_are_customer_ids(self):
        """Test that partition values are customer_id UUIDs."""
        customer = Customer.objects.create(name="Values Customer", github_org="org-vals")

        strategy = CustomerListPartitioningStrategy()
        partitions = list(strategy.to_create())

        self.assertEqual(len(partitions), 1)
        values = partitions[0].values
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0], customer.id)

    def test_to_create_empty_when_no_customers(self):
        """Test that to_create() yields nothing when no customers exist."""
        Customer.objects.all().delete()

        strategy = CustomerListPartitioningStrategy()
        partitions = list(strategy.to_create())

        self.assertEqual(len(partitions), 0)

    def test_to_delete_yields_nothing(self):
        """Test that to_delete() is empty (no auto-deletion)."""
        Customer.objects.create(name="Customer", github_org="org-del")

        strategy = CustomerListPartitioningStrategy()
        partitions = list(strategy.to_delete())

        self.assertEqual(len(partitions), 0)


class GetPartitioningManagerTests(TestCase):
    """Tests for get_partitioning_manager function."""

    def test_returns_manager_with_session_embedding_chunk_config(self):
        """Test that manager is configured for SessionEmbeddingChunk."""
        manager = get_partitioning_manager()

        self.assertIsNotNone(manager)
        self.assertTrue(hasattr(manager, "configs"))
        self.assertEqual(len(manager.configs), 1)

        config = manager.configs[0]
        self.assertEqual(config.model, SessionEmbeddingChunk)
        self.assertIsInstance(config.strategy, CustomerListPartitioningStrategy)


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
        """List all partitions of memory_sessionembeddingchunk table."""
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT c.relname
                FROM pg_inherits i
                JOIN pg_class c ON c.oid = i.inhrelid
                JOIN pg_class p ON p.oid = i.inhparent
                WHERE p.relname = 'memory_sessionembeddingchunk'
                """,
            )
            return [row[0] for row in cursor.fetchall()]

    def _make_customer(self, name: str = "Test Customer", org: str = "test-org") -> Customer:
        return Customer.objects.create(name=name, github_org=org)

    def _make_partition(self, customer_id: UUID) -> PostgresListPartition:
        from psqlextra.backend.schema import PostgresSchemaEditor  # noqa: PLC0415

        name = _partition_name(customer_id)
        partition = PostgresListPartition(name=name, values=[customer_id])

        with connection.schema_editor() as schema_editor:
            if isinstance(schema_editor, PostgresSchemaEditor):
                partition.create(model=SessionEmbeddingChunk, schema_editor=schema_editor)

        return partition

    def test_partition_creation_via_strategy(self):
        """Test that customer partitions can be created via schema editor."""
        customer = self._make_customer("Partition Test Customer", "org-pt")
        partition = self._make_partition(customer.id)

        full_name = f"memory_sessionembeddingchunk_{partition.name()}"
        self.assertTrue(
            self._partition_exists(full_name),
            f"Partition {full_name} should exist after creation",
        )

    def test_partition_deletion_via_strategy(self):
        """Test that partitions can be deleted via schema editor."""
        from psqlextra.backend.schema import PostgresSchemaEditor  # noqa: PLC0415

        customer = self._make_customer("Delete Partition Test", "org-dp")
        partition = self._make_partition(customer.id)
        full_name = f"memory_sessionembeddingchunk_{partition.name()}"

        self.assertTrue(self._partition_exists(full_name))

        with connection.schema_editor() as schema_editor:
            if isinstance(schema_editor, PostgresSchemaEditor):
                partition.delete(model=SessionEmbeddingChunk, schema_editor=schema_editor)

        self.assertFalse(
            self._partition_exists(full_name),
            f"Partition {full_name} should not exist after deletion",
        )

    def test_data_routes_to_correct_partition(self):
        """Test that data routes to the correct customer partition."""
        from agents.models import Agent  # noqa: PLC0415

        customer = self._make_customer("Data Routing Test", "org-dr")
        agent = Agent.objects.create(name="Test Agent", model_name="test-model")
        self._make_partition(customer.id)

        chunk = SessionEmbeddingChunk.objects.create(
            agent=agent,
            customer_id=customer.id,
            source="message",
            source_id=1,
            text="Test embedding chunk",
            embedding=[0.1] * 384,
            content_hash="testhash123",
        )

        self.assertIsNotNone(chunk.id)
        retrieved = SessionEmbeddingChunk.objects.get(id=chunk.id, customer_id=customer.id)
        self.assertEqual(retrieved.text, "Test embedding chunk")

    def test_cross_customer_isolation(self):
        """Test that chunks from different customers are isolated by partition."""
        from agents.models import Agent  # noqa: PLC0415

        customer_a = self._make_customer("Customer A", "org-a")
        customer_b = self._make_customer("Customer B", "org-b")
        agent = Agent.objects.create(name="Shared Agent", model_name="test-model")

        self._make_partition(customer_a.id)
        self._make_partition(customer_b.id)

        SessionEmbeddingChunk.objects.create(
            agent=agent,
            customer_id=customer_a.id,
            source="message",
            source_id=100,
            text="Customer A secret",
            embedding=[0.3] * 384,
            content_hash="hash_a",
        )
        SessionEmbeddingChunk.objects.create(
            agent=agent,
            customer_id=customer_b.id,
            source="message",
            source_id=200,
            text="Customer B secret",
            embedding=[0.4] * 384,
            content_hash="hash_b",
        )

        a_chunks = SessionEmbeddingChunk.objects.filter(customer_id=customer_a.id)
        self.assertEqual(a_chunks.count(), 1)
        self.assertEqual(a_chunks.first().text, "Customer A secret")

        b_chunks = SessionEmbeddingChunk.objects.filter(customer_id=customer_b.id)
        self.assertEqual(b_chunks.count(), 1)
        self.assertEqual(b_chunks.first().text, "Customer B secret")


class PartitioningManagerPlanTests(TransactionTestCase):
    """Tests for partitioning manager plan generation."""

    def test_manager_plan_includes_new_customers(self):
        """Test that partitioning manager plan includes partitions for new customers."""
        customer1 = Customer.objects.create(name="Plan Customer 1", github_org="org-plan-1")
        customer2 = Customer.objects.create(name="Plan Customer 2", github_org="org-plan-2")

        manager = get_partitioning_manager()
        plan = manager.plan()

        create_names = [p.name() for p in plan.creations]
        self.assertIn(_partition_name(customer1.id), create_names)
        self.assertIn(_partition_name(customer2.id), create_names)

    def test_manager_plan_no_deletions(self):
        """Test that partitioning manager plan has no deletions (by design)."""
        Customer.objects.create(name="No Delete Customer", github_org="org-nd")

        manager = get_partitioning_manager()
        plan = manager.plan()

        self.assertEqual(len(plan.deletions), 0)
