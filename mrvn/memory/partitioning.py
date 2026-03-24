"""Partition management for SessionEmbeddingChunk table.

Uses psqlextra to manage list partitions by customer_id.
Partitions are created automatically when new customers are added.
"""

from typing import TYPE_CHECKING, Any
from uuid import UUID

from psqlextra.backend.schema import PostgresSchemaEditor
from psqlextra.models import PostgresPartitionedModel
from psqlextra.partitioning import PostgresPartitioningManager
from psqlextra.partitioning.config import PostgresPartitioningConfig
from psqlextra.partitioning.partition import PostgresPartition
from psqlextra.partitioning.strategy import PostgresPartitioningStrategy

if TYPE_CHECKING:
    from collections.abc import Generator


def _partition_name(customer_id: UUID) -> str:
    """Generate partition name within PostgreSQL's 63-char identifier limit.

    Format: {customer_uuid_hex} — full UUID without dashes (32 chars).
    Max full name: memory_sessionembeddingchunk_{32} = 62 chars (within 63-char limit).
    """
    return str(customer_id).replace("-", "")


class PostgresListPartition(PostgresPartition):
    """A list partition for a PostgreSQL partitioned table."""

    def __init__(self, name: str, values: list) -> None:
        self._name = name
        self.values = values

    def name(self) -> str:
        return self._name

    def deconstruct(self) -> dict:
        return {
            **super().deconstruct(),
            "values": self.values,
        }

    def create(
        self,
        model: type[PostgresPartitionedModel],
        schema_editor: PostgresSchemaEditor,
        comment: str | None = None,
    ) -> None:
        schema_editor.add_list_partition(
            model=model,
            name=self._name,
            values=self.values,
            comment=comment,
        )

    def delete(
        self,
        model: type[PostgresPartitionedModel],
        schema_editor: PostgresSchemaEditor,
    ) -> None:
        schema_editor.delete_partition(model, self._name)


class CustomerListPartitioningStrategy(PostgresPartitioningStrategy):
    """Strategy for creating list partitions per customer."""

    def to_create(self) -> "Generator[PostgresPartition]":
        """Generate partitions to create for each customer."""
        from accounts.models import Customer  # noqa: PLC0415

        for customer in Customer.objects.all():
            yield PostgresListPartition(
                name=_partition_name(customer.id),
                values=[customer.id],
            )

    def to_delete(self) -> "Generator[PostgresPartition]":
        """Generate partitions to delete (none by default)."""
        # We don't auto-delete partitions - they should be manually removed
        # when a customer is deleted if desired
        return
        yield  # noqa: RET502 - makes this a generator


def get_partitioning_manager() -> PostgresPartitioningManager:
    """Create the partitioning manager for SessionEmbeddingChunk."""
    from memory.models import SessionEmbeddingChunk  # noqa: PLC0415

    return PostgresPartitioningManager(
        configs=[
            PostgresPartitioningConfig(
                model=SessionEmbeddingChunk,
                strategy=CustomerListPartitioningStrategy(),
            ),
        ]
    )


class LazyPartitioningManager:
    """Lazy wrapper for PostgresPartitioningManager.

    Delays initialization until first access to avoid import-time DB queries.
    """

    _instance: PostgresPartitioningManager | None = None

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the actual manager."""
        if self._instance is None:
            self._instance = get_partitioning_manager()
        return getattr(self._instance, name)


# For settings.PSQLEXTRA_PARTITIONING_MANAGER
manager = LazyPartitioningManager()
