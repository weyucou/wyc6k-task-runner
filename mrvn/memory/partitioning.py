"""Partition management for EmbeddingChunk table.

Uses psqlextra to manage list partitions by composite (customer_id, agent_id).
Partitions are created automatically when new (customer_id, agent_id) pairs appear.
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


class PostgresListPartition(PostgresPartition):
    """A list partition for a PostgreSQL partitioned table.

    Supports both single-column values and composite (tuple) values
    for multi-column partition keys.
    """

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

    def _format_value(self, v: Any) -> str:
        """Format a single partition value for SQL."""
        if isinstance(v, UUID):
            return f"'{v}'"
        if isinstance(v, str):
            return f"'{v}'"
        return str(v)

    def create(
        self,
        model: type[PostgresPartitionedModel],
        schema_editor: PostgresSchemaEditor,
        comment: str | None = None,
    ) -> None:
        if self.values and isinstance(self.values[0], (list, tuple)):
            # Composite list partition — use raw SQL since psqlextra's
            # add_list_partition doesn't handle tuple values.
            table_name = model._meta.db_table
            partition_table = f"{table_name}_{self._name}"
            value_strs = ["({})".format(", ".join(self._format_value(v) for v in val)) for val in self.values]
            values_sql = ", ".join(value_strs)
            schema_editor.execute(
                f"CREATE TABLE {partition_table} PARTITION OF {table_name} FOR VALUES IN ({values_sql});"
            )
        else:
            schema_editor.add_list_partition(
                model=model,
                name=self.name(),
                values=self.values,
                comment=comment,
            )

    def delete(
        self,
        model: type[PostgresPartitionedModel],
        schema_editor: PostgresSchemaEditor,
    ) -> None:
        schema_editor.delete_partition(model, self.name())


def _partition_name(customer_id: UUID, agent_id: int) -> str:
    """Generate a short, unique partition name for a (customer_id, agent_id) pair.

    Uses the first 16 hex chars of customer_id (64 bits) + agent_id.
    Full partition table name stays within PostgreSQL's 63-char identifier limit.
    """
    uuid_hex = str(customer_id).replace("-", "")
    return f"{uuid_hex[:16]}_{agent_id}"


class AgentListPartitioningStrategy(PostgresPartitioningStrategy):
    """Strategy for creating list partitions per (customer_id, agent_id) pair."""

    def to_create(self) -> "Generator[PostgresPartition]":
        """Generate partitions to create for each unique (customer_id, agent_id) pair."""
        from memory.models import EmbeddingChunk, SENTINEL_CUSTOMER_ID  # noqa: PLC0415

        pairs = (
            EmbeddingChunk.objects.exclude(customer_id=SENTINEL_CUSTOMER_ID)
            .values_list("customer_id", "agent_id")
            .distinct()
        )
        for customer_id, agent_id in pairs:
            yield PostgresListPartition(
                name=_partition_name(customer_id, agent_id),
                values=[(customer_id, agent_id)],
            )

    def to_delete(self) -> "Generator[PostgresPartition]":
        """Generate partitions to delete (none by default)."""
        # We don't auto-delete partitions — they should be manually removed
        # when a (customer_id, agent_id) pair is no longer needed.
        return
        yield  # noqa: RET502 - makes this a generator


def get_partitioning_manager() -> PostgresPartitioningManager:
    """Create the partitioning manager for EmbeddingChunk.

    Returns:
        Configured PostgresPartitioningManager instance.
    """
    from memory.models import EmbeddingChunk  # noqa: PLC0415

    return PostgresPartitioningManager(
        configs=[
            PostgresPartitioningConfig(
                model=EmbeddingChunk,
                strategy=AgentListPartitioningStrategy(),
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
