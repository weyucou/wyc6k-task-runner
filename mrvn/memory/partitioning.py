"""Partition management for EmbeddingChunk table.

Uses psqlextra to manage list partitions by (customer_id, agent_id).
Partitions are created automatically when new agents are added.
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


def _partition_name(customer_id: UUID, agent_id: int) -> str:
    """Generate partition name within PostgreSQL's 63-char identifier limit.

    Format: {first_16_hex_of_customer_uuid}_{agent_id}
    Max length: 16 + 1 + 10 = 27 chars (plus table prefix memory_embeddingchunk_ = 49 total).
    """
    customer_hex = str(customer_id).replace("-", "")[:16]
    return f"{customer_hex}_{agent_id}"


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
        if self.values and isinstance(self.values[0], (list, tuple)):
            # Composite key partition — psqlextra's add_list_partition doesn't support
            # tuple values, so use raw SQL for (customer_id, agent_id) pairs.
            from django.db import connection  # noqa: PLC0415

            table_name = model._meta.db_table
            partition_table_name = f"{table_name}_{self._name}"
            placeholders = ", ".join("(%s::uuid, %s)" for _ in self.values)
            params = [item for v in self.values for item in (str(v[0]), v[1])]
            with connection.cursor() as cursor:
                cursor.execute(
                    f"CREATE TABLE {partition_table_name} PARTITION OF {table_name} "
                    f"FOR VALUES IN ({placeholders})",
                    params,
                )
        else:
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


class AgentListPartitioningStrategy(PostgresPartitioningStrategy):
    """Strategy for creating list partitions per (customer_id, agent_id) pair."""

    def to_create(self) -> "Generator[PostgresPartition]":
        """Generate partitions to create for each agent."""
        from agents.models import Agent  # noqa: PLC0415
        from memory.models import SENTINEL_CUSTOMER_ID  # noqa: PLC0415

        for agent in Agent.objects.all():
            # Use agent.customer_id once Agent gains that FK (issue #2).
            # Until then, all agents land in the sentinel partition.
            customer_id = getattr(agent, "customer_id", None) or SENTINEL_CUSTOMER_ID
            yield PostgresListPartition(
                name=_partition_name(customer_id, agent.id),
                values=[(customer_id, agent.id)],
            )

    def to_delete(self) -> "Generator[PostgresPartition]":
        """Generate partitions to delete (none by default)."""
        # We don't auto-delete partitions - they should be manually removed
        # when an agent is deleted if desired
        return
        yield  # noqa: RET502 - makes this a generator


def get_partitioning_manager() -> PostgresPartitioningManager:
    """Create the partitioning manager for EmbeddingChunk."""
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
