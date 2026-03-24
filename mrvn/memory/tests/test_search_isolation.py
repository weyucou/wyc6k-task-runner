"""Unit tests for MemorySearchService cross-customer isolation via text_search."""

from unittest.mock import MagicMock, patch
from uuid import UUID

from django.test import SimpleTestCase

CUSTOMER_A_ID = UUID("aaaaaaaa-0000-0000-0000-000000000001")
CUSTOMER_B_ID = UUID("bbbbbbbb-0000-0000-0000-000000000002")


def _make_mock_session(customer_id: UUID) -> MagicMock:
    """Build a minimal mock Session with customer_id set."""
    session = MagicMock()
    session.id = 1
    session.customer_id = customer_id
    return session


class TextSearchCustomerIsolationTests(SimpleTestCase):
    """Verify text_search scopes its SessionEmbeddingChunk subquery by customer_id from session."""

    def _make_service(self):
        from memory.search import MemorySearchConfig, MemorySearchService

        config = MemorySearchConfig(embedding_model="mock-model")
        return MemorySearchService(config=config)

    @patch("memory.search.SessionEmbeddingChunk.objects")
    @patch("memory.search.SessionMessage.objects")
    @patch("memory.search.SessionSummary.objects")
    def test_customer_id_filter_applied_to_chunk_subquery(self, mock_summary_mgr, mock_msg_mgr, mock_chunk_mgr) -> None:
        """text_search filters SessionEmbeddingChunk by customer_id from session."""
        mock_qs = MagicMock()
        mock_qs.values_list.return_value = []
        mock_chunk_mgr.filter.return_value = mock_qs
        # MagicMock auto-handles chained .filter().order_by()[:n] and .filter()[:n]
        mock_msg_mgr.filter.return_value = MagicMock()
        mock_summary_mgr.filter.return_value = MagicMock()

        svc = self._make_service()
        session_a = _make_mock_session(CUSTOMER_A_ID)
        svc.text_search("secret", session=session_a)

        calls = mock_chunk_mgr.filter.call_args_list
        customer_ids_used = [c.kwargs.get("customer_id") for c in calls]
        self.assertIn(
            CUSTOMER_A_ID,
            customer_ids_used,
            "text_search must filter SessionEmbeddingChunk by customer_id from session",
        )

    @patch("memory.search.SessionEmbeddingChunk.objects")
    @patch("memory.search.SessionMessage.objects")
    @patch("memory.search.SessionSummary.objects")
    def test_different_customers_use_different_filters(self, mock_summary_mgr, mock_msg_mgr, mock_chunk_mgr) -> None:
        """Two text_search calls with different sessions produce different customer_id filters."""
        mock_qs = MagicMock()
        mock_qs.values_list.return_value = []
        mock_chunk_mgr.filter.return_value = mock_qs
        # MagicMock auto-handles chained .filter().order_by()[:n] and .filter()[:n]
        mock_msg_mgr.filter.return_value = MagicMock()
        mock_summary_mgr.filter.return_value = MagicMock()

        svc = self._make_service()
        svc.text_search("secret", session=_make_mock_session(CUSTOMER_A_ID))
        calls_a = list(mock_chunk_mgr.filter.call_args_list)

        mock_chunk_mgr.reset_mock()
        svc.text_search("secret", session=_make_mock_session(CUSTOMER_B_ID))
        calls_b = list(mock_chunk_mgr.filter.call_args_list)

        ids_a = {c.kwargs.get("customer_id") for c in calls_a}
        ids_b = {c.kwargs.get("customer_id") for c in calls_b}

        self.assertIn(CUSTOMER_A_ID, ids_a)
        self.assertIn(CUSTOMER_B_ID, ids_b)
        self.assertNotEqual(ids_a, ids_b)
