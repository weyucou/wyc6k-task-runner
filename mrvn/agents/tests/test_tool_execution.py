"""Tests for tool execution with CalculatorTool and DateTimeTool."""

from datetime import datetime
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch

from agents.tools.base import ToolStatus
from agents.tools.builtin import BrowserTool, CalculatorTool, DateTimeTool


class CalculatorToolTests(IsolatedAsyncioTestCase):
    """Tests for CalculatorTool execution."""

    def setUp(self) -> None:
        """Set up calculator tool."""
        self.tool = CalculatorTool()

    async def test_basic_addition(self) -> None:
        """Test basic addition."""
        result = await self.tool.execute(expression="2 + 2")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.output, "4")

    async def test_basic_subtraction(self) -> None:
        """Test basic subtraction."""
        result = await self.tool.execute(expression="10 - 3")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.output, "7")

    async def test_basic_multiplication(self) -> None:
        """Test basic multiplication."""
        result = await self.tool.execute(expression="6 * 7")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.output, "42")

    async def test_basic_division(self) -> None:
        """Test basic division."""
        result = await self.tool.execute(expression="20 / 4")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.output, "5.0")

    async def test_complex_expression(self) -> None:
        """Test complex expression with parentheses."""
        result = await self.tool.execute(expression="(10 * 5) / 2 + 3")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.output, "28.0")

    async def test_floating_point(self) -> None:
        """Test floating point numbers."""
        result = await self.tool.execute(expression="3.14 * 2")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertAlmostEqual(float(result.output), 6.28, places=2)

    async def test_negative_numbers(self) -> None:
        """Test negative numbers."""
        result = await self.tool.execute(expression="-5 + 10")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.output, "5")

    async def test_exponentiation(self) -> None:
        """Test exponentiation with **."""
        result = await self.tool.execute(expression="2 ** 10")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.output, "1024")

    async def test_modulo_not_allowed(self) -> None:
        """Test modulo operation is not in allowed characters."""
        # % is not in ALLOWED_CHARS for security
        result = await self.tool.execute(expression="17 % 5")
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("invalid", result.error.lower())

    async def test_division_by_zero_error(self) -> None:
        """Test division by zero returns error."""
        result = await self.tool.execute(expression="1 / 0")
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("error", result.error.lower())

    async def test_invalid_characters_rejected(self) -> None:
        """Test that dangerous characters are rejected."""
        result = await self.tool.execute(expression="import os")
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("invalid", result.error.lower())

    async def test_function_calls_rejected(self) -> None:
        """Test that function calls are rejected (letters not allowed)."""
        result = await self.tool.execute(expression="eval('1+1')")
        self.assertEqual(result.status, ToolStatus.ERROR)

    async def test_dunder_rejected(self) -> None:
        """Test that dunder access is rejected (underscores not allowed)."""
        result = await self.tool.execute(expression="__import__('os')")
        self.assertEqual(result.status, ToolStatus.ERROR)

    async def test_syntax_error(self) -> None:
        """Test syntax error handling."""
        result = await self.tool.execute(expression="1 +")
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("error", result.error.lower())

    async def test_result_data_contains_expression(self) -> None:
        """Test that successful result data contains expression and result."""
        result = await self.tool.execute(expression="5 + 5")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertIn("expression", result.data)
        self.assertIn("result", result.data)
        self.assertEqual(result.data["expression"], "5 + 5")
        self.assertEqual(result.data["result"], 10)


class DateTimeToolTests(IsolatedAsyncioTestCase):
    """Tests for DateTimeTool execution."""

    def setUp(self) -> None:
        """Set up datetime tool."""
        self.tool = DateTimeTool()

    async def test_default_iso_format(self) -> None:
        """Test default output is ISO format."""
        result = await self.tool.execute()
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        # ISO format should match pattern like 2024-01-15T10:30:00+00:00
        iso_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        self.assertRegex(result.output, iso_pattern)

    async def test_iso_format_explicit(self) -> None:
        """Test explicit ISO format."""
        result = await self.tool.execute(output_format="iso")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        iso_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        self.assertRegex(result.output, iso_pattern)

    async def test_human_format(self) -> None:
        """Test human-readable format."""
        result = await self.tool.execute(output_format="human")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        # Human format should contain readable elements
        # Expected format like: "Monday, January 15, 2024 at 10:30:00 AM"
        self.assertIn(",", result.output)  # Contains commas in date formatting

    async def test_timezone_utc(self) -> None:
        """Test UTC timezone."""
        result = await self.tool.execute(timezone="UTC")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        # Should contain UTC or +00:00
        self.assertTrue(
            "UTC" in result.output or "+00:00" in result.output or "Z" in result.output,
            f"Expected UTC indicator in: {result.output}",
        )

    async def test_timezone_new_york(self) -> None:
        """Test America/New_York timezone."""
        result = await self.tool.execute(timezone="America/New_York")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        # Should be a valid datetime string
        self.assertIsNotNone(result.output)

    async def test_timezone_tokyo(self) -> None:
        """Test Asia/Tokyo timezone."""
        result = await self.tool.execute(timezone="Asia/Tokyo")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertIsNotNone(result.output)

    async def test_invalid_timezone_fallback_to_utc(self) -> None:
        """Test invalid timezone falls back to UTC (graceful degradation)."""
        result = await self.tool.execute(timezone="Invalid/Timezone")
        # Implementation falls back to UTC instead of erroring
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertIn("UTC", result.data["timezone"])

    async def test_result_data_contains_components(self) -> None:
        """Test that result data contains datetime components."""
        result = await self.tool.execute()
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertIsNotNone(result.data)

        # Data should contain datetime components
        data = result.data
        self.assertIn("year", data)
        self.assertIn("month", data)
        self.assertIn("day", data)
        self.assertIn("hour", data)
        self.assertIn("minute", data)
        self.assertIn("timestamp", data)
        self.assertIn("timezone", data)

    async def test_result_year_is_current(self) -> None:
        """Test that year in result is current year."""
        result = await self.tool.execute()
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        current_year = datetime.now().year
        self.assertEqual(result.data["year"], current_year)


class BrowserToolTests(IsolatedAsyncioTestCase):
    """Tests for BrowserTool execution."""

    def setUp(self) -> None:
        """Set up browser tool."""
        self.tool = BrowserTool()

    def _make_mock_response(self, content: str) -> MagicMock:
        """Build a context-manager mock for urlopen."""
        mock_response = MagicMock()
        mock_response.read.return_value = content.encode("utf-8")
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)
        return mock_response

    async def test_successful_fetch(self) -> None:
        """Test successful URL fetch returns page content."""
        html = "<html><body>Hello</body></html>"
        with patch("urllib.request.urlopen", return_value=self._make_mock_response(html)):
            result = await self.tool.execute(url="http://example.com")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(result.output, html)
        self.assertEqual(result.data["url"], "http://example.com")
        self.assertEqual(result.data["length"], len(html))

    async def test_fetch_error_returns_error_result(self) -> None:
        """Test that a network error returns an ERROR result."""
        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            result = await self.tool.execute(url="http://bad-url.example")
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn("Failed to fetch URL", result.error)

    async def test_output_truncated_to_4000_chars(self) -> None:
        """Test that output is truncated to 4000 characters for large pages."""
        long_content = "x" * 5000
        with patch("urllib.request.urlopen", return_value=self._make_mock_response(long_content)):
            result = await self.tool.execute(url="http://example.com")
        self.assertEqual(result.status, ToolStatus.SUCCESS)
        self.assertEqual(len(result.output), 4000)
        self.assertEqual(result.data["length"], 5000)
