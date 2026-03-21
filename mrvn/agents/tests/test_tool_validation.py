"""Tests for tool parameter validation."""

from django.test import TestCase

from agents.tools.builtin import BrowserTool, CalculatorTool, DateTimeTool, WebSearchTool


class CalculatorToolValidationTests(TestCase):
    """Tests for CalculatorTool parameter validation."""

    def setUp(self) -> None:
        """Set up calculator tool."""
        self.tool = CalculatorTool()

    def test_missing_required_parameter(self) -> None:
        """Test that missing required parameter is rejected."""
        is_valid, error = self.tool.validate_params({})
        self.assertFalse(is_valid)
        self.assertIn("expression", error)

    def test_wrong_type_number_for_string(self) -> None:
        """Test that wrong type (number instead of string) is rejected."""
        is_valid, error = self.tool.validate_params({"expression": 123})
        self.assertFalse(is_valid)
        self.assertIn("string", error.lower())

    def test_wrong_type_list_for_string(self) -> None:
        """Test that wrong type (list instead of string) is rejected."""
        is_valid, error = self.tool.validate_params({"expression": ["1", "+", "1"]})
        self.assertFalse(is_valid)
        self.assertIn("string", error.lower())

    def test_wrong_type_dict_for_string(self) -> None:
        """Test that wrong type (dict instead of string) is rejected."""
        is_valid, error = self.tool.validate_params({"expression": {"value": "1+1"}})
        self.assertFalse(is_valid)
        self.assertIn("string", error.lower())

    def test_valid_string_parameter(self) -> None:
        """Test that valid string parameter is accepted."""
        is_valid, error = self.tool.validate_params({"expression": "1 + 1"})
        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_extra_parameters_accepted(self) -> None:
        """Test that extra parameters don't cause validation failure."""
        is_valid, error = self.tool.validate_params(
            {
                "expression": "1 + 1",
                "unknown_param": "value",
            }
        )
        self.assertTrue(is_valid)
        self.assertIsNone(error)


class DateTimeToolValidationTests(TestCase):
    """Tests for DateTimeTool parameter validation."""

    def setUp(self) -> None:
        """Set up datetime tool."""
        self.tool = DateTimeTool()

    def test_no_parameters_valid(self) -> None:
        """Test that no parameters is valid (all optional)."""
        is_valid, error = self.tool.validate_params({})
        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_valid_timezone_parameter(self) -> None:
        """Test valid timezone parameter."""
        is_valid, error = self.tool.validate_params({"timezone": "America/New_York"})
        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_valid_output_format_iso(self) -> None:
        """Test valid output_format 'iso'."""
        is_valid, error = self.tool.validate_params({"output_format": "iso"})
        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_valid_output_format_human(self) -> None:
        """Test valid output_format 'human'."""
        is_valid, error = self.tool.validate_params({"output_format": "human"})
        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_invalid_output_format_enum(self) -> None:
        """Test that invalid enum value is rejected."""
        is_valid, error = self.tool.validate_params({"output_format": "invalid"})
        self.assertFalse(is_valid)
        self.assertIn("must be one of", error.lower())

    def test_wrong_type_for_timezone(self) -> None:
        """Test that wrong type for timezone is rejected."""
        is_valid, error = self.tool.validate_params({"timezone": 123})
        self.assertFalse(is_valid)
        self.assertIn("string", error.lower())

    def test_both_parameters_valid(self) -> None:
        """Test both parameters together."""
        is_valid, error = self.tool.validate_params(
            {
                "timezone": "UTC",
                "output_format": "iso",
            }
        )
        self.assertTrue(is_valid)
        self.assertIsNone(error)


class WebSearchToolValidationTests(TestCase):
    """Tests for WebSearchTool parameter validation."""

    def setUp(self) -> None:
        """Set up web search tool."""
        self.tool = WebSearchTool()

    def test_missing_required_query(self) -> None:
        """Test that missing query parameter is rejected."""
        is_valid, error = self.tool.validate_params({})
        self.assertFalse(is_valid)
        self.assertIn("query", error)

    def test_valid_query_only(self) -> None:
        """Test valid query parameter only."""
        is_valid, error = self.tool.validate_params({"query": "test search"})
        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_valid_query_with_num_results(self) -> None:
        """Test valid query with num_results."""
        is_valid, error = self.tool.validate_params(
            {
                "query": "test search",
                "num_results": 5,
            }
        )
        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_wrong_type_for_num_results(self) -> None:
        """Test that wrong type for num_results is rejected."""
        is_valid, error = self.tool.validate_params(
            {
                "query": "test search",
                "num_results": "five",
            }
        )
        self.assertFalse(is_valid)
        self.assertIn("number", error.lower())

    def test_wrong_type_for_query(self) -> None:
        """Test that wrong type for query is rejected."""
        is_valid, error = self.tool.validate_params({"query": 123})
        self.assertFalse(is_valid)
        self.assertIn("string", error.lower())


class BrowserToolValidationTests(TestCase):
    """Tests for BrowserTool parameter validation."""

    def setUp(self) -> None:
        """Set up browser tool."""
        self.tool = BrowserTool()

    def test_missing_required_url(self) -> None:
        """Test that missing url parameter is rejected."""
        is_valid, error = self.tool.validate_params({})
        self.assertFalse(is_valid)
        self.assertIn("url", error)

    def test_wrong_type_for_url(self) -> None:
        """Test that wrong type for url is rejected."""
        is_valid, error = self.tool.validate_params({"url": 123})
        self.assertFalse(is_valid)
        self.assertIn("string", error.lower())

    def test_valid_url_parameter(self) -> None:
        """Test that valid url string is accepted."""
        is_valid, error = self.tool.validate_params({"url": "http://example.com"})
        self.assertTrue(is_valid)
        self.assertIsNone(error)


class ToolSchemaGenerationTests(TestCase):
    """Tests for tool JSON schema generation."""

    def test_calculator_schema_has_required_fields(self) -> None:
        """Test calculator schema has required expression field."""
        tool = CalculatorTool()
        schema = tool.get_schema()

        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        self.assertIn("expression", schema["properties"])
        self.assertIn("required", schema)
        self.assertIn("expression", schema["required"])

    def test_datetime_schema_has_enum(self) -> None:
        """Test datetime schema has enum for output_format."""
        tool = DateTimeTool()
        schema = tool.get_schema()

        self.assertIn("properties", schema)
        self.assertIn("output_format", schema["properties"])
        self.assertIn("enum", schema["properties"]["output_format"])
        self.assertEqual(schema["properties"]["output_format"]["enum"], ["iso", "human"])

    def test_datetime_schema_no_required_fields(self) -> None:
        """Test datetime schema has no required fields (all optional)."""
        tool = DateTimeTool()
        schema = tool.get_schema()

        # required should be empty or not present
        required = schema.get("required", [])
        self.assertEqual(len(required), 0)

    def test_web_search_schema_has_default(self) -> None:
        """Test web search schema has default value for num_results."""
        tool = WebSearchTool()
        schema = tool.get_schema()

        self.assertIn("properties", schema)
        self.assertIn("num_results", schema["properties"])
        self.assertIn("default", schema["properties"]["num_results"])
        self.assertEqual(schema["properties"]["num_results"]["default"], 5)
