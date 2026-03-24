"""Tests for marvin_manager tool validation and execution."""

import asyncio

import pytest

from marvin_manager.tools.builtin import (
    BrowserTool,
    CalculatorTool,
    DateTimeTool,
    MemorySearchTool,
    MemoryStoreTool,
    WebSearchTool,
)
from marvin_manager.tools.registry import ToolRegistry


# ---- CalculatorTool ----


class TestCalculatorToolValidation:
    def setup_method(self) -> None:
        self.tool = CalculatorTool()

    def test_missing_required_parameter(self) -> None:
        is_valid, error = self.tool.validate_params({})
        assert not is_valid
        assert "expression" in error

    def test_wrong_type_number_for_string(self) -> None:
        is_valid, error = self.tool.validate_params({"expression": 123})
        assert not is_valid
        assert "string" in error.lower()

    def test_valid_string_parameter(self) -> None:
        is_valid, error = self.tool.validate_params({"expression": "1 + 1"})
        assert is_valid
        assert error is None

    def test_extra_parameters_accepted(self) -> None:
        is_valid, error = self.tool.validate_params({"expression": "1 + 1", "unknown_param": "value"})
        assert is_valid
        assert error is None

    def test_execute_valid_expression(self) -> None:
        result = asyncio.run(self.tool.execute(expression="2 + 3"))
        assert result.output == "5"

    def test_execute_invalid_expression(self) -> None:
        result = asyncio.run(self.tool.execute(expression="1 / 0"))
        assert result.error is not None

    def test_execute_disallowed_chars(self) -> None:
        result = asyncio.run(self.tool.execute(expression="__import__('os')"))
        assert result.error is not None


# ---- DateTimeTool ----


class TestDateTimeToolValidation:
    def setup_method(self) -> None:
        self.tool = DateTimeTool()

    def test_no_parameters_valid(self) -> None:
        is_valid, error = self.tool.validate_params({})
        assert is_valid
        assert error is None

    def test_valid_timezone_parameter(self) -> None:
        is_valid, error = self.tool.validate_params({"timezone": "America/New_York"})
        assert is_valid
        assert error is None

    def test_valid_output_format_iso(self) -> None:
        is_valid, error = self.tool.validate_params({"output_format": "iso"})
        assert is_valid

    def test_valid_output_format_human(self) -> None:
        is_valid, error = self.tool.validate_params({"output_format": "human"})
        assert is_valid

    def test_invalid_output_format_enum(self) -> None:
        is_valid, error = self.tool.validate_params({"output_format": "invalid"})
        assert not is_valid
        assert "must be one of" in error.lower()

    def test_wrong_type_for_timezone(self) -> None:
        is_valid, error = self.tool.validate_params({"timezone": 123})
        assert not is_valid
        assert "string" in error.lower()


# ---- WebSearchTool ----


class TestWebSearchToolValidation:
    def setup_method(self) -> None:
        self.tool = WebSearchTool()

    def test_missing_required_query(self) -> None:
        is_valid, error = self.tool.validate_params({})
        assert not is_valid
        assert "query" in error

    def test_valid_query_only(self) -> None:
        is_valid, error = self.tool.validate_params({"query": "test search"})
        assert is_valid
        assert error is None

    def test_wrong_type_for_num_results(self) -> None:
        is_valid, error = self.tool.validate_params({"query": "test", "num_results": "five"})
        assert not is_valid
        assert "number" in error.lower()


# ---- BrowserTool ----


class TestBrowserToolValidation:
    def setup_method(self) -> None:
        self.tool = BrowserTool()

    def test_missing_required_url(self) -> None:
        is_valid, error = self.tool.validate_params({})
        assert not is_valid
        assert "url" in error

    def test_valid_url_parameter(self) -> None:
        is_valid, error = self.tool.validate_params({"url": "http://example.com"})
        assert is_valid
        assert error is None


# ---- MemorySearchTool (stateless) ----


class TestMemorySearchToolStateless:
    def test_returns_empty_without_session(self) -> None:
        tool = MemorySearchTool()
        result = asyncio.run(tool.execute(query="test"))
        assert result.output == "Memory search not available in stateless mode."
        assert result.data["count"] == 0

    def test_returns_empty_with_session_id(self) -> None:
        tool = MemorySearchTool(session_id="some-session")
        result = asyncio.run(tool.execute(query="test"))
        assert result.output == "Memory search not available in stateless mode."
        assert result.data["results"] == []


# ---- ToolSchemaGeneration ----


class TestToolSchemaGeneration:
    def test_calculator_schema_has_required_fields(self) -> None:
        tool = CalculatorTool()
        schema = tool.get_schema()
        assert schema["type"] == "object"
        assert "expression" in schema["properties"]
        assert "expression" in schema["required"]

    def test_datetime_schema_has_enum(self) -> None:
        tool = DateTimeTool()
        schema = tool.get_schema()
        assert "output_format" in schema["properties"]
        assert "enum" in schema["properties"]["output_format"]
        assert schema["properties"]["output_format"]["enum"] == ["iso", "human"]

    def test_datetime_schema_no_required_fields(self) -> None:
        tool = DateTimeTool()
        schema = tool.get_schema()
        assert len(schema.get("required", [])) == 0

    def test_web_search_schema_has_default(self) -> None:
        tool = WebSearchTool()
        schema = tool.get_schema()
        assert "num_results" in schema["properties"]
        assert schema["properties"]["num_results"]["default"] == 5


# ---- ToolRegistry ----


class TestToolRegistry:
    def test_register_and_list(self) -> None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        assert "calculator" in registry.list_tools()

    def test_duplicate_registration_raises(self) -> None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(CalculatorTool())

    def test_unregister(self) -> None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.unregister("calculator")
        assert "calculator" not in registry.list_tools()

    def test_execute_unknown_tool_returns_error(self) -> None:
        registry = ToolRegistry()
        result = asyncio.run(registry.execute("no_such_tool", {}))
        assert result.error is not None
        assert "not found" in result.error

    def test_to_anthropic_tools_format(self) -> None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        tools = registry.to_anthropic_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "calculator"
        assert "input_schema" in tools[0]

    def test_to_openai_tools_format(self) -> None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        tools = registry.to_openai_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "calculator"
