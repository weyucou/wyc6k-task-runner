# Copyright (c) 2026 MNKT
"""Tests for marvin.definitions — enum base classes."""

from marvin.definitions import StringEnumWithChoices
from marvin.launchers import ComputeBackend


class Color(StringEnumWithChoices):
    RED = "red"
    DARK_BLUE = "dark-blue"


class TestStringEnumWithChoices:
    def test_str_and_format_give_the_value(self) -> None:
        assert str(Color.RED) == "red"
        assert f"{Color.DARK_BLUE}" == "dark-blue"
        assert str(ComputeBackend.ECS) == "ecs"

    def test_members_are_their_string_values(self) -> None:
        assert Color.RED == "red"
        assert Color("dark-blue") is Color.DARK_BLUE

    def test_choices_and_values(self) -> None:
        assert Color.choices() == (("red", "red"), ("dark-blue", "dark-blue"))
        assert Color.values() == ("red", "dark-blue")
