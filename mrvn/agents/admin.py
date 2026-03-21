from django.contrib import admin

from .models import Agent, AgentCredential, AgentTool, Tool


class AgentCredentialInline(admin.StackedInline):
    model = AgentCredential
    extra = 0


class AgentToolInline(admin.TabularInline):
    model = AgentTool
    extra = 0


@admin.register(Agent)
class AgentAdmin(admin.ModelAdmin):
    list_display = [
        "name",
        "provider",
        "model_name",
        "rate_limit_display",
        "is_active",
        "created_datetime",
    ]
    list_filter = ["provider", "is_active", "rate_limit_enabled"]
    search_fields = ["name", "description"]
    inlines = [AgentCredentialInline, AgentToolInline]

    fieldsets = (
        (
            None,
            {
                "fields": ("name", "description", "is_active"),
            },
        ),
        (
            "LLM Configuration",
            {
                "fields": ("provider", "model_name", "base_url", "system_prompt"),
            },
        ),
        (
            "Model Parameters",
            {
                "fields": ("temperature", "max_tokens", "config"),
                "classes": ("collapse",),
            },
        ),
        (
            "Tool Access Control",
            {
                "fields": ("tool_profile", "tools_allow", "tools_deny"),
                "description": "Control which tools this agent can use.",
            },
        ),
        (
            "Memory Search",
            {
                "fields": ("memory_search_enabled", "memory_search_config"),
                "description": "Configure agent memory search capability.",
                "classes": ("collapse",),
            },
        ),
        (
            "Rate Limiting",
            {
                "fields": ("rate_limit_enabled", "rate_limit_rpm"),
                "description": "Configure rate limiting to avoid API throttling errors.",
            },
        ),
    )

    def rate_limit_display(self, obj: Agent) -> str:
        if not obj.rate_limit_enabled:
            return "Disabled"
        if obj.rate_limit_rpm == 0:
            return "Unlimited"
        return f"{obj.rate_limit_rpm} rpm"

    rate_limit_display.short_description = "Rate Limit"  # type: ignore[attr-defined]


@admin.register(Tool)
class ToolAdmin(admin.ModelAdmin):
    list_display = ["name", "is_active", "allow_in_groups", "require_approval"]
    list_filter = ["is_active", "allow_in_groups", "require_approval"]
    search_fields = ["name", "description"]
