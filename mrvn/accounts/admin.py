from django.contrib import admin
from django.contrib.admin.models import DELETION, LogEntry
from django.contrib.auth.admin import UserAdmin
from django.db.models import QuerySet
from django.http import HttpRequest
from django.urls import NoReverseMatch, reverse
from django.utils.html import escape
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _

from .models import Customer, CustomUser


@admin.register(Customer)
class CustomerAdmin(admin.ModelAdmin):
    list_display = ["name", "github_org", "is_active", "created_datetime"]
    search_fields = ["name", "github_org"]
    list_filter = ["is_active"]


@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    # limit displayed fields
    fieldsets = (
        (None, {"fields": ("username", "password")}),
        (_("Personal info"), {"fields": ("first_name", "last_name", "email")}),
        (_("Permissions"), {"fields": ("is_active", "is_staff", "is_superuser")}),
        (_("Important dates"), {"fields": ("last_login", "date_joined")}),
    )


@admin.register(LogEntry)
class LogEntryAdmin(admin.ModelAdmin):
    date_hierarchy = "action_time"
    readonly_fields = [field.name for field in LogEntry._meta.get_fields()]
    list_filter = ["user", "content_type"]
    search_fields = ["object_repr", "change_message"]
    list_display = ["__str__", "content_type", "action_time", "user", "object_link"]

    def has_add_permission(self, request: HttpRequest) -> bool:
        return False

    def has_change_permission(self, request: HttpRequest, obj: LogEntry = None) -> bool:
        return False

    def has_delete_permission(self, request: HttpRequest, obj: LogEntry = None) -> bool:
        return False

    def has_view_permission(self, request: HttpRequest, obj: LogEntry = None) -> bool:
        # only for superusers, cannot return False, the module
        # wouldn't be visible in admin
        return request.user.is_superuser and request.method != "POST"

    def object_link(self, obj: LogEntry) -> str:
        if obj.action_flag == DELETION:
            link = obj.object_repr
        else:
            ct = obj.content_type
            try:
                model_change_reference = f"admin:{ct.app_label}_{ct.model}_change"
                model_change_url = -reverse(model_change_reference, args=[obj.object_id])
                model_display_name = escape(obj.object_repr)
                link = mark_safe(f'<a href="{model_change_url}">{model_display_name}</a>')  # noqa: S308
            except NoReverseMatch:
                link = obj.object_repr
        return link

    object_link.admin_order_field = "object_repr"
    object_link.short_description = "object"

    def queryset(self, request: HttpRequest) -> QuerySet[LogEntry]:
        return super().queryset(request).prefetch_related("content_type")
