from django.contrib import admin

from tasks.models import TaskEnvelope


@admin.register(TaskEnvelope)
class TaskEnvelopeAdmin(admin.ModelAdmin):
    list_display = ["id", "customer", "action", "status", "issue_url", "created_datetime"]
    list_filter = ["status", "action"]
    search_fields = ["issue_url", "sqs_message_id"]
    readonly_fields = ["id", "sqs_message_id", "created_datetime", "updated_datetime"]
