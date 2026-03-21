import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0003_customer_github_token"),
        ("agents", "0006_agent_customer"),
    ]

    operations = [
        migrations.CreateModel(
            name="ProjectContext",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_datetime", models.DateTimeField(auto_now_add=True)),
                ("updated_datetime", models.DateTimeField(auto_now=True)),
                ("project_id", models.CharField(help_text="GitHub project node ID", max_length=255)),
                ("goals_markdown", models.TextField(blank=True, help_text="Project goals (manually set or synced)")),
                ("sops_snapshot", models.JSONField(default=dict, help_text="Cached SOP content from S3")),
                ("s3_prefix", models.CharField(help_text="s3://bucket/customer/projects/<name>/", max_length=512)),
                ("last_synced", models.DateTimeField(blank=True, null=True)),
                (
                    "customer",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="project_contexts",
                        to="accounts.customer",
                    ),
                ),
            ],
            options={
                "unique_together": {("customer", "project_id")},
            },
        ),
    ]
