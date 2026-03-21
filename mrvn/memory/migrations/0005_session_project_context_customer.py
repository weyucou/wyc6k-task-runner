import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0003_customer_github_token"),
        ("agents", "0007_project_context"),
        ("memory", "0004_improve_hnsw_parameters"),
    ]

    operations = [
        migrations.AddField(
            model_name="session",
            name="project_context",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="sessions",
                to="agents.projectcontext",
            ),
        ),
        migrations.AddField(
            model_name="session",
            name="customer",
            field=models.ForeignKey(
                blank=True,
                db_index=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="sessions",
                to="accounts.customer",
            ),
        ),
    ]
