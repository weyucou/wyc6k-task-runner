import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("agents", "0007_project_context"),
        ("memory", "0005_customer"),
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
    ]
