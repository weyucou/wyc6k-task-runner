from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("agents", "0007_project_context"),
    ]

    operations = [
        migrations.AddField(
            model_name="agent",
            name="context_window_messages",
            field=models.PositiveIntegerField(
                default=0,
                help_text="Number of prior session messages to inject as context on resume (0 = disabled)",
            ),
        ),
    ]
