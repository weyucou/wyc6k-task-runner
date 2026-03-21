from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0002_customer"),
    ]

    operations = [
        migrations.AddField(
            model_name="customer",
            name="github_token",
            field=models.CharField(blank=True, help_text="GitHub API token for this organization", max_length=512),
        ),
    ]
