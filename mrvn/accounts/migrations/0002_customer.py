import commons.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Customer',
            fields=[
                ('id', commons.models.UUID7Field(primary_key=True, serialize=False)),
                ('created_datetime', models.DateTimeField(auto_now_add=True)),
                ('updated_datetime', models.DateTimeField(auto_now=True)),
                ('name', models.CharField(max_length=255)),
                ('github_org', models.CharField(blank=True, max_length=255, null=True, unique=True)),
                ('s3_context_prefix', models.CharField(blank=True, editable=False, max_length=512)),
                ('is_active', models.BooleanField(default=True)),
            ],
            options={
                'abstract': False,
            },
        ),
    ]
