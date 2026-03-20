from django.conf import settings
from django.db import migrations


class Migration(migrations.Migration):
    """Remove owner FK (and its unique_together constraint) from Agent.

    The customer FK added in this branch is not being added to Agent.
    """

    dependencies = [
        ('accounts', '0002_customer'),
        ('agents', '0005_improve_hnsw_parameters'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name='agent',
            unique_together=set(),
        ),
        migrations.RemoveField(
            model_name='agent',
            name='owner',
        ),
    ]
