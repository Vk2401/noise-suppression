# Generated by Django 3.2.5 on 2023-03-08 10:53

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('vk_nc', '0003_auto_20230301_1509'),
    ]

    operations = [
        migrations.RenameField(
            model_name='tt_audio1',
            old_name='audio_file',
            new_name='afile',
        ),
    ]