# Generated by Django 3.2.5 on 2023-03-08 11:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('vk_nc', '0004_rename_audio_file_tt_audio1_afile'),
    ]

    operations = [
        migrations.CreateModel(
            name='myad',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('afile', models.FileField(upload_to='vk_nc/static/vk_nc/audio_files')),
            ],
        ),
        migrations.DeleteModel(
            name='tt_audio1',
        ),
    ]