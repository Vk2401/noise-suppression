from django.db import models

# Create your models here.


class myad(models.Model):
    
    afile = models.FileField(upload_to='vk_nc/static/vk_nc/audio_files')