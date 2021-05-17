from django.db import models
from imagekit import processors
from imagekit.models import ProcessedImageField
from imagekit.processors import ResizeToFill
from pilkit.processors.resize import Resize



class Post(models.Model):

    name = models.CharField(max_length=200,blank=True, null=True)
    # mainImage = models.ImageField(blank=False, null=True,upload_to='uploaded_photo/%Y/%m/%d')
    image = ProcessedImageField(
        upload_to='uploaded_photo/',
        #processors=[ResizeToFill(600,600)],
        format='JPEG',
    )
    created = models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return str(self.image)
