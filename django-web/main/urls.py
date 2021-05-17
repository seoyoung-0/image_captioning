from django.urls import path
from .views import *

from django.conf.urls.static import static
from django.conf import settings

urlpatterns =[
    path('',index,name='index'),
    path('create',create,name='create'),
    path('result',result,name='result'),
]

# 이미지 URL 설정
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)