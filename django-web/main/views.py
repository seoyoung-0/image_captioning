from django.shortcuts import render,redirect
from .models import Post

#초기화면
def index(request):
    return render(request,'index.html')

#사진 받아서 내부 처리
def create(request):
    post = Post()
    post.image = request.FILES['image']
    post.save()
    return redirect('/result')

# create -> 모델로 사진 전달
# result 에서 캡션이랑 이미지 result.html 로 전달
# 모델 결과 리턴 화면
def result(request):
    post = Post.objects.last()
    caption='코로 피리부는 소년'

    context = {
        'post':post,
        'caption':caption
    }
    return render(request, 'result.html', {'context':context})

