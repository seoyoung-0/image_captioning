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


# 모델 결과 리턴 화면
def result(request):
    post = Post.objects.last()

    #여기서 모델로 사진 전달 후, caption 받아서 context 에 합치기
    caption='코로 피리부는 소년'

    context = {
        'post':post,
        'caption':caption
    }
    return render(request, 'result.html', {'context':context})

