from django.shortcuts import render

def home(request):
    return render(request, 'fedst7_app/home.html')

def about(request):
    return render(request, 'fedst7_app/about.html')
