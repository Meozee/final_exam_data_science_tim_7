from django.shortcuts import render

def home(request):
    return render(request, 'fedst7_app/home.html')

def user_dashboard(request):
    return render(request, 'fedst7_app/user_dashboard.html')

def about(request):
    return render(request, 'fedst7_app/about.html')  # Pastikan file about.html ada
