# fedst7_app/views.py

from django.shortcuts import render

def home_view(request):
    """Menampilkan halaman utama (home page)."""
    return render(request, 'fedst7_app/home.html')

def about_view(request):
    """Menampilkan halaman tentang kami (about page)."""
    return render(request, 'fedst7_app/about.html')