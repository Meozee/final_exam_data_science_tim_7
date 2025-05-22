# fedst7_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),           # Homepage
    path('about/', views.about, name='about'),   # Tentang tim
]
