from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('user_dashboard/', views.user_dashboard, name='user_dashboard'),
    path('result/', views.result, name='result'),
]
