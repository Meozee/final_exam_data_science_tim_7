# fedst7_app/urls.py

from django.urls import path
from . import views  # Impor seluruh file views.py

app_name = 'fedst7_app'

urlpatterns = [
    # Panggil fungsi home_view dari dalam views
    path('', views.home_view, name='home'),
    
    # Panggil fungsi about_view dari dalam views
    path('about/', views.about_view, name='about'),
]