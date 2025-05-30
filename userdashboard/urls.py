# userdashboard/urls.py

from django.urls import path
from . import views

app_name = 'userdashboard'

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
]