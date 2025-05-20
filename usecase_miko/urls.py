
# usecase_miko/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_score, name='predict_score'),
]
