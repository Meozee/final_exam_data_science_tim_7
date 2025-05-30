# usecase_nada/urls.py
from django.urls import path
from . import views

app_name = 'usecase_nada'

urlpatterns = [
    path('predict-career/', views.predict_career_view, name='predict_career'),
]