# usecase_miko/urls.py

from django.urls import path
from . import views

app_name = 'usecase_miko'

urlpatterns = [
    path('predict-risk/', views.predict_risk_view, name='predict_risk'),
]