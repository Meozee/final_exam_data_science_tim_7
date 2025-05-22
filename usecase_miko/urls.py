from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.PredictForm, name='PredictForm'),
    path('instructor/', views.instructor_analysis, name='instructor_analysis'),
    path('instructor/classify/', views.instructor_classification, name='instructor_classification'),
]
