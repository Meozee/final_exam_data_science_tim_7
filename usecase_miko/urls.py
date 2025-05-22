from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_student_performance, name='predict_student_performance'),
]