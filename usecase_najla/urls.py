from django.urls import path
from . import views

app_name = 'usecase_najla'

urlpatterns = [
    path('predict-attendance/', views.predict_attendance_view, name='predict_attendance'),
]