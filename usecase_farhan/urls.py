from django.urls import path
from . import views

app_name = 'usecase_farhan' # Namespace untuk URL

urlpatterns = [
    path('predict-ip/', views.predict_ip_view, name='predict_ip'),
    path('classify-difficulty/', views.classify_course_difficulty_view, name='classify_difficulty'),

]