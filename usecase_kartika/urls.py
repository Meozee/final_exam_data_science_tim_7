from django.urls import path
from . import views

app_name = 'usecase_kartika'

urlpatterns = [
    path('at-risk/', views.at_risk_form, name='at_risk_form'),
    path('at-risk/predict/', views.at_risk_predict, name='at_risk_predict'),
    path('at-risk/result/', views.at_risk_result, name='at_risk_result'),
    path('course-recommendation/', views.course_recommendation_form, name='course_recommendation_form'),
    path('course-recommendation/get/', views.get_recommendations, name='get_recommendations'),
    path('course-recommendation/result/', views.recommendation_result, name='recommendation_result'),
]