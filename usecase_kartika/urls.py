from django.urls import path
from . import views

app_name = 'usecase_kartika'

urlpatterns = [
    path('assess-risk/', views.assess_risk_view, name='assess_risk'),
]