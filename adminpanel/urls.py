from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('login/', auth_views.LoginView.as_view(template_name='adminpanel/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('models/', views.model_list, name='model_list'),
    path('models/<int:pk>/', views.model_detail, name='model_detail'),
]