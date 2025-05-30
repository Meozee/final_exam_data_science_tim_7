# /usecase_miko/urls.py
from django.urls import path
from . import views

app_name = 'usecase_miko'

urlpatterns = [
    # URL Anda yang lain
    path('predict-risk/', views.predict_risk_view, name='predict_risk'),
    path('fraud-detection/', views.fraud_detection_view, name='fraud_detection'),
    
    # URL untuk prediksi IP dengan model "ENHANCED"
    path('predict-enhanced-gpa/', views.predict_enhanced_gpa_view, name='predict_enhanced_gpa'),

    # Jika Anda masih ingin mempertahankan view untuk model klasifikasi biner:
    # path('predict-gpa-category/', views.predict_gpa_category_view, name='predict_gpa_category'),
    path('predict-ip-next-semester/', views.predict_ip_view, name='predict_ip_next_semester'),
        path('predict-ip-lecturer-effect/', views.predict_ip_lecturer_effect_view, name='predict_ip_lecturer_effect'),


]