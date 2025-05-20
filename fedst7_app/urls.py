from django.urls import path, include

urlpatterns = [
    path('', include('fedst7_app.urls')),
    path('miko/', include('usecase_miko.urls')),  # 🔗 Tambahkan ini
]
