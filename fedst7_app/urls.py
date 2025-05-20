from django.urls import path, include

urlpatterns = [
    path('', include('main_app.urls')),
    path('miko/', include('usecase_miko.urls')),  # 🔗 Tambahkan ini
]
