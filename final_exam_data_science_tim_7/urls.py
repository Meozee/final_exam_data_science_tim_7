"""
URL configuration for final_exam_data_science_tim_7 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', include('fedst7_app.urls')),
    path('admin/', admin.site.urls),
    path('miko/', include('usecase_miko.urls')),
    path('kartika/', include('usecase_kartika.urls')),
    path('farhan/', include('usecase_farhan.urls')),
    path('nada/', include('usecase_nada.urls')),
    path('najla/', include('usecase_najla.urls')),
]

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])
