# userdashboard/views.py

from django.shortcuts import render
from adminpanel.models import MLModel

def dashboard_view(request):
    """Menampilkan daftar semua use case/model yang tersedia."""
    models = MLModel.objects.all()
    context = {'models': models}
    return render(request, 'userdashboard/user_dashboard.html', context)