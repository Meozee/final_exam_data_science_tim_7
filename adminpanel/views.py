from django.contrib.auth.decorators import login_required, user_passes_test
from django.shortcuts import render
from fedst7_app.models import MLModel

def is_admin(user):
    return user.is_staff

@user_passes_test(is_admin)
def model_list(request):
    models = MLModel.objects.all()
    return render(request, 'adminpanel/model_list.html', {'models': models})

@user_passes_test(is_admin)
def model_detail(request, pk):
    model = MLModel.objects.get(pk=pk)
    return render(request, 'adminpanel/model_detail.html', {'model': model})

