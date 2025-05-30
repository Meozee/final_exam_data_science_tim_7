# usecase_xxx/views.py
from django.http import HttpResponse

def placeholder(request):
    return HttpResponse("This is a placeholder for {{ nama anggota }}'s app.")
