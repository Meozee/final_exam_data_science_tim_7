# adminpanel/admin.py

from django.contrib import admin
from .models import MLModel

@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'creator', 'model_type', 'date_created')
    list_filter = ('model_type', 'creator')
    search_fields = ('name', 'description')