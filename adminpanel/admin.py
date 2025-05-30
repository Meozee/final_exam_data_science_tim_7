# adminpanel/admin.py

from django.contrib import admin
from .models import MLModel

@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'creator', 'model_type', 'date_created')
    # Gunakan field yang ada di model MLModel untuk filtering
    list_filter = ('model_type', 'creator', 'date_created') 
    search_fields = ('name', 'description', 'creator')