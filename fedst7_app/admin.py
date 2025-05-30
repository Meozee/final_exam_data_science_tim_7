from django.contrib import admin
from .models import MLModel, PredictionLog

@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'user',
        'get_model_info',
        'get_execution_time',
        'get_created_at',
    )
    
    list_filter = ('model_used', 'created_at')
    search_fields = ('user__username', 'model_used')
    readonly_fields = ('created_at', 'input_data', 'prediction_result')
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('user', 'model_used', 'model_version')
        }),
        ('Prediction Data', {
            'fields': ('input_data', 'prediction_result'),
            'classes': ('collapse',)
        }),
        ('Performance Metrics', {
            'fields': ('execution_time', 'created_at')
        }),
    )

    def get_model_info(self, obj):
        return f"{obj.model_used} (v{obj.model_version})"
    get_model_info.short_description = 'Model'
    
    def get_execution_time(self, obj):
        return f"{obj.execution_time:.2f}s"
    get_execution_time.short_description = 'Time'
    
    def get_created_at(self, obj):
        return obj.created_at.strftime("%Y-%m-%d %H:%M")
    get_created_at.short_description = 'Created At'

@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = (
        'name', 
        'creator',
        'model_type',
        'accuracy_score',
        'created_at'
    )
    list_filter = ('model_type', 'creator')
    search_fields = ('name', 'description')
    readonly_fields = ('created_at',)
    
    fieldsets = (
        ('Model Information', {
            'fields': ('name', 'creator', 'model_type')
        }),
        ('Model Details', {
            'fields': ('description', 'use_case', 'model_file')
        }),
        ('Performance', {
            'fields': ('accuracy', 'created_at')
        }),
    )

    def accuracy_score(self, obj):
        return f"{obj.accuracy:.2%}" if obj.accuracy else "N/A"
    accuracy_score.short_description = 'Accuracy'