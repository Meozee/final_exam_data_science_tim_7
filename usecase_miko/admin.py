# usecase_miko/admin.py

from django.contrib import admin
from .models import PredictionLog

# usecase_miko/admin.py
from django.contrib import admin
from .models import PredictionLog

@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = ('prediction_type', 'prediction_result', 'prediction_confidence', 'created_at', 'user')
    # Hapus 'difficulty_level' karena sudah tidak ada sebagai field langsung
    list_filter = ('prediction_type', 'created_at', 'user') # Ganti dengan field yang valid
    search_fields = ('prediction_reason', 'user__username', 'input_details')
    readonly_fields = ('created_at', 'input_details', 'prediction_result', 'prediction_confidence', 'prediction_reason')
    
    fieldsets = (
        (None, {
            'fields': ('user', 'prediction_type', 'created_at')
        }),
        ('Prediction Details', {
            'fields': ('input_details', 'prediction_result', 'prediction_confidence', 'prediction_reason')
        }),
    )
    
    # Fungsi-fungsi kustom Anda sudah benar, tidak perlu diubah.
    def formatted_attendance(self, obj):
        return f"{obj.attendance_percentage}%"
    formatted_attendance.short_description = 'Attendance'
    
    def get_difficulty_level(self, obj):
        return obj.difficulty_level
    get_difficulty_level.short_description = 'Difficulty'
    
    def formatted_created_at(self, obj):
        return obj.created_at.strftime("%Y-%m-%d %H:%M")
    formatted_created_at.short_description = 'Created At'