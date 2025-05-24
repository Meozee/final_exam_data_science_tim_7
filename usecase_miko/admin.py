from django.contrib import admin
from .models import PredictionLog

@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'user',
        'formatted_attendance',
        'midterm_score',
        'get_difficulty_level',
        'predicted_grade',
        'formatted_created_at',
        'actual_grade'
    )
    
    list_filter = ('difficulty_level', 'created_at')
    search_fields = ('user__username', 'difficulty_level')
    
    def formatted_attendance(self, obj):
        return f"{obj.attendance_percentage}%"
    formatted_attendance.short_description = 'Attendance'
    
    def get_difficulty_level(self, obj):
        return obj.difficulty_level
    get_difficulty_level.short_description = 'Difficulty'
    
    def formatted_created_at(self, obj):
        return obj.created_at.strftime("%Y-%m-%d %H:%M")
    formatted_created_at.short_description = 'Created At'