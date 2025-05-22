from django.contrib import admin
from .models import PredictionLog

# Pendekatan 1
@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = ('id', 'attendance_percentage', 'midterm_score', 'project_score', 'difficulty_level', 'predicted_grade', 'created_at')  # ✅ BENAR


# Pendekatan 2 (JANGAN dipakai bersamaan dengan @admin.register)
# admin.site.register(PredictionLog, PredictionLogAdmin)
