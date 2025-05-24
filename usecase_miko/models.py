from django.db import models
from django.contrib.auth.models import User

class PredictionLog(models.Model):
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE,
        related_name='miko_prediction_logs'  # Unique related_name
    )
    attendance_percentage = models.FloatField(verbose_name="Attendance Percentage")
    midterm_score = models.FloatField()
    difficulty_level = models.CharField(
        max_length=10,
        choices=[('Easy', 'Easy'), ('Medium', 'Medium'), ('Hard', 'Hard')]
    )
    predicted_grade = models.FloatField()
    actual_grade = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Miko's Prediction Log"
        verbose_name_plural = "Miko's Prediction Logs"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Prediction by {self.user.username} (Miko)"