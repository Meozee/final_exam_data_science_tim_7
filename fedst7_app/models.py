from django.db import models
from django.contrib.auth.models import User

class MLModel(models.Model):
    MODEL_TYPES = [
        ('regression', 'Regression'),
        ('classification', 'Classification'),
        ('clustering', 'Clustering'),
        ('other', 'Other'),
    ]
    
    name = models.CharField(max_length=200)
    creator = models.ForeignKey(
        User, 
        on_delete=models.CASCADE,
        related_name='fedst7_ml_models'  # Added related_name
    )
    created_at = models.DateTimeField(auto_now_add=True)
    model_type = models.CharField(max_length=50, choices=MODEL_TYPES)
    accuracy = models.FloatField(null=True, blank=True)
    model_file = models.FileField(upload_to='ml_models/')
    description = models.TextField()
    use_case = models.TextField()

    def __str__(self):
        return f"{self.name} ({self.model_type})"

    class Meta:
        verbose_name = "ML Model"
        verbose_name_plural = "ML Models"
        ordering = ['-created_at']

class PredictionLog(models.Model):
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE,
        related_name='fedst7_prediction_logs'  # Unique related_name
    )
    input_data = models.JSONField()
    prediction_result = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    execution_time = models.FloatField(help_text="Execution time in seconds")
    
    # Untuk analisis model
    model_used = models.CharField(max_length=100)
    model_version = models.CharField(max_length=50)
    
    class Meta:
        verbose_name = "Prediction Log"
        verbose_name_plural = "Prediction Logs"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Prediction by {self.user.username} at {self.created_at}"