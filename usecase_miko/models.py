from django.db import models

class PredictionLog(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    attendance = models.FloatField()
    quiz_score = models.FloatField()
    midterm_score = models.FloatField()
    project_score = models.FloatField()
    predicted_grade = models.FloatField()
