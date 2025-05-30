from django.db import models

class AtRiskPredictionLog(models.Model):
    RISK_LEVELS = [
        ('LOW', 'Low Risk'),
        ('MEDIUM', 'Medium Risk'),
        ('HIGH', 'High Risk')
    ]
    
    student_id = models.CharField(max_length=20)
    gpa = models.FloatField()
    credits_taken = models.IntegerField()
    attendance_rate = models.FloatField()
    risk_level = models.CharField(max_length=10, choices=RISK_LEVELS)
    probability = models.FloatField()
    prediction_result = models.JSONField()
    prediction_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.student_id} - {self.risk_level}"

class CourseLoadRecommendationLog(models.Model):
    student_id = models.CharField(max_length=20)
    gpa = models.FloatField()
    work_hours_per_week = models.IntegerField()
    recommended_courses = models.JSONField()
    recommendation_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Recommendation for {self.student_id}"