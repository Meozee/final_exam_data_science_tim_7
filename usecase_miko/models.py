# usecase_miko/models.py
from django.db import models
from django.contrib.auth.models import User

class PredictionLog(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='miko_prediction_logs'
    )
    # Input fields (bisa lebih generik atau spesifik per model)
    input_details = models.TextField(help_text="Data input yang digunakan untuk prediksi, bisa dalam format JSON.")
    
    # Output fields
    prediction_type = models.CharField(max_length=100, help_text="Jenis prediksi, cth: Risk Prediction, Anomaly Detection")
    prediction_result = models.TextField(help_text="Hasil utama prediksi, cth: 'Beresiko Gagal' atau 'Anomaly Score: 0.75'")
    prediction_confidence = models.FloatField(null=True, blank=True)
    prediction_reason = models.TextField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Miko's Prediction Log"
        verbose_name_plural = "Miko's Prediction Logs"
        ordering = ['-created_at']

    def __str__(self):
        return f"Log: {self.prediction_type} by {self.user.username} on {self.created_at.strftime('%Y-%m-%d')}"