# adminpanel/models.py
from django.db import models # <-- Pastikan baris ini benar
from django.utils import timezone

class MLModel(models.Model): # <-- Pastikan baris ini benar, menggunakan 'models.Model'
    MODEL_TYPES = [
        ('Classification', 'Classification'),
        ('Regression', 'Regression'),
        ('Clustering', 'Clustering'),
    ]

    name = models.CharField(max_length=200, help_text="Nama model, cth: Student Risk Predictor")
    description = models.TextField(help_text="Deskripsi singkat mengenai kasus penggunaan dan tujuan model.")
    model_type = models.CharField(max_length=50, choices=MODEL_TYPES, help_text="Jenis model machine learning.")
    creator = models.CharField(max_length=100, help_text="Nama anggota tim yang membuat model.")
    date_created = models.DateTimeField(default=timezone.now, help_text="Tanggal model dibuat.")
    dataset_name = models.CharField(max_length=100, help_text="Nama dataset yang digunakan untuk training, cth: Student Performance Data 2024")
    file_path = models.CharField(max_length=255, help_text="Path relatif ke file .pkl model utama, cth: ml_models/miko_student_risk_pipeline.pkl")
    scaler_path = models.CharField(max_length=255, blank=True, null=True, help_text="Path ke file scaler .pkl (jika ada, untuk model seperti Isolation Forest).")
    explainer_path = models.CharField(max_length=255, blank=True, null=True, help_text="Path ke file explainer .pkl (jika ada, untuk model seperti SHAP).")
    features_path = models.CharField(max_length=255, blank=True, null=True, help_text="Path ke file nama fitur .pkl (jika ada).")
    endpoint_url_name = models.CharField(max_length=100, help_text="Nama URL pattern dari halaman prediksi, cth: usecase_miko:predict_risk")

    def __str__(self):
        return f"{self.name} by {self.creator}"

    class Meta:
        ordering = ['-date_created']