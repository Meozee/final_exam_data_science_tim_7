from django.db import models

# Model ini bisa diisi jika kamu ingin menyimpan data use case atau log model
class MLModelRegistry(models.Model):
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    file_path = models.CharField(max_length=200)
    creator = models.CharField(max_length=100)
    use_case = models.CharField(max_length=100)
    model_type = models.CharField(max_length=50)  # classification, regression, etc

    def __str__(self):
        return f"{self.name} ({self.model_type})"
