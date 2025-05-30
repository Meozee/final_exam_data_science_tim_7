from django.db import models
# Impor model lain jika diperlukan untuk ForeignKey, sesuaikan pathnya
# from student.models import Student
# from fedst7_app.models import Course, Semester

class PredictionRecord(models.Model):
    # 'name' bisa jadi CharField untuk input nama, atau IntegerField jika itu stu_id
    name_or_stu_id = models.CharField(max_length=255, help_text="Nama atau ID Mahasiswa dari input form")
    average_score = models.FloatField(null=True, blank=True)
    grade = models.FloatField(null=True, blank=True)
    
    # Simpan ID, bukan instance ForeignKey, karena input form mungkin hanya ID
    # Atau jika form menggunakan ModelChoiceField, Anda bisa simpan ForeignKey
    # Untuk konsistensi dengan input yang mungkin berupa ID, kita simpan ID.
    course_id_input = models.IntegerField(help_text="ID Mata Kuliah dari input form")
    semester_id_input = models.IntegerField(help_text="ID Semester dari input form")
    
    predicted_attendance = models.FloatField()
    prediction_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediksi untuk {self.name_or_stu_id} - Kehadiran: {self.predicted_attendance:.1f}%"

    class Meta:
        app_label = 'usecase_najla' # Pastikan ini benar