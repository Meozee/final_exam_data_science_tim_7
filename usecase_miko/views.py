from django import forms

# Untuk Use Case 1 - Prediksi nilai siswa
class PredictForm(forms.Form):
    attendance = forms.FloatField(label='Attendance (%)')
    quiz_score = forms.FloatField(label='Quiz Score')
    midterm_score = forms.FloatField(label='Midterm Score')
    project_score = forms.FloatField(label='Project Score')

# Untuk Use Case 5 - Prediksi performa instruktur
class InstructorForm(forms.Form):
    instructor_id = forms.IntegerField(label='Instructor ID')
    semester = forms.IntegerField(label='Semester')
    num_students = forms.IntegerField(label='Jumlah Mahasiswa')
    avg_midterm = forms.FloatField(label='Rata-rata UTS')
    avg_quiz = forms.FloatField(label='Rata-rata Kuis')
    avg_project = forms.FloatField(label='Rata-rata Proyek')
