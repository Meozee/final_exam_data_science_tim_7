# usecase_miko/forms.py

from django import forms
from django.core.validators import MinValueValidator, MaxValueValidator

# Pilihan ini bisa diambil dinamis dari database, namun untuk penyederhanaan
# kita bisa definisikan secara statis terlebih dahulu berdasarkan data yang ada.
# Berdasarkan ringkasan.txt [cite: 98]
DEPT_CHOICES = [
    ('Information Technology', 'Information Technology'),
    ('Information Systems', 'Information Systems'),
]

# Berdasarkan ringkasan.txt [cite: 91]
COURSE_CHOICES = [
    ('Course 1', 'Course 1'),
    ('Course 2', 'Course 2'),
    # Tambahkan nama course lainnya jika ada
    ('Course 20', 'Course 20'),
]

# Berdasarkan ringkasan.txt [cite: 93]
DIFFICULTY_CHOICES = [
    ('Easy', 'Easy'),
    ('Medium', 'Medium'),
    ('Hard', 'Hard'),
]

GENDER_CHOICES = [
    ('Female', 'Female'),
    ('Male', 'Male'),
]

class RiskPredictionForm(forms.Form):
    # Fitur dari tabel 'student' dan 'department'
    gender = forms.ChoiceField(choices=GENDER_CHOICES, label="Jenis Kelamin")
    dept_name = forms.ChoiceField(choices=DEPT_CHOICES, label="Nama Departemen")

    # Fitur dari tabel 'course' dan 'course_difficulty'
    course_name = forms.ChoiceField(choices=COURSE_CHOICES, label="Nama Mata Kuliah")
    difficulty_level = forms.ChoiceField(choices=DIFFICULTY_CHOICES, label="Tingkat Kesulitan")

    # Fitur dari tabel 'attendance' dan 'assessment'
    attendance_percentage = forms.IntegerField(
        label="Persentase Kehadiran (%)",
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    score_midterm = forms.IntegerField(
        label="Nilai Ujian Tengah Semester",
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    score_final = forms.IntegerField(
        label="Nilai Ujian Akhir Semester",
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    score_project = forms.IntegerField(
        label="Nilai Proyek",
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )