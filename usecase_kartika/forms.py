from django import forms
import numpy as np

# Pilihan ini sebaiknya diambil dari data unik di database Anda saat training
# atau definisikan secara komprehensif. Untuk contoh, kita hardcode beberapa.
# Anda HARUS memastikan nilai ini sesuai dengan yang ada di data training Anda.
GENDER_CHOICES = [
    ('', '---------'),
    ('L', 'Laki-laki'),
    ('P', 'Perempuan'),
    # Tambahkan nilai lain jika ada di data Anda
]

# Ambil ini dari data unik df['dept_name'].unique() di skrip training
DEPT_NAME_CHOICES = [
    ('', '---------'),
    ('Teknik Informatika', 'Teknik Informatika'),
    ('Sistem Informasi', 'Sistem Informasi'),
    ('Manajemen Bisnis', 'Manajemen Bisnis'), # Contoh
    # Tambahkan semua dept_name unik dari data training Anda
]

# Ambil ini dari data unik df['course_name'].unique() di skrip training
COURSE_NAME_CHOICES = [
    ('', '---------'),
    ('Pemrograman Dasar', 'Pemrograman Dasar'),
    ('Basis Data', 'Basis Data'),
    ('Kalkulus I', 'Kalkulus I'), # Contoh
    # Tambahkan semua course_name unik dari data training Anda
]

# Ambil ini dari data unik df['difficulty_level'].unique() di skrip training
DIFFICULTY_LEVEL_CHOICES = [
    ('', '---------'),
    ('Mudah', 'Mudah'), # Sesuaikan dengan nilai di data Anda (Easy, Medium, Hard, dll.)
    ('Sedang', 'Sedang'),
    ('Sulit', 'Sulit'),
]


class RiskAssessmentForm(forms.Form):
    gender = forms.ChoiceField(
        label='Jenis Kelamin', 
        choices=GENDER_CHOICES, 
        widget=forms.Select(attrs={'class': 'form-control form-control-sm'})
    )
    dept_name = forms.ChoiceField(
        label='Nama Departemen', 
        choices=DEPT_NAME_CHOICES, 
        widget=forms.Select(attrs={'class': 'form-control form-control-sm'})
    )
    course_name = forms.ChoiceField(
        label='Nama Mata Kuliah', 
        choices=COURSE_NAME_CHOICES, 
        widget=forms.Select(attrs={'class': 'form-control form-control-sm'})
    )
    difficulty_level = forms.ChoiceField(
        label='Tingkat Kesulitan', 
        choices=DIFFICULTY_LEVEL_CHOICES, 
        widget=forms.Select(attrs={'class': 'form-control form-control-sm'})
    )
    attendance_percentage = forms.FloatField(
        label='Persentase Kehadiran (%)', 
        min_value=0, max_value=100, 
        widget=forms.NumberInput(attrs={'class': 'form-control form-control-sm', 'step': '0.1'})
    )
    score_midterm = forms.FloatField(
        label='Nilai Ujian Tengah Semester (0-100)', 
        min_value=0, max_value=100, required=False, # Buat opsional jika bisa kosong
        widget=forms.NumberInput(attrs={'class': 'form-control form-control-sm', 'step': '0.1'})
    )
    score_final = forms.FloatField(
        label='Nilai Ujian Akhir Semester (0-100)', 
        min_value=0, max_value=100, required=False,
        widget=forms.NumberInput(attrs={'class': 'form-control form-control-sm', 'step': '0.1'})
    )
    score_project = forms.FloatField(
        label='Nilai Proyek (0-100)', 
        min_value=0, max_value=100, required=False,
        widget=forms.NumberInput(attrs={'class': 'form-control form-control-sm', 'step': '0.1'})
    )

    def clean(self):
        cleaned_data = super().clean()
        # Imputasi sederhana jika field skor opsional tidak diisi, agar model tidak error
        # Pipeline sudah ada imputer, tapi lebih baik form juga handle
        for score_field in ['score_midterm', 'score_final', 'score_project']:
            if cleaned_data.get(score_field) is None:
                cleaned_data[score_field] = np.nan # Biarkan imputer di pipeline yang handle NaN
        return cleaned_data