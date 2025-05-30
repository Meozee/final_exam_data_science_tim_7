# usecase_miko/forms.py
from django import forms
from django.core.validators import MinValueValidator, MaxValueValidator

# Pilihan untuk Dropdown (Dapat disesuaikan atau diambil dari model jika memungkinkan)
DEPT_CHOICES = [
    ('1', 'Teknik Informatika'), 
    ('2', 'Sistem Informasi'),
] # Pastikan value ini string jika akan di-cast ke int di view, atau sesuaikan dengan tipe data di DB/model
COURSE_CHOICES = [ 
    ('Konsentrasi Sistem Informasi', 'Konsentrasi Sistem Informasi'),
    ('Konsentrasi Teknologi Informasi', 'Konsentrasi Teknologi Informasi'),
    ('Data Science', 'Data Science'),
    ('Artificial Intelligence', 'Artificial Intelligence'),
    ('Software Engineering', 'Software Engineering'),
    ('Cyber Security', 'Cyber Security'),
    ('Cloud Computing', 'Cloud Computing'),
    ('Networking', 'Networking'),
    ('Database Management', 'Database Management'),
    ('Web Development', 'Web Development'),
    ('Mobile Development', 'Mobile Development'),
    ('UI/UX Design', 'UI/UX Design'),
    ('Project Management', 'Project Management'),
    ('Business Analysis', 'Business Analysis'),
    ('IT Audit', 'IT Audit'),
    ('IT Governance', 'IT Governance'),
    ('Enterprise Architecture', 'Enterprise Architecture'),
    ('Digital Marketing', 'Digital Marketing'),
    ('E-commerce', 'E-commerce'),
    ('Game Development', 'Game Development'),
]
DIFFICULTY_CHOICES = [('Easy', 'Easy'), ('Medium', 'Medium'), ('Hard', 'Hard')]
GENDER_CHOICES = [('Male', 'Laki-laki'), ('Female', 'Perempuan')]

# --- Form untuk Model 1: Prediksi Risiko Mahasiswa ---
class RiskPredictionForm(forms.Form):
    gender = forms.ChoiceField(choices=GENDER_CHOICES, label="Jenis Kelamin", widget=forms.Select(attrs={'class': 'form-select'}))
    dept_name = forms.ChoiceField(choices=DEPT_CHOICES, label="Nama Departemen", widget=forms.Select(attrs={'class': 'form-select'}))
    course_name = forms.ChoiceField(choices=COURSE_CHOICES, label="Nama Mata Kuliah", widget=forms.Select(attrs={'class': 'form-select'}))
    difficulty_level = forms.ChoiceField(choices=DIFFICULTY_CHOICES, label="Tingkat Kesulitan", widget=forms.Select(attrs={'class': 'form-select'}))
    attendance_percentage = forms.IntegerField(label="Persentase Kehadiran (%)", validators=[MinValueValidator(0), MaxValueValidator(100)], widget=forms.NumberInput(attrs={'class': 'form-control'}))
    score_midterm = forms.IntegerField(label="Nilai Ujian Tengah Semester", validators=[MinValueValidator(0), MaxValueValidator(100)], widget=forms.NumberInput(attrs={'class': 'form-control'}))
    score_final = forms.IntegerField(label="Nilai Ujian Akhir Semester", validators=[MinValueValidator(0), MaxValueValidator(100)], widget=forms.NumberInput(attrs={'class': 'form-control'}))
    score_project = forms.IntegerField(label="Nilai Proyek", validators=[MinValueValidator(0), MaxValueValidator(100)], widget=forms.NumberInput(attrs={'class': 'form-control'}))

# --- Form untuk Model 2: Deteksi Anomali dengan Input Manual ---
class AnomalyDetectionInputForm(forms.Form):
    score_assessment_1 = forms.IntegerField(label="Skor Penilaian 1 (cth: Tugas)", validators=[MinValueValidator(0), MaxValueValidator(100)], widget=forms.NumberInput(attrs={'class': 'form-control'}))
    score_assessment_2 = forms.IntegerField(label="Skor Penilaian 2 (cth: Kuis)", validators=[MinValueValidator(0), MaxValueValidator(100)], widget=forms.NumberInput(attrs={'class': 'form-control'}))
    current_attendance_percentage = forms.IntegerField(label="Persentase Kehadiran Saat Ini (%)", validators=[MinValueValidator(0), MaxValueValidator(100)], widget=forms.NumberInput(attrs={'class': 'form-control'}))
    historical_average_score = forms.FloatField(label="Rata-rata Skor Historis (Sebelum Penilaian Ini)", required=False, widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Kosongkan jika penilaian pertama'}))
    class_average_score_for_assessment = forms.FloatField(label="Rata-rata Skor Kelas untuk Penilaian Serupa", required=False, widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Kosongkan jika tidak diketahui'}))
    class_std_score_for_assessment = forms.FloatField(label="Standar Deviasi Skor Kelas untuk Penilaian Serupa", required=False, widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Kosongkan jika tidak diketahui'}))

# Form untuk model "Enhanced" (Regresi)
class EnhancedGPAPredictionForm(forms.Form):
    nama_mahasiswa = forms.CharField(label="Nama Anda", max_length=100, widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Contoh: Budi Perkasa'}))
    gender = forms.ChoiceField(label="Jenis Kelamin", choices=GENDER_CHOICES, widget=forms.Select(attrs={'class': 'form-select'}))
    departemen = forms.ChoiceField(label="Departemen", choices=DEPT_CHOICES, widget=forms.Select(attrs={'class': 'form-select'})) # Asumsi 'departemen' masih digunakan & di-OHE
    
    # Fitur-fitur yang mungkin digunakan oleh model "Enhanced" Anda
    # Sesuaikan field ini berdasarkan fitur aktual yang digunakan model Anda sebelum OHE
    ipk_kumulatif = forms.FloatField(label="IPK Kumulatif Saat Ini", widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}))
    ip_semester_terakhir = forms.FloatField(label="IP Semester Terakhir", widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}))
    ip_semester_2_terakhir = forms.FloatField(label="IP 2 Semester Terakhir", widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}))
    ip_trend_rata = forms.FloatField(label="Rata-rata Tren IP", widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}))
    
    kehadiran_semester_ini = forms.FloatField(label="Kehadiran Semester Ini (%)", widget=forms.NumberInput(attrs={'class': 'form-control'}))
    min_kehadiran = forms.FloatField(label="Minimal Kehadiran di Satu MK Semester Ini (%)", widget=forms.NumberInput(attrs={'class': 'form-control'}))

    min_grade_semester = forms.FloatField(label="Nilai Angka Terendah Semester Lalu", widget=forms.NumberInput(attrs={'class': 'form-control'}))
    max_grade_semester = forms.FloatField(label="Nilai Angka Tertinggi Semester Lalu", widget=forms.NumberInput(attrs={'class': 'form-control'}))
    grade_consistency = forms.FloatField(label="Konsistensi Nilai (Std Dev Nilai Semester Lalu)", widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}))
    
    avg_courses_per_semester = forms.FloatField(label="Rata-rata Jumlah MK per Semester", widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}))
    semester_ke_input = forms.IntegerField(label="Semester yang Baru Selesai", widget=forms.NumberInput(attrs={'class': 'form-control'})) # Jika masih relevan
    # Tambahkan field lain jika model "Enhanced" Anda menggunakannya

    # /usecase_miko/forms.py

from django import forms

# Pilihan ini bisa Anda sesuaikan dengan data master Anda
GENDER_CHOICES = [('Male', 'Laki-laki'), ('Female', 'Perempuan')]
DEPT_CHOICES = [
    ('1', 'Teknik Informatika'), 
    ('2', 'Sistem Informasi'),
    # Tambahkan departemen lain jika ada
]
# Asumsi tipe assessment yang mungkin ada (sesuaikan dengan nama kolom setelah pivot)
# Ini hanya untuk referensi, form tidak akan meminta ini secara langsung jika fitur adalah rata-rata
ASSESSMENT_TYPES_FOR_LABELS = {
    'score_midterm': 'Midterm',
    'score_final': 'Final Exam',
    'score_project': 'Project'
    # Tambahkan tipe assessment lain jika ada, misal: 'score_quiz', 'score_tugas'
}


class PredictNextSemesterIPForm(forms.Form):
    nama_mahasiswa = forms.CharField(
        label="Nama Mahasiswa (Opsional)", 
        max_length=100, 
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Contoh: Budi Santoso'})
    )
    gender = forms.ChoiceField(
        label="Jenis Kelamin",
        choices=GENDER_CHOICES, 
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    departemen = forms.ChoiceField(
        label="Departemen",
        choices=DEPT_CHOICES, 
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    # --- Data untuk semester historis pertama yang diinput pengguna ---
    # (Ini akan menjadi data Semester 2 jika target prediksi S4)
    avg_score_midterm_sem1 = forms.FloatField(label="Rata-rata Skor Midterm (Hist. Sem. Pertama)", widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}))
    avg_score_final_sem1 = forms.FloatField(label="Rata-rata Skor Final (Hist. Sem. Pertama)", widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}))
    avg_score_project_sem1 = forms.FloatField(label="Rata-rata Skor Project (Hist. Sem. Pertama)", widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}))
    avg_kehadiran_sem1 = forms.FloatField(label="Rata-rata Kehadiran (Hist. Sem. Pertama) (%)", min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}))
    jumlah_mk_sem1 = forms.IntegerField(label="Jumlah Mata Kuliah (Hist. Sem. Pertama)", min_value=0, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    jumlah_easy_sem1 = forms.IntegerField(label="Jumlah MK Easy (Hist. Sem. Pertama)", min_value=0, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    jumlah_medium_sem1 = forms.IntegerField(label="Jumlah MK Medium (Hist. Sem. Pertama)", min_value=0, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    jumlah_hard_sem1 = forms.IntegerField(label="Jumlah MK Hard (Hist. Sem. Pertama)", min_value=0, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    # jumlah_unknown_sem1 sudah kita hapus dari kesepakatan sebelumnya

    # --- Data untuk semester historis kedua yang diinput pengguna ---
    # (Ini akan menjadi data Semester 3 jika target prediksi S4)
    avg_score_midterm_sem2 = forms.FloatField(label="Rata-rata Skor Midterm (Hist. Sem. Kedua)", widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}))
    avg_score_final_sem2 = forms.FloatField(label="Rata-rata Skor Final (Hist. Sem. Kedua)", widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}))
    avg_score_project_sem2 = forms.FloatField(label="Rata-rata Skor Project (Hist. Sem. Kedua)", widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}))
    avg_kehadiran_sem2 = forms.FloatField(label="Rata-rata Kehadiran (Hist. Sem. Kedua) (%)", min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}))
    jumlah_mk_sem2 = forms.IntegerField(label="Jumlah Mata Kuliah (Hist. Sem. Kedua)", min_value=0, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    jumlah_easy_sem2 = forms.IntegerField(label="Jumlah MK Easy (Hist. Sem. Kedua)", min_value=0, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    jumlah_medium_sem2 = forms.IntegerField(label="Jumlah MK Medium (Hist. Sem. Kedua)", min_value=0, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    jumlah_hard_sem2 = forms.IntegerField(label="Jumlah MK Hard (Hist. Sem. Kedua)", min_value=0, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    # jumlah_unknown_sem2 sudah kita hapus

    # Hapus method clean() jika isinya hanya terkait field 'jumlah_unknown' yang sudah dihapus
    # def clean(self):
    #     cleaned_data = super().clean()
    #     # ... (logika clean lainnya jika ada) ...
    #     return cleaned_data
    # /usecase_miko/forms.py

from django import forms

# ... (GENDER_CHOICES, DEPT_CHOICES tetap sama) ...

class LecturerEffectIPPredictForm(forms.Form): # NAMA BARU
    # Informasi Umum
    # nama_mahasiswa tetap opsional
    gender = forms.ChoiceField(label="Jenis Kelamin Anda", choices=GENDER_CHOICES, widget=forms.Select(attrs={'class': 'form-select'}))
    dept_id = forms.ChoiceField(label="Departemen Anda", choices=DEPT_CHOICES, widget=forms.Select(attrs={'class': 'form-select'}))

    # Semester N-1 (misal, Semester 1 untuk prediksi S3, atau Semester 2 untuk prediksi S4)
    # Label bisa lebih generik karena akan dijelaskan di HTML
    avg_score_midterm_sem_hist1 = forms.FloatField(label="Rata-rata Skor Midterm (Semester Historis ke-1)", widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}))
    avg_score_final_sem_hist1 = forms.FloatField(label="Rata-rata Skor Final (Semester Historis ke-1)", widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}))
    avg_score_project_sem_hist1 = forms.FloatField(label="Rata-rata Skor Project (Semester Historis ke-1)", widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}))
    avg_kehadiran_sem_hist1 = forms.FloatField(label="Rata-rata Kehadiran (Semester Historis ke-1) (%)", min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}))
    jumlah_mk_sem_hist1 = forms.IntegerField(label="Jumlah Mata Kuliah (Semester Historis ke-1)", min_value=0, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    jumlah_easy_sem_hist1 = forms.IntegerField(label="Jumlah MK Easy (Semester Historis ke-1)", min_value=0, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    jumlah_medium_sem_hist1 = forms.IntegerField(label="Jumlah MK Medium (Semester Historis ke-1)", min_value=0, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    jumlah_hard_sem_hist1 = forms.IntegerField(label="Jumlah MK Hard (Semester Historis ke-1)", min_value=0, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    jumlah_unknown_sem_hist1 = forms.IntegerField(label="Jumlah MK Unknown Difficulty (Semester Historis ke-1)", min_value=0, required=False, initial=0, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    jumlah_dosen_unik_sem_hist1 = forms.IntegerField(label="Jumlah Dosen Unik (Semester Historis ke-1)", min_value=0, widget=forms.NumberInput(attrs={'class': 'form-control'}))

    # Semester N (misal, Semester 2 untuk prediksi S3, atau Semester 3 untuk prediksi S4)
    avg_score_midterm_sem_hist2 = forms.FloatField(label="Rata-rata Skor Midterm (Semester Historis ke-2)", widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}))
    avg_score_final_sem_hist2 = forms.FloatField(label="Rata-rata Skor Final (Semester Historis ke-2)", widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}))
    avg_score_project_sem_hist2 = forms.FloatField(label="Rata-rata Skor Project (Semester Historis ke-2)", widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}))
    avg_kehadiran_sem_hist2 = forms.FloatField(label="Rata-rata Kehadiran (Semester Historis ke-2) (%)", min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}))
    jumlah_mk_sem_hist2 = forms.IntegerField(label="Jumlah Mata Kuliah (Semester Historis ke-2)", min_value=0, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    jumlah_easy_sem_hist2 = forms.IntegerField(label="Jumlah MK Easy (Semester Historis ke-2)", min_value=0, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    jumlah_medium_sem_hist2 = forms.IntegerField(label="Jumlah MK Medium (Semester Historis ke-2)", min_value=0, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    jumlah_hard_sem_hist2 = forms.IntegerField(label="Jumlah MK Hard (Semester Historis ke-2)", min_value=0, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    jumlah_unknown_sem_hist2 = forms.IntegerField(label="Jumlah MK Unknown Difficulty (Semester Historis ke-2)", min_value=0, required=False, initial=0, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    jumlah_dosen_unik_sem_hist2 = forms.IntegerField(label="Jumlah Dosen Unik (Semester Historis ke-2)", min_value=0, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    
    # Opsional: Tambahkan nama mahasiswa jika ingin tetap ada
    nama_mahasiswa = forms.CharField(label="Nama Mahasiswa (Opsional)", max_length=100, required=False, widget=forms.TextInput(attrs={'class': 'form-control'}))


    def clean(self):
        cleaned_data = super().clean()
        for sem_suffix in ['_sem_hist1', '_sem_hist2']: # Sesuaikan dengan akhiran field baru
            unknown_field = f'jumlah_unknown{sem_suffix}'
            if unknown_field in cleaned_data and cleaned_data.get(unknown_field) is None:
                cleaned_data[unknown_field] = 0
        return cleaned_data

# Simpan form lama Anda jika masih digunakan oleh view lain
# class PredictNextSemesterIPForm(forms.Form): ... 
# class RiskPredictionForm(forms.Form): ...
# class AnomalyDetectionInputForm(forms.Form): ...