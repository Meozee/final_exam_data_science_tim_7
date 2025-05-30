from django import forms

# GANTI DENGAN PILIHAN AKTUAL DARI DATA TRAINING ANDA
# Ini adalah nilai-nilai SEBELUM proses get_dummies
DEPARTEMEN_CHOICES = [
    ('TI', 'Teknik Informatika'), # ('nilai_aktual_data', 'Label yang Ditampilkan')
    ('SI', 'Sistem Informasi'),
    ('DKV', 'Desain Komunikasi Visual'),
    # Tambahkan departemen lainnya sesuai data Anda
]

GENDER_CHOICES = [
    ('L', 'Laki-laki'), # Sesuaikan dengan nilai di kolom 'gender' pada ml_df sebelum get_dummies
    ('P', 'Perempuan'),
    # Tambahkan gender lainnya jika ada
]

COURSE_DIFFICULTY_CHOICES = [
    ('low', 'Rendah'), # Sesuaikan dengan nilai di kolom 'course_difficulty' pada ml_df sebelum get_dummies
    ('medium', 'Sedang'),
    ('high', 'Tinggi'),
]

class IPPredictionForm(forms.Form):
    ipk_sekarang = forms.FloatField(label='IPK Saat Ini (Kumulatif)', min_value=0.0, max_value=4.0,
                                    widget=forms.NumberInput(attrs={'step': '0.01'}))
    ip_semester_lalu = forms.FloatField(label='IP Semester Lalu', min_value=0.0, max_value=4.0,
                                        widget=forms.NumberInput(attrs={'step': '0.01'}))
    attendance_percentage = forms.FloatField(label='Persentase Kehadiran Semester Lalu (%)', min_value=0.0, max_value=100.0,
                                             widget=forms.NumberInput(attrs={'step': '0.1'}))
    departemen = forms.ChoiceField(label='Departemen', choices=DEPARTEMEN_CHOICES)
    gender = forms.ChoiceField(label='Gender', choices=GENDER_CHOICES)
    course_difficulty = forms.ChoiceField(label='Tingkat Kesulitan Mata Kuliah (Umum Semester Depan)', choices=COURSE_DIFFICULTY_CHOICES)

    def clean_attendance_percentage(self):
        attendance = self.cleaned_data['attendance_percentage']
        # Model Anda sepertinya mengharapkan nilai persentase (misal, 80 untuk 80%)
        # Jika model mengharapkan antara 0-1, lakukan konversi: return attendance / 100.0
        return attendance
    

    from django import forms

# Anda perlu mendapatkan daftar departemen dari database untuk pilihan yang dinamis
# Untuk sementara, kita bisa hardcode atau Anda bisa buat fungsi untuk mengambilnya.
# Contoh:
# from .utils import get_department_choices_for_course # Anda perlu buat file utils.py
# atau ambil dari model Django jika sudah ada
try:
    # Jika Anda sudah punya model Department di Django
    # from your_main_app.models import Department 
    # DEPT_CHOICES = [(dept.dept_id, dept.dept_name) for dept in Department.objects.all()]
    # Jika tidak, gunakan placeholder atau fungsi dari skrip training (jika dimodifikasi)
    DEPT_CHOICES = [
        (1, 'Teknik Informatika (Placeholder)'), 
        (2, 'Sistem Informasi (Placeholder)'),
        # Tambahkan/sesuaikan berdasarkan data dept_id di tabel course Anda
    ]
except ImportError:
    DEPT_CHOICES = [ (i, f'Departemen ID {i}') for i in range(1, 6)]


class CourseDifficultyForm(forms.Form):
    average_grade_course = forms.FloatField(
        label='Perkiraan Rata-rata Nilai Mata Kuliah (0-100)', 
        min_value=0.0, 
        max_value=100.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
    )
    assessment_count_course = forms.IntegerField(
        label='Perkiraan Jumlah Asesmen di Mata Kuliah', 
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    dept_id = forms.ChoiceField(
        label='Departemen Penyelenggara Mata Kuliah',
        choices=DEPT_CHOICES, # Pastikan ini diisi dengan ID departemen yang valid
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    # Tambahkan field lain jika Anda menambahkannya sebagai fitur di model
    # misalnya sks_mata_kuliah = forms.IntegerField(label='Jumlah SKS Mata Kuliah')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Jika Anda ingin memuat DEPT_CHOICES secara dinamis setiap kali form dibuat:
        # self.fields['dept_id'].choices = get_department_choices_for_course()