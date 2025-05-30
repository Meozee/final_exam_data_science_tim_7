# usecase_nada/forms.py

from django import forms

# PENTING: Isi choices ini dengan data valid dari database Anda
# (course_id sebagai value, course_name sebagai display)
# (dept_id sebagai value, dept_name sebagai display)
COURSE_CHOICES = [
    ('0', '--- Tidak Ada/Tidak Tahu ---'), 
    ('1', 'Dasar Pemrograman'), # Contoh
    ('2', 'Statistika Dasar'), # Contoh
    ('3', 'Jaringan Komputer'), # Contoh
    ('4', 'Software Developer Course'),
    ('5', 'IT Consultant Course'),
    ('6', 'System Analyst Course'),
     ('7', 'Web Developer Course'),
     ('8', 'Mobile Developer Course'),
     ('9', 'UI/UX Designer Course'),
     ('10', 'Cloud Engineer Course'),
     ('11', 'Data Scientist Course'),
     ('12', 'Business Analyst Course'),
     ('13', 'Data Engineer Course'),
     ('14', 'Financial Analyst Course'),
     ('15', 'Accounting Course'),
     ('16', 'Marketing Specialist Course'),
     ('17', 'Human Resources Course'),
     ('18', 'Project Manager Course'),
     ('19', 'Operations Manager Course'),
     ('20', 'Supply Chain Manager Course'),
]

GENDER_CHOICES = [
    ('female', 'Female'), 
    ('male', 'Male'),    
]

DEPT_CHOICES = [
    ('0', '--- Tidak Ada/Tidak Tahu ---'), # Tambahkan pilihan default jika departemen bisa tidak diketahui
    ('1', 'Teknik Informatika'), # Contoh
    ('2', 'Sistem Informasi'), # Contoh
    # ... (Lengkapi dengan semua dept_id dan dept_name dari tabel 'department')
]


class CareerPredictionForm(forms.Form):
    # Tambahkan field nama_mahasiswa jika ingin digunakan di template
    nama_mahasiswa = forms.CharField(
        label="Nama Anda (Opsional)", 
        required=False, 
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Masukkan nama Anda'})
    )
    gender = forms.ChoiceField(
        label="Jenis Kelamin", 
        choices=GENDER_CHOICES, 
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    # Nama field di form adalah student_dept_id, akan di-map ke dept_id di views
    student_dept_id = forms.ChoiceField( 
        label="Departemen Anda Saat Ini", 
        choices=DEPT_CHOICES, 
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    course_id_1 = forms.ChoiceField(label="Mata Kuliah Utama ke-1", choices=COURSE_CHOICES, widget=forms.Select(attrs={'class': 'form-select'}))
    grade_c1 = forms.IntegerField(label="Nilai Akhir MK ke-1 (0-100)", min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    attendance_c1 = forms.IntegerField(label="Kehadiran MK ke-1 (%)", min_value=0, max_value=100, widget=forms.NumberInput(attrs={'class': 'form-control'}))

    course_id_2 = forms.ChoiceField(label="Mata Kuliah Utama ke-2 (Opsional)", choices=COURSE_CHOICES, required=False, widget=forms.Select(attrs={'class': 'form-select'}))
    grade_c2 = forms.IntegerField(label="Nilai Akhir MK ke-2 (0-100)", min_value=0, max_value=100, required=False, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    attendance_c2 = forms.IntegerField(label="Kehadiran MK ke-2 (%)", min_value=0, max_value=100, required=False, widget=forms.NumberInput(attrs={'class': 'form-control'}))

    course_id_3 = forms.ChoiceField(label="Mata Kuliah Utama ke-3 (Opsional)", choices=COURSE_CHOICES, required=False, widget=forms.Select(attrs={'class': 'form-select'}))
    grade_c3 = forms.IntegerField(label="Nilai Akhir MK ke-3 (0-100)", min_value=0, max_value=100, required=False, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    attendance_c3 = forms.IntegerField(label="Kehadiran MK ke-3 (%)", min_value=0, max_value=100, required=False, widget=forms.NumberInput(attrs={'class': 'form-control'}))

    def clean(self):
        cleaned_data = super().clean()
        for i in range(1, 4): 
            course_id_field = f'course_id_{i}'
            grade_field = f'grade_c{i}'
            attendance_field = f'attendance_c{i}'

            current_course_id = cleaned_data.get(course_id_field)
            
            # Jika course opsional tidak diisi, pastikan grade dan attendance juga tidak dikirim (atau set default jika model mengharapkannya)
            if not current_course_id or current_course_id == '0': # '0' adalah value untuk "--- Tidak Ada/Tidak Tahu ---"
                cleaned_data[course_id_field] = '0' # Pastikan string '0' jika model dilatih dengan ini sebagai kategori
                cleaned_data[grade_field] = 0
                cleaned_data[attendance_field] = 0
            elif current_course_id != '0': # Kursus dipilih
                # Jika grade atau attendance untuk kursus yang dipilih kosong (karena required=False), set default
                if cleaned_data.get(grade_field) is None:
                    cleaned_data[grade_field] = 0 
                if cleaned_data.get(attendance_field) is None:
                    cleaned_data[attendance_field] = 0
        
        # Pastikan student_dept_id juga memiliki nilai default jika tidak dipilih
        if not cleaned_data.get('student_dept_id') or cleaned_data.get('student_dept_id') == '0':
            cleaned_data['student_dept_id'] = '0' # Asumsi '0' adalah ID untuk departemen tidak diketahui/tidak relevan

        return cleaned_data