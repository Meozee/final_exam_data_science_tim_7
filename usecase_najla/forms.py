from django import forms
# Ganti 'fedst7_app' dengan nama aplikasi Django tempat model Course dan Semester didefinisikan
# Atau jika semua model (Course, Semester, Student) ada di satu app seperti 'student_app'
try:
    from fedst7_app.models import Course, Semester 
    # from student.models import Student # Jika stu_id mau dipilih dari daftar Student
except ImportError:
    Course = Semester = None
    print("WARNING: Django models (Course, Semester) tidak ditemukan untuk form Najla.")


class AttendancePredictionForm(forms.Form):
    # Field 'name' di HTML Anda. Ini bisa diisi nama atau ID mahasiswa.
    # Kita akan tangani di views untuk mendapatkan fitur mahasiswa lainnya.
    name = forms.CharField( # Jika ini seharusnya ID Mahasiswa, ubah ke IntegerField
        label='Nama atau ID Mahasiswa',
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    average_score = forms.FloatField(
        label='Rata-Rata Skor Asesmen Sebelumnya (0-100)',
        min_value=0,
        max_value=100,
        widget=forms.NumberInput(attrs={'step': '0.1', 'class': 'form-control'})
    )
    grade = forms.FloatField(
        label='Grade/Nilai Sebelumnya (0-100)',
        min_value=0,
        max_value=100,
        widget=forms.NumberInput(attrs={'step': '0.1', 'class': 'form-control'})
    )
    
    if Course:
        course_id_form = forms.ModelChoiceField( # Nama field di form (akan jadi course_id di cleaned_data)
            queryset=Course.objects.all(),
            label='Mata Kuliah yang Akan Diambil',
            empty_label="Pilih Mata Kuliah",
            widget=forms.Select(attrs={'class': 'form-control'}),
            to_field_name="course_id" # Pastikan ini menghasilkan ID jika diperlukan
        )
    else: 
        course_id_form = forms.IntegerField(label='ID Mata Kuliah', widget=forms.NumberInput(attrs={'class': 'form-control'}))

    if Semester:
        semester_id_form = forms.ModelChoiceField( # Nama field di form
            queryset=Semester.objects.all(),
            label='Semester yang Akan Diambil',
            empty_label="Pilih Semester",
            widget=forms.Select(attrs={'class': 'form-control'}),
            to_field_name="semester_id"
        )
    else:
        semester_id_form = forms.IntegerField(label='ID Semester', widget=forms.NumberInput(attrs={'class': 'form-control'}))

    # Fitur mahasiswa tambahan yang dibutuhkan oleh model (gender, dept_id, age)
    # Ini HARUS ada jika 'raw_feature_names' dari training menyertakannya.
    # Sesuaikan choices jika perlu
    GENDER_CHOICES = [('L', 'Laki-laki'), ('P', 'Perempuan'), ('Other', 'Lainnya')] # Sesuaikan dengan data Anda
    gender = forms.ChoiceField(choices=GENDER_CHOICES, label="Gender Mahasiswa", widget=forms.Select(attrs={'class':'form-control'}))
    
    # Anda perlu cara untuk mendapatkan DEPT_CHOICES, misalnya dari model Department
    # from department.models import Department
    # DEPT_CHOICES = [(dept.dept_id, dept.dept_name) for dept in Department.objects.all()]
    DEPT_CHOICES_PLACEHOLDER = [(1, 'Dept 1 (Placeholder)'), (2, 'Dept 2 (Placeholder)')] # Ganti ini
    dept_id = forms.ChoiceField(choices=DEPT_CHOICES_PLACEHOLDER, label="Departemen Mahasiswa", widget=forms.Select(attrs={'class':'form-control'}))
    
    age = forms.IntegerField(label="Usia Mahasiswa", min_value=15, max_value=100, widget=forms.NumberInput(attrs={'class':'form-control'}))


    # Ubah nama field agar sesuai dengan {{ form.course_id }} dan {{ form.semester_id }} di HTML
    # Ini adalah trik jika nama field di form berbeda dari yang diinginkan di template
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'course_id_form' in self.fields:
            self.fields['course_id'] = self.fields.pop('course_id_form')
        if 'semester_id_form' in self.fields:
            self.fields['semester_id'] = self.fields.pop('semester_id_form')