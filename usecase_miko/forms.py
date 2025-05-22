from django import forms

DIFFICULTY_LEVELS = [
    ('Easy', 'Easy'),
    ('Medium', 'Medium'),
    ('Hard', 'Hard'),
]

class StudentPerformanceForm(forms.Form):
    attendance_percentage = forms.IntegerField(
        min_value=0,
        max_value=100,
        label="Attendance (%)"
    )
    midterm_score = forms.IntegerField(
        min_value=0,
        max_value=100,
        label="Midterm Score"
    )
    project_score = forms.IntegerField(
        min_value=0,
        max_value=100,
        label="Project Score"
    )
    difficulty_level = forms.ChoiceField(
        choices=DIFFICULTY_LEVELS,
        label="Course Difficulty"
    )

class InstructorForm(forms.Form):
    instructor_id = forms.IntegerField(label='Instructor ID')
    semester = forms.IntegerField(label='Semester')
    num_students = forms.IntegerField(label='Jumlah Mahasiswa')
    avg_midterm = forms.FloatField(label='Rata-rata UTS')
    avg_quiz = forms.FloatField(label='Rata-rata Kuis')
    avg_project = forms.FloatField(label='Rata-rata Proyek')