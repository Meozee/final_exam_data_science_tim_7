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