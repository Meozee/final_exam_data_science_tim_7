from django import forms

class AtRiskForm(forms.Form):
    student_id = forms.CharField(label='Student ID', max_length=20)
    gpa = forms.FloatField(label='Current GPA', min_value=0.0, max_value=4.0)
    credits_taken = forms.IntegerField(label='Credits Taken', min_value=0)
    attendance_rate = forms.FloatField(label='Attendance Rate (%)', min_value=0.0, max_value=100.0)
    gender = forms.ChoiceField(label='Gender', choices=[('Male', 'Male'), ('Female', 'Female')], required=False)

class CourseLoadForm(forms.Form):
    student_id = forms.CharField(label='Student ID', max_length=20)
    gpa = forms.FloatField(label='Current GPA', min_value=0.0, max_value=4.0)
    work_hours_per_week = forms.IntegerField(label='Work Hours Per Week', min_value=0)

