from django import forms

class PredictForm(forms.Form):
    attendance = forms.FloatField(label='Attendance (%)')
    quiz_score = forms.FloatField(label='Quiz Score')
    midterm_score = forms.FloatField(label='Midterm Score')
    project_score = forms.FloatField(label='Project Score')
