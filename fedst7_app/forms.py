from django import forms
from .models import MLModel

class MLModelForm(forms.ModelForm):
    class Meta:
        model = MLModel
        fields = ['name', 'model_type', 'description', 'use_case', 'model_file']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 3}),
            'use_case': forms.Textarea(attrs={'rows': 3}),
        }

class PredictionForm(forms.Form):
    input_data = forms.CharField(
        label='Input Data (JSON format)',
        widget=forms.Textarea(attrs={'rows': 3}),
        help_text='Enter input data in JSON format for prediction'
    )