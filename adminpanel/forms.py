# adminpanel/forms.py

from django import forms
from .models import MLModel

class MLModelForm(forms.ModelForm):
    class Meta:
        model = MLModel
        # Tentukan field mana yang ingin ditampilkan di form
        fields = [
            'name', 'description', 'model_type', 'creator', 
            'dataset_name', 'file_path', 'explainer_path', 
            'features_path', 'endpoint_url_name'
        ]
        # Kustomisasi label agar lebih ramah pengguna
        labels = {
            'name': 'Nama Model',
            'description': 'Deskripsi Use Case',
            'model_type': 'Tipe Model',
            'creator': 'Nama Pembuat',
            'dataset_name': 'Nama Dataset Training',
            'file_path': 'Path ke File Model (.pkl)',
            'explainer_path': 'Path ke File Explainer (.pkl)',
            'features_path': 'Path ke File Nama Fitur (.pkl)',
            'endpoint_url_name': 'Nama URL Endpoint (cth: usecase_miko:predict_risk)',
        }
        # Tambahkan widget untuk styling dengan Bootstrap
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'model_type': forms.Select(attrs={'class': 'form-select'}),
            'creator': forms.TextInput(attrs={'class': 'form-control'}),
            'dataset_name': forms.TextInput(attrs={'class': 'form-control'}),
            'file_path': forms.TextInput(attrs={'class': 'form-control'}),
            'explainer_path': forms.TextInput(attrs={'class': 'form-control'}),
            'features_path': forms.TextInput(attrs={'class': 'form-control'}),
            'endpoint_url_name': forms.TextInput(attrs={'class': 'form-control'}),
        }