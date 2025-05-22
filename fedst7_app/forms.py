from django import forms

class DummyForm(forms.Form):
    name = forms.CharField()
