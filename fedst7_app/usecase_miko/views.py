from django.shortcuts import render
from .forms import PredictForm
import joblib
import numpy as np
import os

def predict_score(request):
    prediction = None
    if request.method == 'POST':
        form = PredictForm(request.POST)
        if form.is_valid():
            data = np.array([[
                form.cleaned_data['attendance'],
                form.cleaned_data['quiz_score'],
                form.cleaned_data['midterm_score'],
                form.cleaned_data['project_score']
            ]])

            model_path = os.path.join('ml_models', 'miko_grade_predictor.pkl')
            model = joblib.load(model_path)
            prediction = model.predict(data)[0]

            return render(request, 'usecase_miko/prediction_result.html', {
                'form': form,
                'prediction': round(prediction, 2)
            })
    else:
        form = PredictForm()

    return render(request, 'usecase_miko/input_student_score.html', {'form': form})
