from django.shortcuts import render
from .forms import PredictForm
import joblib
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

def predict_score(request):
    prediction = None
    mae = None
    rmse = None
    coef_info = []

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

            # Dummy actual (misal nilai aslinya 80)
            actual = [80]
            mae = mean_absolute_error(actual, [prediction])
            rmse = mean_squared_error(actual, [prediction], squared=False)

            # Koefisien model regresi (jika ada)
            try:
                coefs = model.coef_
                features = ['Attendance', 'Quiz', 'Midterm', 'Project']
                coef_info = zip(features, coefs)
            except AttributeError:
                coef_info = []

            return render(request, 'usecase_miko/prediction_result.html', {
                'form': form,
                'prediction': round(prediction, 2),
                'mae': round(mae, 2),
                'rmse': round(rmse, 2),
                'coef_info': coef_info,
            })
    else:
        form = PredictForm()

    return render(request, 'usecase_miko/input_student_score.html', {'form': form})
