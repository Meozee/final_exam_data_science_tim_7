# usecase_miko/views.py

import pickle
import pandas as pd
import numpy as np
from django.shortcuts import render
from .forms import RiskPredictionForm

# Definisikan path ke artefak model Anda
PIPELINE_PATH = 'ml_models/miko_student_risk_pipeline.pkl'
EXPLAINER_PATH = 'ml_models/miko_shap_explainer.pkl'
FEATURES_PATH = 'ml_models/miko_feature_names.pkl'

# Blok pemuatan model
try:
    with open(PIPELINE_PATH, 'rb') as f:
        pipeline = pickle.load(f)
    with open(EXPLAINER_PATH, 'rb') as f:
        explainer = pickle.load(f)
    with open(FEATURES_PATH, 'rb') as f:
        feature_names_processed = pickle.load(f)
except FileNotFoundError:
    pipeline, explainer, feature_names_processed = None, None, None


def generate_reason(shap_values, processed_feature_names):
    """
    Fungsi untuk mengambil fitur paling berpengaruh berdasarkan SHAP values,
    dan mengembalikan alasan yang mudah dimengerti.
    """
    shap_values_flat = shap_values.values.flatten()
    
    # Ambil indeks dengan kontribusi SHAP terbesar
    max_impact_idx = np.argmax(np.abs(shap_values_flat))
    
    # Ambil nama fitur dan nilai SHAP
    top_feature_full_name = processed_feature_names[max_impact_idx]
    top_shap_value = shap_values_flat[max_impact_idx]

    # Ubah nama fitur menjadi lebih manusiawi
    reason = top_feature_full_name.split('__')[-1].replace('_', ' ').title()
    
    if top_shap_value > 0:
        return f"Faktor risiko utama adalah: {reason}."
    else:
        return f"Faktor penekan risiko utama adalah: {reason}."


def predict_risk_view(request):
    form = RiskPredictionForm()
    result = None

    if request.method == 'POST':
        form = RiskPredictionForm(request.POST)
        if form.is_valid():
            # Ganti pengecekan dengan pembanding eksplisit is None
            if any(obj is None for obj in (pipeline, explainer, feature_names_processed)):
                return render(request, 'usecase_miko/risk_prediction.html', {
                    'form': form,
                    'error': 'Model tidak dapat dimuat.'
                })

            input_data = form.cleaned_data
            input_df = pd.DataFrame([input_data])

            # Prediksi
            prediction_val = pipeline.predict(input_df)[0]
            probability_val = pipeline.predict_proba(input_df)[0][1]

            # Proses untuk SHAP
            processed_input = pipeline.named_steps['preprocessor'].transform(input_df)
            shap_values = explainer(processed_input)

            # Alasan prediksi
            reason_text = generate_reason(shap_values, feature_names_processed)

            result = {
                'prediction': 'Beresiko Gagal' if prediction_val == 1 else 'Aman',
                'confidence': f"{probability_val * 100:.1f}%",
                'reason': reason_text,
                'is_risk': prediction_val == 1
            }

    context = {'form': form, 'result': result}
    return render(request, 'usecase_miko/risk_prediction.html', context)
