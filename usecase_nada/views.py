# usecase_nada/views.py

import pandas as pd
import pickle
import os
from django.shortcuts import render
from django.conf import settings # Pastikan settings diimpor
import traceback

from .forms import CareerPredictionForm # Asumsi forms.py ada di direktori yang sama

# --- PEMUATAN MODEL & ARTEFAK UNTUK PREDIKSI KARIR NADA ---
career_model_pipeline = None
career_original_feature_columns = None

try:
    # Path BARU ke folder tempat .pkl disimpan, relatif terhadap BASE_DIR proyek Django
    # Jika folder Anda bernama 'ml_model' (singular) di root proyek:
    NADA_MODEL_BASE_DIR = os.path.join(settings.BASE_DIR, 'ml_models') 

    # Asumsi nama file masih sama seperti output skrip training standalone terakhir
    MODEL_FILENAME = 'nada_career_predictor_standalone.pkl'
    COLUMNS_FILENAME = 'nada_career_predictor_features_standalone.pkl'

    MODEL_FILE_PATH = os.path.join(NADA_MODEL_BASE_DIR, MODEL_FILENAME)
    COLUMNS_FILE_PATH = os.path.join(NADA_MODEL_BASE_DIR, COLUMNS_FILENAME)

    if not os.path.exists(MODEL_FILE_PATH):
        raise FileNotFoundError(f"File model utama tidak ditemukan: {MODEL_FILE_PATH}. Pastikan model sudah dilatih dan diletakkan di path yang benar (ml_model/).")
    if not os.path.exists(COLUMNS_FILE_PATH):
        raise FileNotFoundError(f"File kolom fitur tidak ditemukan: {COLUMNS_FILE_PATH}. Pastikan file fitur sudah dilatih dan diletakkan di path yang benar (ml_model/).")

    with open(MODEL_FILE_PATH, 'rb') as f:
        career_model_pipeline = pickle.load(f)
    with open(COLUMNS_FILE_PATH, 'rb') as f:
        career_original_feature_columns = pickle.load(f)

    print(f"✅ Model Prediksi Karir (Nada) dan fitur berhasil dimuat dari {NADA_MODEL_BASE_DIR}.")
    print(f"   File model: {MODEL_FILENAME}")
    print(f"   File fitur: {COLUMNS_FILENAME}")
    print(f"   Fitur yang diharapkan model: {career_original_feature_columns}")

except FileNotFoundError as fnfe: 
    print(f"❌ ERROR FileNotFoundError (Nada Model): {str(fnfe)}")
    print(f"   Pastikan folder 'ml_model' ada di root proyek Anda ({settings.BASE_DIR}) dan berisi file '{MODEL_FILENAME}' dan '{COLUMNS_FILENAME}'.")
    traceback.print_exc() 
except Exception as e:
    print(f"❌ ERROR UMUM saat memuat Model Prediksi Karir (Nada): {e}")
    traceback.print_exc() 

# ... sisa kode views.py (fungsi predict_career_view) tetap sama seperti yang saya berikan sebelumnya ...
# (Pastikan fungsi predict_career_view dan import lainnya ada di sini)
def predict_career_view(request):
    form = CareerPredictionForm(request.POST or None)
    context = {
        'form': form, 
        'use_case_title': 'Prediksi Rekomendasi Karir',
    }

    if request.method == 'POST' and form.is_valid():
        if not career_model_pipeline or not career_original_feature_columns:
            context['error_message'] = "Model Prediksi Karir tidak dapat dimuat. Hubungi administrator."
        else:
            try:
                input_data_raw = form.cleaned_data.copy() 
                
                if 'student_dept_id' in input_data_raw:
                    input_data_raw['dept_id'] = input_data_raw.pop('student_dept_id')
                
                if 'nama_mahasiswa' in input_data_raw:
                    del input_data_raw['nama_mahasiswa']

                input_df = pd.DataFrame([input_data_raw])
                # print(f"Input DataFrame (sebelum pemilihan kolom): {input_df.to_dict()}") # Untuk debugging
                
                try:
                    input_df_for_model = input_df[career_original_feature_columns]
                except KeyError as ke:
                    missing_cols_in_input = set(career_original_feature_columns) - set(input_df.columns)
                    extra_cols_in_input = set(input_df.columns) - set(career_original_feature_columns)
                    error_msg = (f"Ketidakcocokan kolom input. "
                                 f"Kolom diharapkan model tapi tidak ada di input: {missing_cols_in_input or 'Tidak ada'}. "
                                 f"Kolom ada di input tapi tidak diharapkan model: {extra_cols_in_input or 'Tidak ada'}. "
                                 f"Original Error: {ke}")
                    context['error_message'] = error_msg
                    return render(request, 'usecase_nada/predict_career_form.html', context)

                # print(f"Input DataFrame untuk model (setelah pemilihan kolom): {input_df_for_model.to_dict()}") # Untuk debugging
                prediction = career_model_pipeline.predict(input_df_for_model)
                prediction_proba = career_model_pipeline.predict_proba(input_df_for_model)
                
                context['prediction_result'] = prediction[0]
                
                if 'classifier' in career_model_pipeline.named_steps:
                    classes = career_model_pipeline.named_steps['classifier'].classes_
                else:
                    classes = career_model_pipeline.steps[-1][1].classes_

                probabilities = prediction_proba[0]
                
                career_probabilities = []
                for i, career_class in enumerate(classes):
                    career_probabilities.append({
                        'career': career_class,
                        'probability': round(probabilities[i] * 100, 2)
                    })
                career_probabilities = sorted(career_probabilities, key=lambda x: x['probability'], reverse=True)
                
                context['career_probabilities'] = career_probabilities
                context['top_prediction'] = career_probabilities[0] if career_probabilities else None
                context['nama_mahasiswa_display'] = form.cleaned_data.get('nama_mahasiswa')

            except Exception as e:
                context['error_message'] = f"Terjadi kesalahan saat proses prediksi: {e}"
                traceback.print_exc()
    
    return render(request, 'usecase_nada/predict_career_form.html', context)