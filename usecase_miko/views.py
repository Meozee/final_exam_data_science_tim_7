# usecase_miko/views.py
import pickle
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.conf import settings
# Pastikan semua form yang Anda gunakan di views ini diimpor dari .forms
from .forms import RiskPredictionForm, AnomalyDetectionInputForm, EnhancedGPAPredictionForm 
import os
import traceback

# --- MUAT SEMUA ARTEFAK MODEL ---

# Model 1: Prediksi Risiko
RISK_PIPELINE_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'miko_student_risk_pipeline.pkl')
RISK_EXPLAINER_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'miko_shap_explainer.pkl')
RISK_FEATURES_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'miko_feature_names.pkl')
risk_pipeline, risk_explainer, risk_feature_names = None, None, None
try:
    with open(RISK_PIPELINE_PATH, 'rb') as f: risk_pipeline = pickle.load(f)
    with open(RISK_EXPLAINER_PATH, 'rb') as f: risk_explainer = pickle.load(f)
    with open(RISK_FEATURES_PATH, 'rb') as f: risk_feature_names = pickle.load(f)
    print(f"✅ Model Prediksi Risiko (Miko) berhasil dimuat.")
except FileNotFoundError:
    print(f"⚠️ FileNotFoundError: Salah satu file model Prediksi Risiko tidak ditemukan. Cek path:"
          f"\nPipeline: {RISK_PIPELINE_PATH}"
          f"\nExplainer: {RISK_EXPLAINER_PATH}"
          f"\nFeatures: {RISK_FEATURES_PATH}")
except Exception as e: 
    print(f"⚠️ Peringatan Umum: Gagal memuat model Prediksi Risiko (Miko) - {str(e)}")
    # traceback.print_exc() # Uncomment untuk debug lebih detail jika perlu

# Model 2: Deteksi Anomali
ANOMALY_MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'miko_fraud_detection_model.pkl')
ANOMALY_SCALER_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'miko_fraud_detection_scaler.pkl')
anomaly_model, anomaly_scaler = None, None
try:
    with open(ANOMALY_MODEL_PATH, 'rb') as f: anomaly_model = pickle.load(f)
    with open(ANOMALY_SCALER_PATH, 'rb') as f: anomaly_scaler = pickle.load(f)
    print(f"✅ Model Deteksi Anomali (Miko) berhasil dimuat.")
except FileNotFoundError:
    print(f"⚠️ FileNotFoundError: Salah satu file model Deteksi Anomali tidak ditemukan. Cek path:"
          f"\nModel: {ANOMALY_MODEL_PATH}"
          f"\nScaler: {ANOMALY_SCALER_PATH}")
except Exception as e: 
    print(f"⚠️ Peringatan Umum: Gagal memuat model Deteksi Anomali (Miko) - {str(e)}")
    # traceback.print_exc()

# Model 3: Prediksi IP Semester Berikutnya (Input Manual Fleksibel)
# Menggunakan model yang dilatih oleh 'latih_model_ip_semester_berikutnya.py'
NEXT_GPA_MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'prediksi_ip_semester_berikutnya.pkl')
NEXT_GPA_COLUMNS_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'prediksi_ip_semester_berikutnya_columns.pkl')

next_gpa_model, next_gpa_model_columns = None, None
try:
    with open(NEXT_GPA_MODEL_PATH, 'rb') as f: next_gpa_model = pickle.load(f)
    with open(NEXT_GPA_COLUMNS_PATH, 'rb') as f: next_gpa_model_columns = pickle.load(f)
    print(f"✅ Model Prediksi IP Semester Berikutnya (Miko) berhasil dimuat dari: {NEXT_GPA_MODEL_PATH}")
except FileNotFoundError:
    print(f"⚠️ FileNotFoundError: File model GPA Next Semester ({NEXT_GPA_MODEL_PATH}) atau kolom ({NEXT_GPA_COLUMNS_PATH}) tidak ditemukan.")
    # Set model ke None agar view tahu model tidak termuat
    next_gpa_model = None
    next_gpa_model_columns = None
except Exception as e: 
    print(f"⚠️ Peringatan Umum: Gagal memuat model Prediksi IP Semester Berikutnya (Miko) - {str(e)}")
    next_gpa_model = None
    next_gpa_model_columns = None
    # traceback.print_exc()

# --- Helper Functions ---
def generate_risk_reason(shap_values, processed_feature_names):
    if not hasattr(shap_values, 'values') or not hasattr(processed_feature_names, '__iter__'): return "Tidak dapat menghasilkan alasan (data SHAP tidak valid)."
    if not isinstance(shap_values.values, np.ndarray) or shap_values.values.size == 0: return "Tidak dapat menghasilkan alasan (SHAP values kosong)."
    shap_values_flat = shap_values.values.flatten()
    if not shap_values_flat.size or len(processed_feature_names) != len(shap_values_flat): return "Tidak dapat menghasilkan alasan (SHAP values atau nama fitur tidak cocok)."
    max_impact_idx = np.argmax(np.abs(shap_values_flat))
    top_feature_full_name = processed_feature_names[max_impact_idx]
    top_shap_value = shap_values_flat[max_impact_idx]
    reason_part = top_feature_full_name.split('__')
    reason = reason_part[-1].replace('_', ' ').title()
    if top_shap_value > 0: return f"Faktor risiko utama yang meningkatkan kemungkinan adalah: {reason}."
    return f"Faktor utama yang menekan kemungkinan risiko adalah: {reason}."

def flag_anomaly_patterns(row_data):
    flags = []
    if row_data.get('is_anomaly') == -1:
        if row_data.get('score_jump', 0) > 25: flags.append('Skor Melonjak Tajam (+{:.0f} poin)'.format(row_data.get('score_jump',0)))
        if row_data.get('high_score_low_attendance', 0) == 1: flags.append('Nilai Tinggi (>=90) dengan Kehadiran Rendah (<60%)')
        if row_data.get('z_score_deviation', 0) > 2.0: flags.append('Skor Jauh di Atas Rata-rata Kelas ({:.1f} Std Dev)'.format(row_data.get('z_score_deviation',0)))
    return ', '.join(flags) if flags else 'Tidak ada pola mencurigakan spesifik terdeteksi.'

# --- Views ---
def predict_risk_view(request):
    form = RiskPredictionForm(request.POST or None)
    context = {'form': form, 'use_case_title': 'Prediksi Risiko Mahasiswa'}
    if request.method == 'POST' and form.is_valid():
        if not all([risk_pipeline, risk_explainer, risk_feature_names]):
            context['error'] = 'Model Prediksi Risiko tidak dapat dimuat. Hubungi administrator.'
        else:
            try:
                input_data = form.cleaned_data
                input_df = pd.DataFrame([input_data])
                prediction_val = risk_pipeline.predict(input_df)[0]
                probability_val = risk_pipeline.predict_proba(input_df)[0][1]
                processed_input_df = risk_pipeline.named_steps['preprocessor'].transform(input_df)
                
                actual_feature_names_for_shap = risk_feature_names
                try: # Mencoba mendapatkan nama fitur dari pipeline jika ada
                   actual_feature_names_for_shap = risk_pipeline.named_steps['preprocessor'].get_feature_names_out()
                except AttributeError:
                   pass # Gunakan risk_feature_names yang asli jika tidak bisa

                shap_values = risk_explainer(processed_input_df)
                reason_text = generate_risk_reason(shap_values, actual_feature_names_for_shap)
                context['result'] = {'prediction': 'Beresiko Gagal' if prediction_val == 1 else 'Cenderung Aman', 
                                   'confidence': f"{probability_val * 100:.1f}%", 'reason': reason_text, 'is_risk': prediction_val == 1}
            except Exception as e:
                context['error'] = f"Error saat prediksi risiko: {str(e)}"
                traceback.print_exc()
    return render(request, 'usecase_miko/risk_prediction.html', context)

def fraud_detection_view(request):
    form = AnomalyDetectionInputForm(request.POST or None)
    context = {'form': form, 'use_case_title': 'Deteksi Anomali Data Akademik'}
    if request.method == 'POST' and form.is_valid():
        if not all([anomaly_model, anomaly_scaler]):
            context['error'] = 'Model Deteksi Anomali tidak dapat dimuat. Hubungi administrator.'
        else:
            try:
                input_data = form.cleaned_data
                current_score = input_data.get('score_assessment_1', 0)
                attendance = input_data.get('current_attendance_percentage', 0)
                hist_avg_score = input_data.get('historical_average_score', current_score)
                score_jump_val = current_score - hist_avg_score
                class_avg = input_data.get('class_average_score_for_assessment')
                class_std = input_data.get('class_std_score_for_assessment')
                z_score_dev_val = 0
                if class_avg is not None and class_std is not None and class_std != 0:
                    z_score_dev_val = (current_score - class_avg) / class_std
                high_score_low_att_val = 1 if (current_score >= 90 and attendance < 60) else 0
                engineered_features = {'score': current_score, 'attendance_percentage': attendance, 'historical_avg_score': hist_avg_score, 
                                     'score_jump': score_jump_val, 'z_score_deviation': z_score_dev_val, 'high_score_low_attendance': high_score_low_att_val}
                features_for_model_list = ['score', 'attendance_percentage', 'historical_avg_score', 'score_jump', 'z_score_deviation', 'high_score_low_attendance']
                X_input_df = pd.DataFrame([engineered_features])[features_for_model_list].fillna(0)
                X_scaled = anomaly_scaler.transform(X_input_df)
                is_anomaly_pred = anomaly_model.predict(X_scaled)[0]
                anomaly_score_raw = anomaly_model.decision_function(X_scaled)[0]
                anomaly_score_display = 1 / (1 + np.exp(anomaly_score_raw))
                row_data_for_flagging = {**engineered_features, 'is_anomaly': is_anomaly_pred}
                flagged_patterns_text = flag_anomaly_patterns(row_data_for_flagging)
                context['results'] = {'input_values': {k:v for k,v in input_data.items() if k in ['score_assessment_1', 'score_assessment_2', 'current_attendance_percentage']},
                                   'is_anomaly': 'Terdeteksi Pola Anomali' if is_anomaly_pred == -1 else 'Pola Normal',
                                   'anomaly_score_processed': f"{anomaly_score_display:.2f} (Skor Anomali, mendekati 1 lebih anomali)",
                                   'raw_decision_score': f"{anomaly_score_raw:.4f}", 'flagged_patterns': flagged_patterns_text, 'is_truly_anomaly': is_anomaly_pred == -1}
            except Exception as e:
                context['error'] = f"Error saat deteksi anomali: {str(e)}"
                traceback.print_exc()
    return render(request, 'usecase_miko/fraud_detection_input.html', context)

# --- MUAT MODEL & KOLOM UNTUK PREDIKSI IP SEMESTER BERIKUTNYA (RANDOM FOREST) ---
MODEL_RF_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'prediksi_ip_semester_berikutnya_rf.pkl')
COLUMNS_RF_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'prediksi_ip_semester_berikutnya_rf_columns.pkl')

next_gpa_rf_model, next_gpa_rf_model_columns = None, None
try:
    with open(MODEL_RF_PATH, 'rb') as f:
        next_gpa_rf_model = pickle.load(f) # Ini sekarang adalah PIPELINE
    with open(COLUMNS_RF_PATH, 'rb') as f:
        next_gpa_rf_model_columns = pickle.load(f)
    print(f"✅ Model Prediksi IP Semester Berikutnya (Random Forest) berhasil dimuat dari: {MODEL_RF_PATH}")
except FileNotFoundError:
    print(f"⚠️ FileNotFoundError: File model ({MODEL_RF_PATH}) atau kolom ({COLUMNS_RF_PATH}) untuk Random Forest tidak ditemukan.")
    next_gpa_rf_model = None
    next_gpa_rf_model_columns = None
except Exception as e:
    print(f"⚠️ Peringatan Umum: Gagal memuat model Prediksi IP Semester Berikutnya (Random Forest) - {str(e)}")
    next_gpa_rf_model = None
    next_gpa_rf_model_columns = None
    traceback.print_exc()

# --- MUAT MODEL & KOLOM UNTUK PREDIKSI IP SEMESTER BERIKUTNYA (XGBOOST) ---
MODEL_XGB_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'prediksi_ip_semester_berikutnya_xgb.pkl')
COLUMNS_XGB_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'prediksi_ip_semester_berikutnya_xgb_columns.pkl')

next_gpa_xgb_model, next_gpa_xgb_model_columns = None, None
try:
    with open(MODEL_XGB_PATH, 'rb') as f:
        next_gpa_xgb_model = pickle.load(f) # Ini adalah PIPELINE
    with open(COLUMNS_XGB_PATH, 'rb') as f:
        next_gpa_xgb_model_columns = pickle.load(f) # Ini adalah nama kolom SEBELUM scaling/transformasi pipeline
    print(f"✅ Model Prediksi IP Semester Berikutnya (XGBoost) berhasil dimuat dari: {MODEL_XGB_PATH}")
except FileNotFoundError:
    print(f"⚠️ FileNotFoundError: File model ({MODEL_XGB_PATH}) atau kolom ({COLUMNS_XGB_PATH}) untuk XGBoost tidak ditemukan.")
    next_gpa_xgb_model = None
    next_gpa_xgb_model_columns = None
except Exception as e:
    print(f"⚠️ Peringatan Umum: Gagal memuat model Prediksi IP Semester Berikutnya (XGBoost) - {str(e)}")
    next_gpa_xgb_model = None
    next_gpa_xgb_model_columns = None
    traceback.print_exc()

# --- Model 3: Prediksi IP dengan Model "ENHANCED" (BARU) ---
enhanced_gpa_model, enhanced_gpa_config, enhanced_gpa_columns, enhanced_gpa_scaler = (None,) * 4
try:
    # Perhatikan path ke 'ml_models_enhanced'
    ENHANCED_MODEL_PATH_DIR = os.path.join(settings.BASE_DIR, 'usecase_miko', 'model', 'ml_models_enhanced') 
    
    # Asumsi nama file dari skrip "Enhanced" Anda (sesuaikan jika berbeda)
    MODEL_FILE = os.path.join(ENHANCED_MODEL_PATH_DIR, 'best_model.pkl') # Sesuaikan nama file .pkl model utama
    CONFIG_FILE = os.path.join(ENHANCED_MODEL_PATH_DIR, 'best_model_config.pkl') # Sesuaikan nama file config
    COLUMNS_FILE = os.path.join(ENHANCED_MODEL_PATH_DIR, 'feature_columns.pkl') # Sesuaikan nama file kolom
    SCALER_FILE = os.path.join(ENHANCED_MODEL_PATH_DIR, 'scaler.pkl') # Sesuaikan nama file scaler

    with open(MODEL_FILE, 'rb') as f: enhanced_gpa_model = pickle.load(f)
    with open(CONFIG_FILE, 'rb') as f: enhanced_gpa_config = pickle.load(f)
    with open(COLUMNS_FILE, 'rb') as f: enhanced_gpa_columns = pickle.load(f)
    
    if os.path.exists(SCALER_FILE): # Scaler mungkin tidak selalu ada
        with open(SCALER_FILE, 'rb') as f: enhanced_gpa_scaler = pickle.load(f)

    print("✅ Model Prediksi IP 'ENHANCED' berhasil dimuat.")
except Exception as e:
    print(f"❌ ERROR: Gagal memuat model 'ENHANCED': {e}")
    # traceback.print_exc() # Uncomment untuk debug detail jika pemuatan gagal

# ==============================================================================
# BAGIAN 3: VIEW FUNCTIONS
# ==============================================================================

# --- View untuk model lama (jika masih digunakan) ---
def predict_risk_view(request):
    form = RiskPredictionForm(request.POST or None)
    # ... (Logika lengkap view ini) ...
    return render(request, 'usecase_miko/risk_prediction.html', {'form': form, 'use_case_title': 'Prediksi Risiko Mahasiswa'})

def fraud_detection_view(request):
    form = AnomalyDetectionInputForm(request.POST or None)
    # ... (Logika lengkap view ini) ...
    return render(request, 'usecase_miko/fraud_detection_input.html', {'form': form, 'use_case_title': 'Deteksi Anomali Data Akademik'})


# --- View untuk Prediksi IP dengan Model "ENHANCED" (BARU) ---
def predict_enhanced_gpa_view(request): # Ganti nama view agar jelas
    form = EnhancedGPAPredictionForm(request.POST or None) # Gunakan form baru
    context = {'form': form, 'use_case_title': 'Prediksi IP Semester Berikutnya (Model Enhanced)'}

    if request.method == 'POST' and form.is_valid():
        if not all([enhanced_gpa_model, enhanced_gpa_config, enhanced_gpa_columns]):
            context['error_message'] = "Model Prediksi IP (Enhanced) tidak dapat dimuat. Hubungi administrator."
        else:
            try:
                input_data_raw = form.cleaned_data
                context['student_name'] = input_data_raw.get('nama_mahasiswa', 'Mahasiswa')
                
                # Kumpulkan fitur dari form, HAPUS field non-fitur seperti 'nama_mahasiswa'
                features_for_df = {k: v for k, v in input_data_raw.items() if k in enhanced_gpa_columns or k in ['gender', 'departemen']}
                
                input_df_raw = pd.DataFrame([features_for_df])
                
                # One-Hot Encode jika diperlukan (sesuaikan dengan skrip training "Enhanced")
                # Asumsi 'gender' dan 'departemen' perlu di-OHE
                categorical_features_in_form = []
                if 'gender' in input_df_raw.columns: categorical_features_in_form.append('gender')
                if 'departemen' in input_df_raw.columns: categorical_features_in_form.append('departemen')
                
                if categorical_features_in_form:
                    input_processed_ohe = pd.get_dummies(input_df_raw, columns=categorical_features_in_form)
                else:
                    input_processed_ohe = input_df_raw
                
                # Samakan kolom dengan kolom model (penting!)
                # enhanced_gpa_columns adalah daftar nama fitur SETELAH OHE dari file .pkl
                input_final = input_processed_ohe.reindex(columns=enhanced_gpa_columns, fill_value=0)
                
                # Terapkan scaler jika ada
                input_to_predict = input_final
                if enhanced_gpa_scaler:
                    input_scaled = enhanced_gpa_scaler.transform(input_final)
                    input_to_predict = pd.DataFrame(input_scaled, columns=enhanced_gpa_columns)

                # Prediksi (model ini menghasilkan angka IP)
                prediction = enhanced_gpa_model.predict(input_to_predict)
                context['prediction_result'] = round(prediction[0], 2)
                
                # Semester berikutnya (jika ada field semester di form)
                completed_sem = input_data_raw.get('semester_ke_input') 
                if completed_sem is not None:
                    context['predicting_for_semester'] = completed_sem + 1
                else:
                    context['predicting_for_semester'] = "Berikutnya"

                # Model Insights (Feature Importances untuk GradientBoosting)
                if hasattr(enhanced_gpa_model, 'feature_importances_'):
                    importances = enhanced_gpa_model.feature_importances_
                    insights = []
                    # enhanced_gpa_columns adalah nama fitur yang masuk ke model (setelah OHE dan scaling)
                    for feature_name, importance_score in zip(enhanced_gpa_columns, importances):
                        insights.append({
                            'name': feature_name.replace('_', ' ').title(),
                            'importance': round(importance_score * 100, 2) # Dalam persen
                        })
                    insights = sorted(insights, key=lambda x: x['importance'], reverse=True)
                    context['model_insights'] = insights 
                
            except Exception as e:
                context['error_message'] = f"Terjadi kesalahan saat prediksi: {e}"
                traceback.print_exc() # Cetak traceback ke konsol server untuk debug
    
    # Ganti nama template jika Anda membuat file HTML baru
    return render(request, 'usecase_miko/predict_enhanced_gpa.html', context)


# /usecase_miko/views.py

import pandas as pd
import pickle
import os
import numpy as np
from django.shortcuts import render
from django.conf import settings
import traceback

# Impor form yang akan kita gunakan
from .forms import PredictNextSemesterIPForm
# from .forms import RiskPredictionForm, AnomalyDetectionInputForm # Jika masih dipakai

# ==============================================================================
# PEMUATAN MODEL (Fokus pada Model Prediksi IP Semester 3)
# ==============================================================================
ip_sem3_model_pipeline = None
ip_sem3_original_columns = None
ip_sem3_assessment_map = None

try:
    # ### PERUBAHAN UTAMA DI SINI ###
    # Path sekarang langsung menunjuk ke folder 'ml_models' di root proyek
    MODEL_BASE_PATH = os.path.join(settings.BASE_DIR, 'ml_models') 
    
    MODEL_FILE = os.path.join(MODEL_BASE_PATH, 'best_ip_sem3_predictor_pipeline.pkl')
    COLUMNS_FILE = os.path.join(MODEL_BASE_PATH, 'original_columns_for_sem3_pred.pkl')
    ASSESSMENT_MAP_FILE = os.path.join(MODEL_BASE_PATH, 'assessment_cols_map.pkl') # Jika Anda menyimpan ini juga

    # Memuat pipeline model utama
    with open(MODEL_FILE, 'rb') as f:
        ip_sem3_model_pipeline = pickle.load(f)
    print(f"✅ Model Pipeline ('{os.path.basename(MODEL_FILE)}') berhasil dimuat.")

    # Memuat daftar kolom asli
    with open(COLUMNS_FILE, 'rb') as f:
        ip_sem3_original_columns = pickle.load(f)
    print(f"✅ File Kolom Asli ('{os.path.basename(COLUMNS_FILE)}') berhasil dimuat.")

    # Memuat Assessment Cols Map (opsional, tapi baik jika ada)
    if os.path.exists(ASSESSMENT_MAP_FILE):
        with open(ASSESSMENT_MAP_FILE, 'rb') as f: 
            ip_sem3_assessment_map = pickle.load(f)
        print(f"✅ File Assessment Map ('{os.path.basename(ASSESSMENT_MAP_FILE)}') berhasil dimuat.")
    else:
        print(f"ℹ️ File Assessment Map ('{os.path.basename(ASSESSMENT_MAP_FILE)}') tidak ditemukan, akan menggunakan default jika perlu.")
        # Anda mungkin perlu mendefinisikan ASSESSMENT_COLS_MAP default di dalam view jika file ini tidak ada
        # dan fungsi calculate_na_mk membutuhkannya. Namun, model yang kita latih terakhir
        # tidak secara eksplisit menggunakan ASSESSMENT_COLS_MAP ini di view karena
        # pengguna menginput fitur yang sudah jadi. Ini lebih untuk referensi training.

    print("✅ Semua artefak Model Prediksi IP Semester 3 berhasil dikonfigurasi.")

except FileNotFoundError as fnfe:
    print(f"❌ ERROR FileNotFoundError: Salah satu file model tidak ditemukan. Pastikan file ada di folder 'ml_models'.")
    print(f"   Detail: {fnfe}")
    # traceback.print_exc() # Uncomment untuk debug detail
except Exception as e:
    print(f"❌ ERROR Umum: Gagal memuat model Prediksi IP Semester 3: {e}")
    traceback.print_exc() # Uncomment untuk debug detail


# --- View untuk Prediksi IP Semester N+1 (misal, S3 dari S1&S2, atau S4 dari S2&S3) ---
def predict_ip_view(request):
    form = PredictNextSemesterIPForm(request.POST or None)
    context = {
        'form': form, 
        'use_case_title': 'Prediksi IP Semester Berikutnya (Estimasi Semester 4)',
        'disclaimer': 'Model ini dilatih untuk memprediksi IP Semester 3 berdasarkan data Semester 1 & 2. Hasil prediksi untuk Semester 4 adalah ekstrapolasi dan memiliki keterbatasan akurasi yang signifikan.'
    }

    if request.method == 'POST' and form.is_valid():
        # Pastikan semua artefak model yang WAJIB telah dimuat
        if not ip_sem3_model_pipeline or not ip_sem3_original_columns:
            context['error_message'] = "Model Prediksi IP tidak dapat dimuat. File pipeline atau kolom tidak ditemukan. Hubungi administrator."
        else:
            try:
                input_data_raw = form.cleaned_data
                context['student_name'] = input_data_raw.get('nama_mahasiswa', 'Mahasiswa')
                
                features_from_form = {}
                for col_name in ip_sem3_original_columns:
                    features_from_form[col_name] = input_data_raw.get(col_name, 0) 
                
                if 'gender' in ip_sem3_original_columns and 'gender' in input_data_raw: # Kolom asli sebelum OHE
                     features_from_form['gender'] = input_data_raw.get('gender')
                if 'dept_id' in ip_sem3_original_columns and 'departemen' in input_data_raw: # Kolom asli sebelum OHE
                     features_from_form['dept_id'] = input_data_raw.get('departemen')
                
                input_df = pd.DataFrame([features_from_form])
                
                try:
                    input_df_reordered = input_df[ip_sem3_original_columns]
                except KeyError as ke:
                    context['error_message'] = f"Kesalahan nama fitur saat menyusun input: {ke}. Pastikan field form dan original_columns.pkl cocok."
                    return render(request, 'usecase_miko/predict_ip_semester_3.html', context)

                prediction = ip_sem3_model_pipeline.predict(input_df_reordered)
                context['prediction_result'] = round(prediction[0], 2)
                context['predicting_for_semester'] = "4 (Estimasi)" 
                
                try:
                    regressor_step = ip_sem3_model_pipeline.named_steps['regressor']
                    preprocessor_step = ip_sem3_model_pipeline.named_steps['preprocessor']
                    
                    feature_names_transformed = []
                    # Mendapatkan nama fitur setelah transformasi dari ColumnTransformer
                    for name, trans, cols_original_subset in preprocessor_step.transformers_:
                        if hasattr(trans, 'get_feature_names_out'): # Untuk OneHotEncoder
                            # Berikan input feature names jika transformer membutuhkannya (misal OHE)
                            if name == 'cat': # Jika ini adalah transformer kategorikal
                                feature_names_transformed.extend(trans.get_feature_names_out(cols_original_subset))
                            else: # Untuk transformer lain yang mungkin punya get_feature_names_out
                                feature_names_transformed.extend(trans.get_feature_names_out())
                        elif name == 'num': # Untuk StandardScaler (atau transformer numerik lainnya)
                            feature_names_transformed.extend(cols_original_subset)
                        # Handle 'passthrough' or other transformers if necessary
                        # elif name == 'remainder' and trans == 'passthrough':
                        # feature_names_transformed.extend(preprocessor_step.feature_names_in_[cols_original_subset])


                    if hasattr(regressor_step, 'coef_'): # Untuk RidgeCV
                        coefficients = regressor_step.coef_
                        # Pastikan jumlah koefisien = jumlah nama fitur yang ditransformasi
                        if len(feature_names_transformed) == len(coefficients):
                            insights = []
                            for feature_name, coef_score in zip(feature_names_transformed, coefficients):
                                insights.append({
                                    'name': feature_name.replace('num__', '').replace('cat__', '').replace('_', ' ').title(),
                                    'importance': round(coef_score, 4)
                                })
                            insights = sorted(insights, key=lambda x: abs(x['importance']), reverse=True)
                            context['model_insights'] = insights
                            context['model_insights_type'] = 'coefficients'
                        else:
                            print(f"Peringatan: Jumlah fitur transformed ({len(feature_names_transformed)}) tidak cocok dengan jumlah koefisien ({len(coefficients)}). Insights mungkin tidak akurat.")
                            context['model_insights'] = None

                except Exception as e_insight:
                    print(f"Gagal mengambil insights model: {e_insight}")
                    # traceback.print_exc() # Uncomment untuk debug detail
                    context['model_insights'] = None
                
            except Exception as e:
                context['error_message'] = f"Terjadi kesalahan saat prediksi: {e}"
                traceback.print_exc() # Cetak traceback ke konsol server untuk debug
    
    return render(request, 'usecase_miko/predict_ip_semester_3.html', context)

# Jika Anda masih punya view lain, biarkan di sini
# def predict_risk_view(request): ...
# def fraud_detection_view(request): ...
# /usecase_miko/views.py

# /usecase_miko/views.py

import pandas as pd
import pickle
import os
import numpy as np
from django.shortcuts import render
from django.conf import settings
import traceback # Untuk debugging jika diperlukan

# Impor form yang akan kita gunakan untuk view ini
from .forms import LecturerEffectIPPredictForm 
# Jika Anda masih punya view lain yang menggunakan form lain, impor juga di sini, contoh:
# from .forms import RiskPredictionForm, AnomalyDetectionInputForm 

# ==============================================================================
# BAGIAN 1: PEMUATAN MODEL (FOKUS PADA MODEL "LECTURER EFFECT")
# ==============================================================================

lecturer_effect_model_pipeline = None
lecturer_effect_original_columns = None
# lecturer_effect_assessment_map = None # Anda bisa memuat ini jika diperlukan oleh view Anda

# --- Variabel untuk path dan nama file model "Lecturer Effect" ---
# GANTI INI SESUAI DENGAN STRUKTUR FOLDER DAN NAMA FILE ANDA
MODEL_BASE_FOLDER = 'ml_models' # Folder utama tempat subfolder model Anda berada (di root proyek)
MODEL_SUBFOLDER_NAME_LECTURER = 'ml_model_ip_lecturer_effect' # Nama subfolder untuk model ini
MODEL_FILENAME_LECTURER = 'best_lecturer_effect_pipeline.pkl'
COLUMNS_FILENAME_LECTURER = 'original_columns_lecturer_effect.pkl'
# ASSESSMENT_MAP_FILENAME_LECTURER = 'assessment_cols_map_lecturer_effect.pkl' # Jika ada

try:
    MODEL_DIR_LECTURER = os.path.join(settings.BASE_DIR, MODEL_BASE_FOLDER, MODEL_SUBFOLDER_NAME_LECTURER) 
    
    MODEL_FILE_PATH = os.path.join(MODEL_DIR_LECTURER, MODEL_FILENAME_LECTURER)
    COLUMNS_FILE_PATH = os.path.join(MODEL_DIR_LECTURER, COLUMNS_FILENAME_LECTURER)
    # ASSESSMENT_MAP_FILE_PATH = os.path.join(MODEL_DIR_LECTURER, ASSESSMENT_MAP_FILENAME_LECTURER)

    # Pengecekan Keberadaan File sebelum memuat
    if not os.path.exists(MODEL_FILE_PATH):
        raise FileNotFoundError(f"File model utama tidak ditemukan: {MODEL_FILE_PATH}")
    if not os.path.exists(COLUMNS_FILE_PATH):
        raise FileNotFoundError(f"File kolom tidak ditemukan: {COLUMNS_FILE_PATH}")
    # if not os.path.exists(ASSESSMENT_MAP_FILE_PATH): # Jika Anda menggunakan ini
    #     raise FileNotFoundError(f"File assessment map tidak ditemukan: {ASSESSMENT_MAP_FILE_PATH}")

    with open(MODEL_FILE_PATH, 'rb') as f:
        lecturer_effect_model_pipeline = pickle.load(f)
    with open(COLUMNS_FILE_PATH, 'rb') as f:
        lecturer_effect_original_columns = pickle.load(f)
    # with open(ASSESSMENT_MAP_FILE_PATH, 'rb') as f:
    #     lecturer_effect_assessment_map = pickle.load(f)

    print(f"✅ Model Prediksi IP (dengan Efek Dosen) berhasil dimuat dari {MODEL_DIR_LECTURER}.")

except FileNotFoundError as fnfe: 
    print(f"❌ ERROR FileNotFoundError: {str(fnfe)}")
    # Variabel tetap None jika file tidak ditemukan, view akan menampilkan error ke pengguna
except Exception as e:
    print(f"❌ ERROR UMUM saat memuat Model Prediksi IP (Efek Dosen): {e}")
    traceback.print_exc() 

# ==============================================================================
# BAGIAN 2: VIEW FUNCTIONS
# ==============================================================================

# --- VIEW BARU untuk Prediksi IP dengan Efek Dosen ---
def predict_ip_lecturer_effect_view(request):
    form = LecturerEffectIPPredictForm(request.POST or None)
    context = {
        'form': form, 
        'use_case_title': 'Prediksi IP Semester dengan Analisis Efek Dosen',
        'disclaimer': 'Model ini dilatih untuk memprediksi IP Semester 3 berdasarkan data detail Semester 1 & 2 (termasuk aspek dosen). Hasil prediksi untuk Semester 4 adalah ekstrapolasi dengan memasukkan data Semester 2 & 3 sebagai input, dan akurasinya mungkin terbatas.',
        'predicting_for_semester_context': 'Semester 3 (atau Estimasi Semester 4 jika input S2 & S3)' # Akan diupdate di bawah
    }

    if request.method == 'POST' and form.is_valid():
        if not lecturer_effect_model_pipeline or not lecturer_effect_original_columns:
            context['error_message'] = "Model Prediksi IP (Efek Dosen) tidak dapat dimuat. Pastikan path dan nama file benar di views.py dan file .pkl ada."
        else:
            try:
                input_data_raw = form.cleaned_data
                
                student_name_from_form = input_data_raw.pop('nama_mahasiswa', None) 
                if student_name_from_form: # Hanya tambahkan ke context jika ada isinya
                    context['student_name'] = student_name_from_form
                
                # DataFrame dari input form. Field di form HARUS SESUAI dengan original_columns.
                # Kolom 'nama_mahasiswa' sudah di-pop.
                input_df = pd.DataFrame([input_data_raw])
                
                # Pastikan urutan kolom input_df sama dengan saat training
                try:
                    input_df_reordered = input_df[lecturer_effect_original_columns]
                except KeyError as ke:
                    missing_cols = set(lecturer_effect_original_columns) - set(input_df.columns)
                    extra_cols = set(input_df.columns) - set(lecturer_effect_original_columns)
                    error_msg = f"Ketidakcocokan kolom fitur. "
                    if missing_cols: error_msg += f"Kolom diharapkan model ({missing_cols}) tidak ditemukan di form/input. "
                    if extra_cols: error_msg += f"Kolom di form ({extra_cols}) tidak diharapkan model. "
                    context['error_message'] = error_msg
                    return render(request, 'usecase_miko/predict_ip_lecturer_effect.html', context)

                # Prediksi menggunakan pipeline (sudah termasuk preprocessor)
                prediction = lecturer_effect_model_pipeline.predict(input_df_reordered)
                context['prediction_result'] = round(prediction[0], 2)
                
                # Menentukan semester yang diprediksi berdasarkan konteks (bisa disesuaikan)
                # Karena form meminta data "Semester Historis ke-1" dan "Semester Historis ke-2",
                # maka outputnya adalah untuk "Semester Historis ke-3" (atau N+1).
                # Jika pengguna menginput S2 dan S3, maka ini adalah estimasi untuk S4.
                context['predicting_for_semester_display'] = "Semester Berikutnya (Estimasi)" 
                
                # Model Insights
                insights_list_for_template = []
                try:
                    regressor_step = lecturer_effect_model_pipeline.named_steps['regressor']
                    preprocessor_step = lecturer_effect_model_pipeline.named_steps['preprocessor']
                    transformed_feature_names = preprocessor_step.get_feature_names_out()
                    
                    insights_data = None
                    if hasattr(regressor_step, 'feature_importances_'):
                        insights_data = regressor_step.feature_importances_
                        context['model_insights_type'] = 'importances'
                    elif hasattr(regressor_step, 'coef_'):
                        insights_data = regressor_step.coef_
                        context['model_insights_type'] = 'coefficients'
                    
                    if insights_data is not None:
                        for feature_name, importance_score in zip(transformed_feature_names, insights_data):
                            clean_name = feature_name.replace('num__', '').replace('cat__', '').replace('remainder__', '')
                            insights_list_for_template.append({
                                'name': clean_name.replace('_', ' ').title(),
                                'importance': round(importance_score * 100 if context.get('model_insights_type') == 'importances' else importance_score, 4)
                            })
                        # Urutkan berdasarkan nilai absolut importance/coefficient
                        insights_list_for_template = sorted(insights_list_for_template, key=lambda x: abs(x['importance']), reverse=True)
                        context['model_insights'] = insights_list_for_template
                except Exception as e_insight:
                    print(f"Gagal mengambil atau memproses insights: {e_insight}")
                    context['model_insights'] = None # Pastikan tetap None jika ada error
                
            except Exception as e:
                context['error_message'] = f"Terjadi kesalahan saat proses prediksi: {e}"
                traceback.print_exc() # Untuk debug di konsol server
    
    return render(request, 'usecase_miko/predict_ip_lecturer_effect.html', context)

# --- View lain yang mungkin masih Anda gunakan ---
# def predict_risk_view(request):
#     # ... logika view risiko ...
#     pass

# def fraud_detection_view(request):
#     # ... logika view anomali ...
#     pass

# def predict_ip_semester_view(request): # View lama jika masih ada
#     # ... logika view IP lama ...
#     pass