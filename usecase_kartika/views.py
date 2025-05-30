# usecase_kartika/views.py

from django.shortcuts import render
from django.conf import settings # Untuk settings.BASE_DIR
import pandas as pd
import numpy as np # Jika ada operasi numpy, meskipun imputer form bisa handle None
import pickle
import os
import shap # Pastikan shap diinstal di environment Django Anda: pip install shap

from .forms import RiskAssessmentForm # Pastikan RiskAssessmentForm ada di forms.py aplikasi ini

# --- Path ke Model dan Komponen Lainnya (dari root proyek) ---
MODEL_STORAGE_DIR = os.path.join(settings.BASE_DIR, 'ml_models')
PIPELINE_PATH = os.path.join(MODEL_STORAGE_DIR, 'kartika_risk_assessment_pipeline.pkl')
RAW_FEATURES_LIST_PATH = os.path.join(MODEL_STORAGE_DIR, 'kartika_risk_raw_features.pkl')
EXPLAINER_PATH = os.path.join(MODEL_STORAGE_DIR, 'kartika_risk_shap_explainer.pkl')
PROCESSED_FEATURES_PATH = os.path.join(MODEL_STORAGE_DIR, 'kartika_risk_processed_features.pkl')

# --- Muat Model, Explainer, dan Daftar Fitur saat Aplikasi Dimuat ---
risk_pipeline = None
raw_feature_columns = None # Ini adalah nama kolom MENTAH yang diharapkan form
shap_explainer = None
processed_feature_names = None # Ini adalah nama kolom SETELAH preprocessing (untuk SHAP)
initialization_error = None # Untuk menyimpan pesan error saat inisialisasi

try:
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(f"File pipeline model tidak ditemukan: {PIPELINE_PATH}")
    with open(PIPELINE_PATH, 'rb') as f:
        risk_pipeline = pickle.load(f)
    print(f"✅ [Kartika] Pipeline '{PIPELINE_PATH}' berhasil dimuat.")

    if not os.path.exists(RAW_FEATURES_LIST_PATH):
        raise FileNotFoundError(f"File daftar fitur mentah tidak ditemukan: {RAW_FEATURES_LIST_PATH}")
    with open(RAW_FEATURES_LIST_PATH, 'rb') as f:
        raw_feature_columns = pickle.load(f)
    print(f"✅ [Kartika] Daftar fitur mentah '{RAW_FEATURES_LIST_PATH}' berhasil dimuat: {raw_feature_columns}")

    # Komponen SHAP (opsional, hanya dimuat jika ada dan akan digunakan)
    if os.path.exists(EXPLAINER_PATH) and os.path.exists(PROCESSED_FEATURES_PATH):
        with open(EXPLAINER_PATH, 'rb') as f:
            shap_explainer = pickle.load(f)
        print(f"✅ [Kartika] SHAP Explainer '{EXPLAINER_PATH}' berhasil dimuat.")
        
        with open(PROCESSED_FEATURES_PATH, 'rb') as f:
            processed_feature_names = pickle.load(f)
        print(f"✅ [Kartika] Nama fitur terproses '{PROCESSED_FEATURES_PATH}' berhasil dimuat.")
    else:
        print("⚠️ [Kartika] File SHAP explainer atau fitur terproses tidak ditemukan. Penjelasan SHAP tidak akan tersedia.")
        # Tidak perlu error, penjelasan SHAP adalah fitur tambahan

except FileNotFoundError as fnf_e:
    error_msg = f"Kesalahan Inisialisasi Model [Kartika]: {fnf_e}. Pastikan model sudah dilatih dan path benar."
    print(f"❌ {error_msg}")
    initialization_error = error_msg
except Exception as e:
    error_msg = f"Kesalahan Umum Inisialisasi Model [Kartika]: {e}"
    print(f"❌ {error_msg}")
    initialization_error = error_msg


def assess_risk_view(request):
    form = RiskAssessmentForm()
    prediction_status = None
    prediction_proba = None
    input_details_for_template = None # Ini akan dikirim ke template
    shap_plot_html = None
    error_message = initialization_error # Tampilkan error inisialisasi jika ada

    if request.method == 'POST':
        form = RiskAssessmentForm(request.POST)
        if form.is_valid():
            if not risk_pipeline or not raw_feature_columns:
                error_message = "Model penilaian risiko atau konfigurasi fitur tidak berhasil dimuat. Tidak dapat melakukan prediksi."
                # Log ini juga akan membantu debugging jika initialization_error tidak ter-set dengan benar
                print("[Kartika] Gagal melakukan prediksi karena model/fitur tidak dimuat saat startup.")
            else:
                try:
                    cleaned_data_from_form = form.cleaned_data
                    # Simpan data yang dibersihkan untuk ditampilkan kembali di template
                    input_details_for_template = cleaned_data_from_form.copy()

                    # Buat DataFrame dari input pengguna, pastikan urutan kolom sesuai raw_feature_columns
                    # raw_feature_columns adalah daftar nama fitur MENTAH yang diharapkan oleh pipeline.
                    # Kolom-kolom ini harus ada di cleaned_data_from_form.
                    
                    # Persiapkan input_dict untuk DataFrame, hanya dengan fitur yang ada di raw_feature_columns
                    input_dict_for_df = {}
                    for feature_name in raw_feature_columns:
                        if feature_name in cleaned_data_from_form:
                            input_dict_for_df[feature_name] = [cleaned_data_from_form[feature_name]]
                        else:
                            # Jika fitur tidak ada di form, dan pipeline tidak menghandle (misal dgn imputer),
                            # ini akan error. Pastikan form Anda meminta semua fitur di raw_feature_columns.
                            # Atau, jika pipeline meng-handle missing values, biarkan (SimpleImputer akan bekerja).
                            # Untuk fitur skor yang opsional (required=False di form) dan diisi None oleh form,
                            # SimpleImputer di pipeline akan mengisinya dengan median.
                            input_dict_for_df[feature_name] = [np.nan] # Biarkan imputer pipeline yg handle
                            print(f"PERINGATAN [Kartika]: Fitur '{feature_name}' tidak ada di form, diisi NaN untuk diproses imputer.")


                    input_df = pd.DataFrame(input_dict_for_df, columns=raw_feature_columns) # Pastikan urutan kolom
                    
                    # Prediksi menggunakan pipeline (sudah termasuk preprocessing)
                    prediction_raw = risk_pipeline.predict(input_df)[0] # Ambil hasil pertama dari array
                    prediction_proba_raw = risk_pipeline.predict_proba(input_df)[0][1] # Probabilitas kelas 1 (Berisiko)

                    prediction_status = "Berisiko Tinggi" if prediction_raw == 1 else "Aman"
                    prediction_proba = round(prediction_proba_raw * 100, 2) # Dalam persen

                    # Generate SHAP explanation (jika explainer dan fitur terproses ada)
                    if shap_explainer and processed_feature_names and risk_pipeline:
                        try:
                            # Dapatkan data yang sudah ditransformasi oleh preprocessor di dalam pipeline
                            preprocessor_step = risk_pipeline.named_steps.get('preprocessor')
                            if preprocessor_step:
                                transformed_input_array = preprocessor_step.transform(input_df)
                                # Buat DataFrame dari hasil transformasi dengan nama kolom yang benar
                                transformed_input_df = pd.DataFrame(transformed_input_array, columns=processed_feature_names)
                                
                                shap_values_instance = shap_explainer.shap_values(transformed_input_df)
                                
                                # Pastikan shap_values_instance adalah untuk kelas positif (risiko) jika model klasifikasi biner
                                # Untuk XGBClassifier, shap_values bisa return list (jika multiclass) atau array (jika biner)
                                # Jika outputnya adalah list (satu array per kelas), ambil untuk kelas positif (kelas 1)
                                shap_values_for_positive_class = shap_values_instance
                                if isinstance(shap_values_instance, list) and len(shap_values_instance) > 1:
                                     shap_values_for_positive_class = shap_values_instance[1]


                                # Membuat plot force untuk satu instance
                                # shap.initjs() # Panggil di base template atau di sini (jika belum global)
                                force_plot = shap.force_plot(
                                    shap_explainer.expected_value, # Atau expected_value[1] jika list
                                    shap_values_for_positive_class[0,:], # SHAP values untuk instance pertama, kelas positif
                                    transformed_input_df.iloc[0,:],    # Fitur instance pertama yang sudah ditransformasi
                                    feature_names=processed_feature_names, # Gunakan nama fitur yang sudah diproses
                                    matplotlib=False, # Agar menghasilkan objek yang bisa di-render ke HTML
                                    show=False
                                )
                                shap_plot_html = f"<div class='shap-plot'>{force_plot.html()}</div>"
                            else:
                                print("⚠️ [Kartika] Langkah 'preprocessor' tidak ditemukan di pipeline untuk SHAP.")
                                shap_plot_html = "<p class='text-muted'>Tidak dapat memproses data untuk penjelasan SHAP.</p>"
                        except Exception as shap_e:
                            print(f"⚠️ Error saat membuat penjelasan SHAP [Kartika]: {shap_e}")
                            shap_plot_html = "<p class='text-muted'>Tidak dapat menampilkan penjelasan detail karena error.</p>"

                except KeyError as ke:
                    error_message = f"Kesalahan Input [Kartika]: Fitur yang dibutuhkan model tidak lengkap atau salah nama. Pastikan form sesuai dengan fitur model. Detail: {ke}"
                    print(f"Prediction input key error [Kartika]: {ke}")
                except Exception as e:
                    error_message = f"Terjadi kesalahan saat melakukan prediksi [Kartika]: {type(e).__name__} - {e}"
                    print(f"Prediction error [Kartika]: {type(e).__name__} - {e}")
        else: # form not valid
            error_message = "Input tidak valid. Harap periksa kembali data yang Anda masukkan."
            # Pesan error per field akan otomatis ditampilkan oleh Django di template
            print(f"[Kartika] Form errors: {form.errors.as_json()}")

    context = {
        'form': form,
        'prediction_status': prediction_status,
        'prediction_proba': prediction_proba,
        'input_details': input_details_for_template, # Kirim data yang sudah dibersihkan
        'shap_plot_html': shap_plot_html,
        'error_message': error_message,
    }
    # Pastikan nama aplikasi adalah 'usecase_kartika' jika template ada di 'usecase_kartika/templates/usecase_kartika/'
    return render(request, 'usecase_kartika/risk_assessment.html', context)