from django.shortcuts import render
from django.conf import settings # Untuk path yang lebih baik
import pandas as pd
import pickle
import os

from .forms import IPPredictionForm

# Path ke model dan kolom
# Menggunakan os.path.join untuk kompatibilitas lintas OS
# __file__ adalah path ke file views.py saat ini
APP_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(APP_DIR, 'ml_models', 'student_peer_group_predictor_rf.pkl')
COLUMNS_PATH = os.path.join(APP_DIR, 'ml_models', 'student_peer_group_model_columns_rf.pkl')

# Muat model dan kolom saat aplikasi Django dimulai
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model berhasil dimuat.")
except FileNotFoundError:
    model = None
    print(f"Error: File model tidak ditemukan di {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"Error saat memuat model: {e}")

try:
    with open(COLUMNS_PATH, 'rb') as f:
        model_columns = pickle.load(f)
    print("Kolom model berhasil dimuat.")
except FileNotFoundError:
    model_columns = None
    print(f"Error: File kolom tidak ditemukan di {COLUMNS_PATH}")
except Exception as e:
    model_columns = None
    print(f"Error saat memuat kolom model: {e}")


def predict_ip_view(request):
    form = IPPredictionForm()
    prediction_result = None
    error_message = None

    if not model or not model_columns:
        error_message = "Model atau file kolom tidak berhasil dimuat. Silakan periksa log server."
        return render(request, 'usecase_farhan/predict_ip.html', {'form': form, 'error_message': error_message})

    if request.method == 'POST':
        form = IPPredictionForm(request.POST)
        if form.is_valid():
            try:
                # Ambil data dari form
                data = form.cleaned_data
                
                # Buat DataFrame dari input pengguna
                # Nama kolom di sini harus sesuai dengan nama kolom asli SEBELUM get_dummies
                # yang digunakan di skrip training untuk 'departemen', 'gender', 'course_difficulty'
                input_df = pd.DataFrame({
                    'ipk_sekarang': [data['ipk_sekarang']],
                    'ip_semester_lalu': [data['ip_semester_lalu']],
                    'attendance_percentage': [data['attendance_percentage']],
                    'departemen': [data['departemen']], # Ini akan menjadi 'TI', 'SI', dll.
                    'gender': [data['gender']],         # Ini akan menjadi 'L', 'P', dll.
                    'course_difficulty': [data['course_difficulty']] # Ini akan menjadi 'low', 'medium', 'high'
                })

                # Lakukan one-hot encoding dengan cara yang sama seperti saat training
                # drop_first=True harus konsisten dengan skrip training
                # Kolom kategorikal yang di-encode adalah 'departemen', 'gender', dan 'course_difficulty'
                input_df_processed = pd.get_dummies(input_df, 
                                                    columns=['departemen', 'gender', 'course_difficulty'], 
                                                    drop_first=True) # Sesuai training script Anda

                # Sejajarkan kolom dengan kolom yang digunakan saat training
                # Ini penting untuk memastikan semua fitur ada dan dalam urutan yang benar,
                # mengisi fitur yang tidak ada di input dengan 0.
                input_df_aligned = input_df_processed.reindex(columns=model_columns, fill_value=0)
                
                # Lakukan prediksi
                prediction = model.predict(input_df_aligned)
                prediction_result = round(prediction[0], 2) # Ambil hasil pertama dan bulatkan

            except Exception as e:
                error_message = f"Terjadi kesalahan saat prediksi: {e}"
                print(f"Prediction error: {e}") # Untuk debugging di console server

    context = {
        'form': form,
        'prediction_result': prediction_result,
        'error_message': error_message,
    }
    return render(request, 'usecase_farhan/predict_ip.html', context)


from django.shortcuts import render
from django.conf import settings
import pandas as pd
import pickle
import os

# ... (mungkin sudah ada IPPredictionForm dan modelnya) ...
from .forms import CourseDifficultyForm # Tambahkan import ini

# Path untuk model Course Difficulty Classifier
APP_DIR_FARHAN = os.path.join(settings.BASE_DIR, 'ml_models') # Asumsi app 'usecase_farhan' ada di root
DIFFICULTY_MODEL_PATH = os.path.join(APP_DIR_FARHAN,  'course_difficulty_classifier_pipeline.pkl')
DIFFICULTY_RAW_FEATURES_PATH = os.path.join(APP_DIR_FARHAN, 'course_difficulty_raw_features.pkl')

difficulty_model_pipeline = None
difficulty_raw_feature_names = None

try:
    with open(DIFFICULTY_MODEL_PATH, 'rb') as f:
        difficulty_model_pipeline = pickle.load(f)
    print(f"✅ Model Difficulty Classifier '{DIFFICULTY_MODEL_PATH}' berhasil dimuat.")
except FileNotFoundError:
    print(f"❌ Error: File model Difficulty Classifier tidak ditemukan di {DIFFICULTY_MODEL_PATH}")
except Exception as e:
    print(f"❌ Error saat memuat model Difficulty Classifier: {e}")

try:
    with open(DIFFICULTY_RAW_FEATURES_PATH, 'rb') as f:
        difficulty_raw_feature_names = pickle.load(f)
    print(f"✅ Daftar fitur mentah Difficulty Classifier '{DIFFICULTY_RAW_FEATURES_PATH}' berhasil dimuat.")
except FileNotFoundError:
    print(f"❌ Error: File daftar fitur mentah Difficulty Classifier tidak ditemukan di {DIFFICULTY_RAW_FEATURES_PATH}")
except Exception as e:
    print(f"❌ Error saat memuat daftar fitur mentah Difficulty Classifier: {e}")


# ... (view predict_ip_view yang mungkin sudah ada) ...


def classify_course_difficulty_view(request):
    form = CourseDifficultyForm()
    prediction_result = None
    error_message = None

    if not difficulty_model_pipeline or not difficulty_raw_feature_names:
        error_message = "Model klasifikasi kesulitan mata kuliah atau konfigurasi fitur tidak berhasil dimuat. Silakan periksa log server."
        # Tetap render form dengan pesan error
        return render(request, 'usecase_farhan/classify_difficulty.html', { # Pastikan template path benar
            'form': form, 
            'error_message': error_message
        })

    if request.method == 'POST':
        form = CourseDifficultyForm(request.POST)
        if form.is_valid():
            try:
                input_data = {}
                for feature_name in difficulty_raw_feature_names:
                    if feature_name in form.cleaned_data:
                        input_data[feature_name] = [form.cleaned_data[feature_name]]
                    else:
                        raise KeyError(f"Fitur '{feature_name}' yang dibutuhkan model tidak ada di input form.")

                input_df = pd.DataFrame.from_dict(input_data)
                input_df = input_df[difficulty_raw_feature_names] # Reorder/select

                prediction = difficulty_model_pipeline.predict(input_df)
                prediction_result = prediction[0]
            except KeyError as ke:
                error_message = f"Input Error: Fitur yang dibutuhkan model tidak lengkap atau salah nama. Detail: {ke}"
                print(f"Prediction input key error: {ke}")
            except Exception as e:
                error_message = f"Terjadi kesalahan saat melakukan klasifikasi: {e}"
                print(f"Prediction error: {e}")

    context = {
        'form': form,
        'prediction_result': prediction_result,
        'error_message': error_message,
    }
    # Pastikan path ke template ini benar
    return render(request, 'usecase_farhan/classify_difficulty.html', context)