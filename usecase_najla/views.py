from django.shortcuts import render
from django.conf import settings
import pandas as pd
import pickle
import os
import plotly.express as px
from .forms import AttendancePredictionForm

# Impor model Django Anda (sesuaikan 'main_app', 'student', 'department' jika perlu)
try:
    from student.models import Student 
    from fedst7_app.models import Course, Semester 
    from .models import PredictionRecord # Dari usecase_najla.models
except ImportError as e:
    Student = PredictionRecord = Course = Semester = None
    print(f"WARNING: Django models tidak bisa diimpor di usecase_najla.views: {e}")

# --- Path ke Model dan Kolom Fitur ---
MODEL_PIPELINE_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'najla_attendance_predictor_pipeline.pkl')
RAW_FEATURES_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'najla_attendance_model_raw_features.pkl')

# --- Muat Model Pipeline dan Daftar Fitur Mentah ---
model_pipeline = None
raw_feature_names = None
initialization_error = None

try:
    with open(MODEL_PIPELINE_PATH, 'rb') as f: model_pipeline = pickle.load(f)
    print(f"✅ Najla Model pipeline '{MODEL_PIPELINE_PATH}' berhasil dimuat.")
except FileNotFoundError:
    error_msg = f"❌ Error: File model pipeline Najla tidak ditemukan di {MODEL_PIPELINE_PATH}"
    print(error_msg)
    if initialization_error is None: initialization_error = error_msg
except Exception as e:
    error_msg = f"❌ Error saat memuat model pipeline Najla: {e}"
    print(error_msg)
    if initialization_error is None: initialization_error = error_msg

try:
    with open(RAW_FEATURES_PATH, 'rb') as f: raw_feature_names = pickle.load(f)
    print(f"✅ Daftar fitur mentah Najla '{RAW_FEATURES_PATH}' berhasil dimuat: {raw_feature_names}")
except FileNotFoundError:
    error_msg = f"❌ Error: File daftar fitur mentah Najla tidak ditemukan di {RAW_FEATURES_PATH}"
    print(error_msg)
    if initialization_error is None: initialization_error = error_msg
except Exception as e:
    error_msg = f"❌ Error saat memuat daftar fitur mentah Najla: {e}"
    print(error_msg)
    if initialization_error is None: initialization_error = error_msg


def predict_attendance_view(request):
    form = AttendancePredictionForm()
    prediction = None
    student_name_display = "Input Pengguna" # Default
    chart = None
    error_message = initialization_error

    if request.method == 'POST':
        form = AttendancePredictionForm(request.POST)
        if form.is_valid():
            if not model_pipeline or not raw_feature_names:
                error_message = "Model prediksi atau konfigurasi fitur tidak berhasil dimuat. Tidak dapat melakukan prediksi."
            else:
                try:
                    data = form.cleaned_data
                    student_name_display = data.get('name', 'Input Pengguna')

                    input_features = {}
                    # Membangun dictionary input_features berdasarkan raw_feature_names
                    # Ini HARUS sesuai dengan apa yang ada di forms.py dan skrip training
                    
                    # Asumsi 'name' di form adalah untuk stu_id jika stu_id ada di raw_feature_names
                    if 'stu_id' in raw_feature_names:
                        try:
                            # Coba konversi input 'name' ke integer untuk stu_id
                            input_features['stu_id'] = [int(data['name'])]
                        except ValueError:
                            error_message = "Input 'Nama atau ID Mahasiswa' harus berupa ID Mahasiswa (angka)."
                            raise ValueError(error_message) # Hentikan proses jika stu_id tidak valid
                    
                    # Fitur lain dari form
                    direct_mapping_features = ['average_score', 'grade', 'gender', 'dept_id', 'age']
                    for f_name in direct_mapping_features:
                        if f_name in raw_feature_names and f_name in data:
                            input_features[f_name] = [data[f_name]]
                        elif f_name in raw_feature_names and f_name not in data:
                             raise KeyError(f"Fitur '{f_name}' dari model tidak ada di form.cleaned_data. Pastikan forms.py sudah benar.")


                    # course_id dan semester_id dari ModelChoiceField adalah instance model
                    if 'course_id' in raw_feature_names and isinstance(data.get('course_id'), Course):
                        input_features['course_id'] = [data['course_id'].course_id]
                    elif 'course_id' in raw_feature_names: # Jika inputnya sudah ID (misal dari IntegerField)
                         input_features['course_id'] = [data['course_id']]


                    if 'semester_id' in raw_feature_names and isinstance(data.get('semester_id'), Semester):
                        input_features['semester_id'] = [data['semester_id'].semester_id]
                    elif 'semester_id' in raw_feature_names:
                         input_features['semester_id'] = [data['semester_id']]


                    # Pastikan semua fitur di raw_feature_names ada di input_features
                    for feature_name_model in raw_feature_names:
                        if feature_name_model not in input_features:
                            raise KeyError(f"Fitur model '{feature_name_model}' tidak berhasil disiapkan dari input form. Cek logika view dan forms.py.")
                    
                    input_df = pd.DataFrame.from_dict(input_features)
                    input_df = input_df[raw_feature_names] # Pastikan urutan kolom

                    pred_value = model_pipeline.predict(input_df)
                    prediction = round(max(0, min(100, pred_value[0])), 2) # Batasi dan bulatkan

                    # Simpan prediksi
                    if PredictionRecord and isinstance(data.get('course_id'), Course) and isinstance(data.get('semester_id'), Semester):
                        try:
                            PredictionRecord.objects.create(
                                name_or_stu_id=student_name_display,
                                average_score=data['average_score'],
                                grade=data['grade'],
                                course_id_input=data['course_id'].course_id,
                                semester_id_input=data['semester_id'].semester_id,
                                predicted_attendance=prediction
                            )
                            print("✅ Hasil prediksi disimpan.")
                        except Exception as db_e:
                            print(f"❌ Gagal menyimpan prediksi: {db_e}")
                    
                    # Buat chart
                    fig = px.bar(
                        x=['Prediksi Kehadiran'], y=[prediction],
                        title=f'Prediksi Kehadiran untuk {student_name_display}',
                        labels={'y': 'Persentase (%)', 'x': ''},
                        text=[f"{prediction:.1f}%"], range_y=[0, 100]
                    )
                    fig.update_traces(marker_color='#4e73df', textposition='outside')
                    chart = fig.to_html(full_html=False, include_plotlyjs='cdn')

                except KeyError as ke:
                    error_message = f"Kesalahan Input: {ke}"
                except ValueError as ve: # Untuk error konversi stu_id
                    error_message = str(ve)
                except Exception as e:
                    error_message = f"Terjadi kesalahan: {type(e).__name__} - {e}"
                
                if error_message: print(f"Prediction View Error: {error_message}")

        else: # form not valid
            error_message = "Input tidak valid. Silakan periksa kembali."
            print(f"Form errors: {form.errors.as_json()}")


    context = {
        'form': form,
        'prediction': prediction,
        'chart': chart,
        'name': student_name_display,
        'error_message': error_message,
    }
    return render(request, 'usecase_najla/attendance_prediction.html', context)