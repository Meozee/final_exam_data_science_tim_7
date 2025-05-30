# usecase_nada/ml_development/MLmodel.py

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from django.conf import settings
from django.db import connection # PENTING: Untuk koneksi database Django

# --- Konfigurasi Mapping Implisit Course ke Karir ---
# Ini adalah mapping yang disepakati: course_id -> nama karir.
# ANDA HARUS menyesuaikan ini dengan course_id yang ada di database Anda
# dan karir yang relevan yang ingin Anda prediksi.
COURSE_CAREER_MAPPING = {
    1: 'Cyber Security', 2: 'Network Administrator', 3: 'Database Administrator',
    4: 'Software Developer', 5: 'IT Consultant', 6: 'System Analyst',
    7: 'Web Developer', 8: 'Mobile Developer', 9: 'UI/UX Designer',
    10: 'Cloud Engineer', 11: 'Data Scientist', 12: 'Business Analyst',
    13: 'Data Engineer', 14: 'Financial Analyst', 15: 'Accounting',
    16: 'Marketing Specialist', 17: 'Human Resources', 18: 'Project Manager',
    19: 'Operations Manager', 20: 'Supply Chain Manager',
    # Tambahkan lebih banyak mapping sesuai kebutuhan Anda
}

# --- Fungsi untuk Membuat Dummy Data (Fallback) ---
def create_dummy_training_data():
    print("🔄 Creating dummy training data as a fallback...")
    # Data dummy ini harus mencerminkan struktur data setelah agregasi
    # dan memiliki label 'actual_career' yang valid (bukan 'Other' jika 'Other' difilter)
    # Pastikan course_id dalam dummy data ada di COURSE_CAREER_MAPPING jika ingin konsisten,
    # atau sediakan 'actual_career' secara langsung.
    dummy_data = {
        'stu_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135],
        'gender': ['female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female'],
        'dept_id': [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1], # 1=IT, 2=Business
        'course_id_1': [11, 13, 11, 15, 1, 11, 13, 15, 1, 11, 15, 11, 13, 15, 1, 11, 13, 15, 1, 11, 4, 7, 10, 14, 18, 20, 5, 12, 16, 17, 19, 3, 6, 9, 2],
        'grade_c1': [85, 75, 90, 60, 88, 72, 95, 65, 80, 70, 40, 85, 70, 50, 92, 78, 60, 80, 75, 55, 90, 85, 88, 75, 92, 80, 78, 70, 65, 80, 72, 90, 82, 75, 68],
        'attendance_c1': [90, 80, 95, 70, 92, 85, 98, 75, 88, 82, 50, 90, 78, 60, 95, 80, 70, 85, 88, 65, 95, 90, 92, 80, 95, 85, 80, 75, 70, 85, 78, 92, 88, 80, 72],
        'course_id_2': [13, 11, 13, 11, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 12, 8, 5, 10, 19, 16, 18, 4, 7, 3, 6, 9, 2, 14, 17],
        'grade_c2': [70, 80, 75, 50, 80, 65, 85, 45, 75, 60, 30, 78, 65, 40, 85, 70, 55, 70, 68, 50, 75, 80, 70, 65, 80, 72, 60, 78, 55, 68, 58, 80, 70, 60, 70],
        'attendance_c2': [80, 85, 82, 60, 88, 75, 90, 55, 82, 70, 40, 85, 70, 50, 90, 75, 65, 80, 75, 60, 85, 90, 80, 75, 90, 82, 70, 85, 65, 78, 68, 90, 80, 70, 75],
        'course_id_3': [15, 15, 15, 13, 15, 15, 15, 11, 15, 15, 13, 15, 15, 11, 15, 15, 15, 11, 15, 15, 7, 14, 17, 18, 20, 5, 12, 16, 19, 4, 8, 10, 3, 6, 9],
        'grade_c3': [60, 65, 55, 30, 70, 50, 75, 20, 65, 40, 20, 60, 50, 30, 70, 55, 40, 60, 58, 45, 70, 60, 65, 50, 75, 68, 50, 55, 45, 60, 52, 70, 62, 58, 48],
        'attendance_c3': [70, 75, 65, 40, 80, 60, 85, 30, 75, 50, 30, 70, 60, 40, 80, 65, 50, 70, 68, 55, 80, 70, 75, 60, 85, 78, 60, 65, 55, 70, 62, 80, 72, 68, 58],
        'actual_career': [ # Pastikan nilai-nilai ini valid dan tidak semua 'Other'
            'Data Scientist', 'Data Engineer', 'Data Scientist', 'Accounting', 'Cyber Security',
            'Data Scientist', 'Data Engineer', 'Accounting', 'Cyber Security', 'Data Scientist',
            'Accounting', 'Data Scientist', 'Data Engineer', 'Accounting', 'Cyber Security',
            'Data Scientist', 'Data Engineer', 'Accounting', 'Cyber Security', 'Data Scientist',
            'Software Developer', 'Financial Analyst', 'Marketing Specialist', 'Project Manager', 'Operations Manager',
            'IT Consultant', 'Business Analyst', 'Human Resources', 'Supply Chain Manager', 'Database Administrator',
            'System Analyst', 'Cloud Engineer', 'Network Administrator', 'UI/UX Designer', 'Web Developer'
        ]
    }
    df_dummy = pd.DataFrame(dummy_data)
    print(f"✅ Dummy data created with {len(df_dummy)} rows.")
    return df_dummy

# Fungsi untuk melatih dan menyimpan model
def train_and_save_career_prediction_model():
    print("🚀 Starting ML model training process for Career Prediction...")

    df = None # Inisialisasi df

    # --- 1. Ekstraksi Data Historis dari Database PostgreSQL ---
    print("\n---  tahap 1: Ekstraksi Data dari Database ---")
    sql_query = """
    SELECT
        s.stu_id,
        s.gender,
        s.dept_id,
        e.course_id,
        e.grade, -- Pastikan kolom ini ada dan berisi nilai numerik yang bisa di-cast ke int/float
        a.attendance_percentage -- Pastikan kolom ini ada dan berisi nilai numerik
    FROM
        public.student s
    JOIN
        public.enrollment e ON s.stu_id = e.stu_id
    JOIN
        public.attendance a ON e.enroll_id = a.enroll_id
    ORDER BY s.stu_id, e.semester_id, e.course_id;
    """
    try:
        with connection.cursor() as cursor:
            raw_df = pd.read_sql_query(sql_query, connection)
        print(f"✅ Successfully loaded {len(raw_df)} rows from PostgreSQL.")

        if raw_df.empty:
            print("⚠️ Warning: No data loaded from database. Raw data is empty.")
            df = create_dummy_training_data()
        else:
            # --- AGREGASI DATA HISTORIS ---
            print("\n--- Tahap 2: Agregasi Data Historis ---")
            processed_data = []
            for stu_id, group in raw_df.groupby('stu_id'):
                group = group.sort_values(by=['grade', 'course_id'], ascending=[False, True])
                student_features = {
                    'stu_id': stu_id,
                    'gender': group['gender'].iloc[0],
                    'dept_id': group['dept_id'].iloc[0],
                }
                for i in range(3):
                    if i < len(group):
                        student_features[f'course_id_{i+1}'] = group['course_id'].iloc[i]
                        student_features[f'grade_c{i+1}'] = int(group['grade'].iloc[i]) # Pastikan tipe data grade numerik
                        student_features[f'attendance_c{i+1}'] = int(group['attendance_percentage'].iloc[i]) # Pastikan tipe data attendance numerik
                    else:
                        student_features[f'course_id_{i+1}'] = 0
                        student_features[f'grade_c{i+1}'] = 0
                        student_features[f'attendance_c{i+1}'] = 0
                
                # --- PENTING: Menambahkan LABEL 'actual_career' ---
                main_course_id = group['course_id'].iloc[0] if not group.empty else None
                student_features['actual_career'] = COURSE_CAREER_MAPPING.get(main_course_id, 'Other')
                processed_data.append(student_features)
            
            df_processed = pd.DataFrame(processed_data)
            print(f"📊 Data aggregated. {len(df_processed)} students processed.")
            
            # Filter out students with 'Other' career
            # PERHATIAN: Jika COURSE_CAREER_MAPPING Anda tidak mencakup banyak course_id,
            # atau jika data mahasiswa Anda tidak memiliki course_id yang terpetakan,
            # langkah ini bisa menghilangkan banyak (atau semua) data.
            df = df_processed[df_processed['actual_career'] != 'Other']
            print(f" lọc Setelah memfilter karir 'Other', tersisa {len(df)} baris data.")
            
            if df.empty:
                print("⚠️ Warning: Processed data is empty after filtering 'Other' careers. This might be due to COURSE_CAREER_MAPPING coverage.")
                df = create_dummy_training_data()
                print("🔄 Using dummy data because real data became empty after processing.")

    except Exception as e:
        print(f"❌ Error loading or processing data from PostgreSQL: {e}")
        print("🔄 Falling back to dummy data for training due to an error.")
        df = create_dummy_training_data()

    if df is None or df.empty:
        print("❌ FATAL: DataFrame is still None or empty even after trying dummy data. Cannot proceed with training.")
        return

    print(f"จำนวน Total data yang akan digunakan untuk training: {len(df)} baris.")
    print("Contoh data yang akan dilatih (5 baris pertama):")
    print(df.head())

    # --- 3. Praproses Fitur ---
    print("\n--- Tahap 3: Praproses Fitur ---")
    # Konversi ID numerik ke string untuk OneHotEncoding
    # Pastikan kolom ini ada di df Anda. Jika menggunakan dummy, mereka sudah ada.
    categorical_to_convert = ['course_id_1', 'course_id_2', 'course_id_3', 'dept_id']
    for col in categorical_to_convert:
        if col in df.columns:
            df[col] = df[col].astype(str)
        else:
            print(f"⚠️ Warning: Kolom kategorikal '{col}' tidak ditemukan di DataFrame. Mungkin akan error di preprocessor.")


    numeric_features = [col for col in df.columns if 'grade_c' in col or 'attendance_c' in col]
    # Pastikan kolom gender juga ada, jika tidak ada bisa error
    if 'gender' not in df.columns:
        print("⚠️ Warning: Kolom 'gender' tidak ditemukan. Menggunakan list fitur kategorikal tanpa 'gender'.")
        categorical_features = [col for col in categorical_to_convert if col in df.columns]
    else:
         categorical_features = [col for col in categorical_to_convert if col in df.columns] + ['gender']


    # Cek apakah fitur numerik dan kategorikal benar-benar ada di DataFrame
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]

    if not numeric_features and not categorical_features:
        print("❌ FATAL: Tidak ada fitur numerik maupun kategorikal yang teridentifikasi. Tidak bisa melanjutkan training.")
        return
    
    print(f"Fitur Numerik: {numeric_features}")
    print(f"Fitur Kategorikal: {categorical_features}")

    transformers_list = []
    if numeric_features:
        transformers_list.append(('num', StandardScaler(), numeric_features))
    if categorical_features:
        transformers_list.append(('cat', OneHotEncoder(handle_unknown='ignore', drop='first' if len(categorical_features) > 1 else None), categorical_features))

    if not transformers_list:
        print("❌ FATAL: Tidak ada transformer yang bisa dibuat (tidak ada fitur numerik atau kategorikal).")
        return
        
    preprocessor = ColumnTransformer(transformers=transformers_list, remainder='drop') # 'drop' sisa kolom seperti stu_id

    # --- 4. Bangun Pipeline Model ML ---
    print("\n--- Tahap 4: Membangun dan Melatih Model ---")
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', LogisticRegression(max_iter=1000, solver='liblinear', random_state=42))]) # Mengganti solver untuk mengatasi potensi masalah konvergensi

    # Pastikan 'stu_id' dan 'actual_career' tidak masuk ke fitur X
    # Kolom yang akan digunakan sebagai fitur adalah gabungan numeric_features dan categorical_features
    feature_columns = numeric_features + categorical_features
    X = df[feature_columns]
    y = df['actual_career']

    if X.empty or y.empty:
        print("❌ FATAL: Fitur (X) atau target (y) kosong. Tidak bisa melatih model.")
        return

    try:
        model_pipeline.fit(X, y)
        print("✅ Model training complete.")
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        return

    # --- 5. Simpan Model ke Lokasi Pusat `ml_models/` ---
    print("\n--- Tahap 5: Menyimpan Model ---")
    try:
        # Pastikan settings.BASE_DIR terdefinisi jika menggunakan Django context
        # Jika tidak, tentukan path secara manual atau relatif.
        if hasattr(settings, 'BASE_DIR'):
            model_dir_base = settings.BASE_DIR
        else:
            # Fallback jika settings.BASE_DIR tidak ada (misal dijalankan di luar Django sepenuhnya)
            model_dir_base = os.path.dirname(os.path.abspath(__file__)) # Direktori skrip ini
            print(f"⚠️ Warning: settings.BASE_DIR tidak ditemukan. Model akan disimpan relatif terhadap direktori skrip: {model_dir_base}")

        model_dir = os.path.join(model_dir_base, 'ml_models_career') # Nama folder berbeda untuk menghindari konflik
        os.makedirs(model_dir, exist_ok=True)
        print(f"Direktori model: {model_dir}")

        model_path = os.path.join(model_dir, 'nada_career_predictor.pkl')
        with open(model_path, 'wb') as file:
            pickle.dump(model_pipeline, file)
        print(f"✅ Model ML berhasil disimpan ke: {model_path}")

        features_path = os.path.join(model_dir, 'nada_career_predictor_features.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump(feature_columns, f) # Simpan feature_columns yang digunakan
        print(f"✅ Nama fitur berhasil disimpan ke: {features_path}")
        print("\n🎉 Proses Selesai! 🎉")

    except AttributeError as e:
        print(f"❌ Error: Atribut Django 'settings.BASE_DIR' mungkin tidak tersedia. Pastikan skrip dijalankan dalam konteks Django atau sesuaikan path penyimpanan model. Detail: {e}")
    except IOError as e:
        print(f"❌ Error: Tidak bisa menulis file model atau fitur. Periksa izin folder atau path. Detail: {e}")
    except Exception as e:
        print(f"❌ Error saat menyimpan model: {e}")


# --- Cara Menjalankan Skrip Ini ---
# 1. Jika Anda memiliki proyek Django:
#    - Cara terbaik adalah membuat management command Django.
#      Contoh: python manage.py train_career_model
#    - Atau, jika ingin menjalankan langsung (pastikan path Django & settings benar):
#      python path/to/your/MLmodel.py
#
# 2. Pastikan file settings Django Anda (misal: 'final_exam_data_science_tim_7.settings')
#    dapat diakses oleh Python (mungkin perlu menambahkan path proyek ke PYTHONPATH).

if __name__ == '__main__':
    print("🏁 Menjalankan skrip sebagai program utama...")
    
    # Cek apakah ini dijalankan dalam konteks Django yang sudah di-setup
    # (misalnya dari manage.py shell_plus atau ekstensi Jupyter yang setup Django)
    if not settings.configured:
        print("🛠️ Mencoba mengkonfigurasi Django secara manual...")
        # GANTI 'final_exam_data_science_tim_7.settings' dengan nama file settings proyek Anda yang sebenarnya
        django_settings_module = os.environ.get('DJANGO_SETTINGS_MODULE', 'final_exam_data_science_tim_7.settings')
        print(f"Menggunakan DJANGO_SETTINGS_MODULE: {django_settings_module}")
        
        try:
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', django_settings_module)
            import django
            django.setup()
            print("✅ Django berhasil dikonfigurasi.")
        except ImportError:
            print("❌ Error: Django tidak terinstal atau tidak ditemukan. Tidak bisa melanjutkan tanpa Django untuk koneksi database dan settings.")
            exit()
        except Exception as e:
            print(f"❌ Error saat mengkonfigurasi Django: {e}. Pastikan DJANGO_SETTINGS_MODULE sudah benar dan proyek Django Anda valid.")
            print("   Jika 'settings.BASE_DIR' tidak ditemukan, pastikan itu terdefinisi di file settings.py Anda.")
            exit()
    else:
        print("✅ Django sudah terkonfigurasi.")
        
    train_and_save_career_prediction_model()