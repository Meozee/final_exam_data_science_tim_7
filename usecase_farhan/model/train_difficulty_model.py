import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Menggunakan Classifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

print("🚀 Melatih Model Klasifikasi Kesulitan Mata Kuliah 🚀")

# --- 1. Koneksi ke Database ---
DB_USER = "postgres"
DB_PASSWORD = "DBmiko" # GANTI DENGAN PASSWORD ANDA
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "dbexam"
try:
    db_engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    print("✅ Koneksi database berhasil")
except Exception as e:
    print(f"❌ Koneksi gagal: {e}"); exit()

# --- 2. Pengambilan Data Mentah ---
print("\n🔄 Mengambil data mentah dari database...")
try:
    course_df = pd.read_sql("SELECT course_id, dept_id FROM public.course", db_engine) # Ambil dept_id penyelenggara MK
    course_difficulty_df = pd.read_sql("SELECT course_id, difficulty_level FROM public.course_difficulty", db_engine) # Ini target
    enrollment_df = pd.read_sql("SELECT course_id, grade FROM public.enrollment WHERE grade IS NOT NULL", db_engine)
    assessment_df = pd.read_sql("SELECT enroll_id FROM public.assessment", db_engine) # Hanya untuk count
    # Gabungkan assessment dengan enrollment untuk mendapatkan course_id
    enrollment_for_assessment_df = pd.read_sql("SELECT enroll_id, course_id FROM public.enrollment", db_engine)
    assessment_df = pd.merge(assessment_df, enrollment_for_assessment_df, on='enroll_id', how='left')

    # Ambil SKS jika ada tabelnya, jika tidak, kita bisa asumsikan atau abaikan
    # Misal, kita asumsikan SKS ada di tabel course atau kita buat nilai default
    # Untuk contoh ini, kita tidak sertakan SKS kecuali Anda memiliki tabelnya.
    
    print("✅ Data mentah berhasil diambil.")
except Exception as e:
    print(f"❌ Gagal mengambil data mentah: {e}"); exit()

# --- 3. Feature Engineering & Penggabungan Data ---
print("\n🔄 Melakukan feature engineering & penggabungan data...")
try:
    # Target variable
    df_target = course_difficulty_df.dropna(subset=['difficulty_level'])
    if df_target.empty:
        print("❌ Tidak ada data difficulty_level yang valid. Tidak bisa melatih model.")
        exit()

    # Fitur 1: Rata-rata nilai per mata kuliah
    avg_grade_per_course = enrollment_df.groupby('course_id')['grade'].mean().reset_index()
    avg_grade_per_course.rename(columns={'grade': 'average_grade_course'}, inplace=True)

    # Fitur 2: Jumlah asesmen per mata kuliah
    # Kita hitung jumlah unik enroll_id per course_id di assessment, lalu count per course_id
    # Ini lebih mengarah ke jumlah mahasiswa yang di-assess daripada jumlah jenis assessment.
    # Untuk jumlah jenis assessment, perlu skema assessment_type per course_id
    # Mari kita hitung jumlah record assessment per course_id sebagai proxy
    assessment_count_per_course = assessment_df.groupby('course_id')['enroll_id'].count().reset_index()
    assessment_count_per_course.rename(columns={'enroll_id': 'assessment_count_course'}, inplace=True)


    # Gabungkan fitur dengan target
    df_model = pd.merge(df_target, course_df[['course_id', 'dept_id']], on='course_id', how='left') # Tambahkan dept_id dari course
    df_model = pd.merge(df_model, avg_grade_per_course, on='course_id', how='left')
    df_model = pd.merge(df_model, assessment_count_per_course, on='course_id', how='left')
    
    # Isi NaN yang mungkin muncul setelah merge
    # Untuk average_grade_course dan assessment_count_course, isi dengan median atau 0 jika tidak ada data
    df_model['average_grade_course'] = df_model['average_grade_course'].fillna(df_model['average_grade_course'].median())
    df_model['assessment_count_course'] = df_model['assessment_count_course'].fillna(0) # Jika tidak ada asesmen, anggap 0
    df_model['dept_id'] = df_model['dept_id'].fillna(-1) # Placeholder untuk dept_id yang hilang


    # Hapus baris jika ada NaN krusial yang tersisa (misalnya di target atau fitur utama)
    df_model.dropna(subset=['difficulty_level', 'average_grade_course', 'assessment_count_course', 'dept_id'], inplace=True)
    
    if df_model.empty:
        print("❌ Tidak ada data yang cukup setelah penggabungan dan pembersihan NaN.")
        exit()

    print("✅ Data berhasil digabungkan dan fitur dibuat.")
    print("Beberapa baris data untuk model:")
    print(df_model.head())
    print(f"\nInfo data model:\n")
    df_model.info()
    print(f"\nDistribusi Target (difficulty_level):\n{df_model['difficulty_level'].value_counts()}")


    # Hapus course_id karena sudah digunakan untuk join dan bukan fitur langsung untuk generalisasi
    X = df_model.drop(['difficulty_level', 'course_id'], axis=1)
    y = df_model['difficulty_level']


except Exception as e:
    print(f"❌ Gagal dalam feature engineering: {e}"); exit()


# --- 4. Preprocessing (Scaling & Encoding) ---
print("\n🔄 Melakukan preprocessing (scaling & encoding)...")

# Identifikasi fitur numerik dan kategorikal
numerical_features = ['average_grade_course', 'assessment_count_course']
# dept_id akan di-one-hot encode
categorical_features = ['dept_id']


# Pastikan semua fitur ada di X
missing_cols_num = [col for col in numerical_features if col not in X.columns]
missing_cols_cat = [col for col in categorical_features if col not in X.columns]

if missing_cols_num: print(f"❌ Kolom numerik hilang dari X: {missing_cols_num}"); exit()
if missing_cols_cat: print(f"❌ Kolom kategorik hilang dari X: {missing_cols_cat}"); exit()


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

# --- 5. Pembuatan Pipeline dan Pelatihan Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # class_weight untuk imbalanced target

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])

# Split data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y if len(y.unique()) > 1 else None)

if len(X_train) == 0 or len(y_train.unique()) < 2 : # Cek jika data training kosong atau hanya 1 kelas
     print(f"❌ Data training tidak cukup atau hanya memiliki satu kelas setelah split. Jumlah sampel: {len(X_train)}, Kelas unik: {y_train.unique()}")
     exit()


print("\n⚙️ Melatih model...")
pipeline.fit(X_train, y_train)
print("✅ Model berhasil dilatih.")

# --- 6. Evaluasi Model ---
print("\n📊 Mengevaluasi model...")
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print(f"  Training Accuracy: {accuracy_train:.4f}")
print(f"  Test Accuracy    : {accuracy_test:.4f}")
print("\nClassification Report (Test Set):")
try:
    print(classification_report(y_test, y_pred_test, zero_division=0))
except ValueError as ve:
    print(f"  Tidak bisa membuat classification report: {ve}. Mungkin hanya ada satu kelas di y_test atau y_pred.")


# --- 7. Penyimpanan Model (Pipeline) ---
# Sesuaikan nama direktori aplikasi Anda
APP_NAME = "usecase_farhan" # Ganti jika nama aplikasi Django berbeda
MODEL_DIR = os.path.join(APP_NAME, "ml_models")

if not os.path.exists(MODEL_DIR):
    try:
        os.makedirs(MODEL_DIR)
        print(f"Direktori {MODEL_DIR} dibuat.")
    except OSError as e:
        print(f"Gagal membuat direktori {MODEL_DIR}: {e}")
        # Jika gagal buat direktori di dalam app, coba buat di root project
        MODEL_DIR = "ml_models_difficulty" 
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        print(f"Model akan disimpan di direktori root: {MODEL_DIR}")


MODEL_FILENAME = os.path.join(MODEL_DIR, 'course_difficulty_classifier_pipeline.pkl')
# Simpan juga kolom fitur MENTAH yang dibutuhkan form (sebelum preprocessing)
RAW_FEATURES_FOR_FORM = X.columns.tolist()
RAW_FEATURES_FILENAME = os.path.join(MODEL_DIR, 'course_difficulty_raw_features.pkl')

try:
    with open(MODEL_FILENAME, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"\n✅ Model pipeline disimpan ke: {MODEL_FILENAME}")

    with open(RAW_FEATURES_FILENAME, 'wb') as f:
        pickle.dump(RAW_FEATURES_FOR_FORM, f)
    print(f"✅ Daftar fitur mentah untuk form disimpan ke: {RAW_FEATURES_FILENAME}")
except Exception as e:
    print(f"❌ Gagal menyimpan model atau fitur: {e}")


print("\n🎉 Selesai!")