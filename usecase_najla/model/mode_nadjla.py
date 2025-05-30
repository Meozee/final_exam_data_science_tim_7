# train_attendance_model_najla.py (jalankan dari root proyek atau sesuaikan path)
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

print("🚀 Melatih Model Prediksi Kehadiran Mahasiswa (Najla Usecase V2) 🚀")

# --- 1. Koneksi ke Database ---
DB_USER = "postgres"
DB_PASSWORD = "DBmiko"  # GANTI DENGAN PASSWORD ANDA
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
    q_enrollment = "SELECT enroll_id, stu_id, course_id, semester_id, grade FROM public.enrollment"
    enrollment_df = pd.read_sql(q_enrollment, db_engine)

    q_student = "SELECT stu_id, gender, dob, dept_id FROM public.student"
    student_df = pd.read_sql(q_student, db_engine)

    q_attendance = "SELECT enroll_id, attendance_percentage FROM public.attendance"
    attendance_df = pd.read_sql(q_attendance, db_engine)

    q_assessment = "SELECT enroll_id, score FROM public.assessment"
    assessment_df = pd.read_sql(q_assessment, db_engine)
    
    print("✅ Data mentah berhasil diambil.")
except Exception as e:
    print(f"❌ Gagal mengambil data mentah: {e}"); exit()

# --- 3. Penggabungan Data ---
print("\n🔄 Menggabungkan tabel-tabel...")
try:
    df_merged = pd.merge(attendance_df, enrollment_df, on='enroll_id', how='inner')
    avg_assessment_score_df = assessment_df.groupby('enroll_id')['score'].mean().reset_index()
    avg_assessment_score_df = avg_assessment_score_df.rename(columns={'score': 'average_score'})
    df_merged = pd.merge(df_merged, avg_assessment_score_df, on='enroll_id', how='left')
    df_merged = pd.merge(df_merged, student_df, on='stu_id', how='left')
    df_merged = df_merged.drop(columns=['enroll_id'])
    print("✅ Tabel berhasil digabungkan.")
except Exception as e:
    print(f"❌ Gagal menggabungkan data: {e}"); exit()

# --- 4. Feature Engineering ---
print("\n🔄 Melakukan feature engineering...")
df_processed = df_merged.copy()

if 'dob' in df_processed.columns:
    df_processed['dob'] = pd.to_datetime(df_processed['dob'], errors='coerce')
    current_year = pd.Timestamp.now().year
    df_processed['age'] = current_year - df_processed['dob'].dt.year # Perhitungan umur yang lebih sederhana
    df_processed = df_processed.drop(columns=['dob'])
    df_processed['age'] = df_processed['age'].fillna(df_processed['age'].median()).astype(int)
else:
    # Jika 'age' akan digunakan sebagai fitur dan 'dob' tidak ada
    # df_processed['age'] = 20 # Atau strategi imputasi lain
    print("Kolom 'dob' tidak ditemukan, fitur 'age' tidak akan dibuat dari 'dob'. Pastikan 'age' ada jika dibutuhkan model.")


df_processed['grade'] = df_processed['grade'].fillna(df_processed['grade'].median())
df_processed['average_score'] = df_processed['average_score'].fillna(df_processed['average_score'].median())

# Fitur yang akan digunakan (pastikan ini ada semua di df_processed)
# 'stu_id' akan di-encode, jadi tidak perlu khawatir jika bukan integer murni di input form (selama konsisten)
# 'dept_id', 'course_id', 'semester_id' juga akan di-encode
required_features_for_training = ['attendance_percentage', 'average_score', 'grade', 
                                  'course_id', 'semester_id', 'stu_id', 
                                  'gender', 'dept_id'] # Tambahkan 'age' jika dibuat dan digunakan
if 'age' in df_processed.columns:
    required_features_for_training.append('age')


df_processed.dropna(subset=required_features_for_training, inplace=True)


if df_processed.empty or len(df_processed) < 20: # Butuh cukup data untuk split dan train
    print(f"❌ Tidak ada data tersisa setelah pembersihan NaN atau data terlalu sedikit ({len(df_processed)} baris). Tidak bisa melatih model.")
    exit()

print("Beberapa baris data setelah feature engineering:")
print(df_processed.head())
df_processed.info()

X = df_processed.drop('attendance_percentage', axis=1)
y = df_processed['attendance_percentage']

# --- 5. Preprocessing (Scaling & Encoding) ---
print("\n🔄 Melakukan preprocessing (scaling & encoding)...")

numerical_features = ['average_score', 'grade']
if 'age' in X.columns:
    numerical_features.append('age')

categorical_features = ['course_id', 'semester_id', 'stu_id', 'gender', 'dept_id']

# Validasi bahwa semua fitur ada di X
for col_list in [numerical_features, categorical_features]:
    for col in col_list:
        if col not in X.columns:
            print(f"❌ Kolom '{col}' yang dibutuhkan untuk preprocessing tidak ditemukan di X. Kolom X: {X.columns.tolist()}"); exit()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='drop' # Drop kolom lain yang tidak disebutkan
)

# --- 6. Pembuatan Pipeline dan Pelatihan Model ---
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if len(X_train) == 0:
     print(f"❌ Data training kosong setelah split."); exit()

print("\n⚙️ Melatih model...")
pipeline.fit(X_train, y_train)
print("✅ Model berhasil dilatih.")

# --- 7. Evaluasi Model ---
print("\n📊 Mengevaluasi model...")
# ... (kode evaluasi MAE dan R2 sama seperti sebelumnya) ...
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
print(f"  Training MAE   : {mae_train:.4f}")
print(f"  Training R^2   : {r2_train:.4f}")
print(f"  Test MAE       : {mae_test:.4f}")
print(f"  Test R^2       : {r2_test:.4f}")

# --- 8. Penyimpanan Model (Pipeline) ---
MODEL_DIR = "ml_models" # Simpan di root/ml_models
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

MODEL_FILENAME = os.path.join(MODEL_DIR, 'najla_attendance_predictor_pipeline.pkl')
RAW_FEATURES_FOR_FORM = X.columns.tolist() 
RAW_FEATURES_FILENAME = os.path.join(MODEL_DIR, 'najla_attendance_model_raw_features.pkl')

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