# train_risk_model_kartika.py
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import os # Untuk path yang lebih baik

# Scikit-learn untuk preprocessing dan modeling
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Model dan Explainability
import xgboost as xgb
import shap # Pastikan shap sudah terinstall: pip install shap
import warnings

warnings.filterwarnings('ignore') # Untuk menyembunyikan warning dari XGBoost atau lainnya

print("🚀 [Kartika] Melatih Model Penilaian Risiko Mahasiswa 🚀")

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

# --- 2. Pengambilan dan Penggabungan Data ---
# Query ini melakukan PIVOT pada tabel assessment
sql_query = """
SELECT
    e.enroll_id,
    s.stu_id,
    s.gender,
    d.dept_name,
    c.course_name,
    cd.difficulty_level,
    a.attendance_percentage,
    -- Pivoting assessment scores
    MAX(CASE WHEN ass.assessment_type = 'Midterm' THEN ass.score END) AS score_midterm,
    MAX(CASE WHEN ass.assessment_type = 'Final' THEN ass.score END) AS score_final,
    MAX(CASE WHEN ass.assessment_type = 'Project' THEN ass.score END) AS score_project,
    e.grade
FROM enrollment e
LEFT JOIN student s ON e.stu_id = s.stu_id
LEFT JOIN department d ON s.dept_id = d.dept_id
LEFT JOIN course c ON e.course_id = c.course_id
LEFT JOIN course_difficulty cd ON e.course_id = cd.course_id
LEFT JOIN attendance a ON e.enroll_id = a.enroll_id
LEFT JOIN assessment ass ON e.enroll_id = ass.enroll_id
GROUP BY e.enroll_id, s.stu_id, s.gender, d.dept_name, c.course_name, cd.difficulty_level, a.attendance_percentage
ORDER BY e.enroll_id;
"""
try:
    df = pd.read_sql_query(sql_query, db_engine)
    print(f"✅ Data berhasil dimuat. Jumlah baris: {len(df)}")
    if df.empty:
        print("❌ Data kosong. Tidak bisa melanjutkan pelatihan. Pastikan database Anda terisi.")
        exit()
except Exception as e:
    print(f"❌ Gagal memuat data dari database: {e}"); exit()

# --- 3. Pembuatan Target Variable dan Pembersihan ---
# Definisikan "Beresiko" jika nilai < 55 (sesuai skrip Anda)
df['status'] = df['grade'].apply(lambda x: 1 if pd.notnull(x) and x < 55 else 0)

# Hapus kolom asli 'grade' dan identifier yang tidak relevan untuk model
df_model = df.drop(columns=['grade', 'enroll_id', 'stu_id'])
print("✅ Kolom target 'status' berhasil dibuat dan kolom tidak relevan dihapus.")
print("\nInfo data untuk model:")
df_model.info()
print("\nContoh data untuk model:")
print(df_model.head())
print(f"\nDistribusi target (status):\n{df_model['status'].value_counts(normalize=True)}")

if len(df_model['status'].unique()) < 2:
    print("❌ Target variable hanya memiliki satu kelas. Tidak bisa melatih model klasifikasi.")
    exit()

# --- 4. Pemisahan Fitur dan Target, Pembagian Data ---
X = df_model.drop('status', axis=1)
y = df_model['status']

# Simpan nama kolom fitur mentah (sebelum preprocessing) untuk digunakan di form Django
RAW_FEATURE_COLUMNS = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nData latih: {X_train.shape}, Data uji: {X_test.shape}")
if X_train.empty:
    print("❌ Data latih kosong setelah split. Tidak bisa melanjutkan.")
    exit()

# --- 5. Preprocessing Pipeline ---
numeric_features = ['attendance_percentage', 'score_midterm', 'score_final', 'score_project']
categorical_features = ['gender', 'dept_name', 'course_name', 'difficulty_level']

# Cek apakah semua fitur ada di X_train
for feature_list in [numeric_features, categorical_features]:
    for feature in feature_list:
        if feature not in X_train.columns:
            print(f"❌ Fitur '{feature}' tidak ditemukan di data training. Harap periksa query SQL atau daftar fitur.")
            exit()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False untuk XGBoost
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Drop kolom lain yang tidak didefinisikan
)
print("✅ Pipeline pra-pemrosesan berhasil dibuat.")

# --- 6. Pelatihan Model ---
# eval_metric='logloss' sudah baik. use_label_encoder=False juga sudah benar untuk XGBoost > 1.0
model = xgb.XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

print("\n⚙️ Memulai pelatihan model...")
full_pipeline.fit(X_train, y_train)
print("✅ Pelatihan model selesai.")

# --- 7. Evaluasi Model ---
y_pred = full_pipeline.predict(X_test)
y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1]

print("\n📊 Laporan Klasifikasi:")
print(classification_report(y_test, y_pred, zero_division=0))
print(f"  Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print(f"  ROC AUC Score : {roc_auc_score(y_test, y_pred_proba):.4f}")

# --- 8. Penyimpanan Model, Explainer, dan Fitur ---
# Tentukan BASE_DIR proyek Django Anda secara manual jika skrip ini di luar proyek
# atau jalankan skrip ini dari root direktori proyek Django Anda.
# Untuk kemudahan, kita asumsikan skrip ini dijalankan dari root proyek.
BASE_DIR = os.getcwd() # Mendapatkan direktori kerja saat ini
MODEL_STORAGE_DIR = os.path.join(BASE_DIR, 'ml_models') # Folder di root proyek

if not os.path.exists(MODEL_STORAGE_DIR):
    os.makedirs(MODEL_STORAGE_DIR)
    print(f"Direktori '{MODEL_STORAGE_DIR}' telah dibuat.")

PIPELINE_PATH = os.path.join(MODEL_STORAGE_DIR, 'kartika_risk_assessment_pipeline.pkl')
RAW_FEATURES_LIST_PATH = os.path.join(MODEL_STORAGE_DIR, 'kartika_risk_raw_features.pkl') # Menyimpan daftar fitur mentah
# Untuk SHAP, kita perlu menyimpan explainer dan nama fitur SETELAH preprocessing
# karena SHAP bekerja pada data yang sudah ditransformasi
PROCESSED_FEATURES_PATH = os.path.join(MODEL_STORAGE_DIR, 'kartika_risk_processed_features.pkl')
EXPLAINER_PATH = os.path.join(MODEL_STORAGE_DIR, 'kartika_risk_shap_explainer.pkl')


with open(PIPELINE_PATH, 'wb') as f:
    pickle.dump(full_pipeline, f)
print(f"\n✅ Pipeline lengkap telah disimpan di: {PIPELINE_PATH}")

with open(RAW_FEATURES_LIST_PATH, 'wb') as f:
    pickle.dump(RAW_FEATURE_COLUMNS, f)
print(f"✅ Daftar fitur mentah telah disimpan di: {RAW_FEATURES_LIST_PATH}")

# Membuat SHAP Explainer
# SHAP bekerja pada data yang sudah di-transform oleh preprocessor
# Kita perlu melatih ulang preprocessor pada X_train untuk mendapatkan nama fitur yang benar
# atau mendapatkan nama fitur dari pipeline yang sudah dilatih.
try:
    preprocessor_fitted = full_pipeline.named_steps['preprocessor']
    model_fitted = full_pipeline.named_steps['model']
    
    # Dapatkan nama fitur setelah diproses (terutama setelah OneHotEncoding)
    feature_names_processed = preprocessor_fitted.get_feature_names_out()

    explainer = shap.TreeExplainer(model_fitted) # Untuk XGBoost

    with open(EXPLAINER_PATH, 'wb') as f:
        pickle.dump(explainer, f)
    print(f"✅ SHAP Explainer telah disimpan di: {EXPLAINER_PATH}")

    with open(PROCESSED_FEATURES_PATH, 'wb') as f:
        pickle.dump(feature_names_processed.tolist(), f) # Simpan sebagai list
    print(f"✅ Nama Fitur (setelah preprocessing) telah disimpan di: {PROCESSED_FEATURES_PATH}")
except Exception as e:
    print(f"❌ Gagal membuat atau menyimpan SHAP explainer/fitur terproses: {e}")

print("\n🎉 Selesai!")