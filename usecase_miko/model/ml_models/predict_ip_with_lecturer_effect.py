import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

print("🚀 Melatih Model Prediksi IP Semester 3 (dengan Fitur Dosen) 🚀")

# --- 1. Koneksi ke Database ---
DB_USER = "postgres"
DB_PASSWORD = "DBmiko"  # Sesuaikan
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "dbexam"

try:
    db_engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    print("✅ Koneksi database berhasil")
except Exception as e:
    print(f"❌ Koneksi gagal: {e}"); exit()

# --- 2. Fungsi-Fungsi Helper ---
def calculate_na_mk(row, assessment_cols_map):
    bobot_mid = 0.25; bobot_final = 0.35; bobot_project = 0.30; bobot_kehadiran = 0.10
    mid_score = row.get(assessment_cols_map.get('Mid', 'score_mid_placeholder'), 0)
    final_score = row.get(assessment_cols_map.get('Final', 'score_final_placeholder'), 0)
    project_score = row.get(assessment_cols_map.get('Project', 'score_project_placeholder'), 0)
    kehadiran_mk = row.get('attendance_percentage', 0)
    na_mk = (mid_score * bobot_mid) + (final_score * bobot_final) + (project_score * bobot_project) + (kehadiran_mk * bobot_kehadiran)
    return na_mk

def na_to_ip_mk(na_mk):
    if pd.isna(na_mk): return 0.0
    if na_mk >= 85: return 4.0
    elif na_mk >= 75: return 3.0
    elif na_mk >= 65: return 2.0
    elif na_mk >= 55: return 1.0
    else: return 0.0

# --- 3. AMBIL DATA DASAR ---
print("\n📊 Mengambil data dari database...")
try:
    student_df = pd.read_sql("SELECT stu_id, gender, dept_id FROM student", db_engine)
    enrollment_df = pd.read_sql("SELECT enroll_id, stu_id, course_id, semester_id FROM enrollment WHERE semester_id IN (1, 2, 3)", db_engine)
    assessment_df = pd.read_sql("SELECT enroll_id, assessment_type, score FROM assessment", db_engine)
    attendance_df = pd.read_sql("SELECT enroll_id, attendance_percentage FROM attendance", db_engine)
    course_difficulty_df = pd.read_sql("SELECT course_id, difficulty_level FROM course_difficulty", db_engine)
    course_instructor_df = pd.read_sql("SELECT course_id, instructor_id, semester_id FROM course_instructor WHERE semester_id IN (1, 2, 3)", db_engine)
    print("✅ Data dasar berhasil diambil.")
except Exception as e:
    print(f"❌ Gagal mengambil data dasar: {e}"); exit()

# --- 4. PREPROCESSING DATA DASAR & HITUNG IP_KOMPREHENSIF_MK ---
print("\n⚙️ Preprocessing data dasar & Hitung IP Komprehensif per MK...")
assessment_pivot = assessment_df.pivot_table(index='enroll_id', columns='assessment_type', values='score', aggfunc='mean').reset_index()
assessment_pivot.columns = ['enroll_id' if col == 'enroll_id' else f"score_{col.lower().replace(' ', '_').replace('-', '_')}" for col in assessment_pivot.columns]
ASSESSMENT_COLS_MAP = {'Mid': 'score_midterm', 'Final': 'score_final', 'Project': 'score_project'} # WAJIB SESUAIKAN INI
print("Kolom skor terdeteksi setelah pivot (pastikan ASSESSMENT_COLS_MAP sesuai):")
detected_score_cols = [col for col in assessment_pivot.columns if col.startswith('score_')]
print(detected_score_cols)
for key_map, val_map in ASSESSMENT_COLS_MAP.items():
    if val_map not in assessment_pivot.columns:
        print(f"Peringatan: Kolom skor untuk '{key_map}' ('{val_map}') tidak ditemukan. Akan diisi 0.")
        assessment_pivot[val_map] = 0

attendance_processed = attendance_df.groupby('enroll_id')['attendance_percentage'].mean().reset_index()
enrollment_details = pd.merge(enrollment_df, assessment_pivot, on='enroll_id', how='left')
enrollment_details = pd.merge(enrollment_details, attendance_processed, on='enroll_id', how='left')
enrollment_details = pd.merge(enrollment_details, course_difficulty_df, on='course_id', how='left')
enrollment_details['difficulty_level'].fillna('Unknown', inplace=True)
enrollment_details['na_mk'] = enrollment_details.apply(lambda row: calculate_na_mk(row, ASSESSMENT_COLS_MAP), axis=1)
enrollment_details['ip_mk'] = enrollment_details['na_mk'].apply(na_to_ip_mk)

# --- 5. FEATURE ENGINEERING: Membuat Fitur Agregat per Mahasiswa per Semester (S1, S2) & Target (IP S3) ---
print("\n🔧 Membuat fitur historis (S1 & S2) dan target (IP Sem 3)...")
all_students_with_sem3_ip = enrollment_details[enrollment_details['semester_id'] == 3]['stu_id'].unique()
if len(all_students_with_sem3_ip) == 0: print("❌ Tidak ada data S3 untuk target."); exit()
base_features_df = pd.DataFrame({'stu_id': all_students_with_sem3_ip})
student_features_list = [base_features_df]

for sem_hist in [1, 2]: # Fitur dari S1 dan S2
    df_sem = enrollment_details[enrollment_details['semester_id'] == sem_hist]
    temp_features_for_sem = base_features_df[['stu_id']].copy() # Mulai dengan semua stu_id yg punya target
    if not df_sem.empty:
        agg_functions = {}
        for score_type_key, actual_score_col_name in ASSESSMENT_COLS_MAP.items():
            if actual_score_col_name in df_sem.columns: agg_functions[actual_score_col_name] = 'mean'
        if 'attendance_percentage' in df_sem.columns: agg_functions['attendance_percentage'] = 'mean'
        agg_functions['course_id'] = 'count'
        sem_agg_features = df_sem.groupby('stu_id').agg(agg_functions).reset_index()
        new_feature_names = {'stu_id': 'stu_id'}
        for score_type_key, actual_score_col_name in ASSESSMENT_COLS_MAP.items():
            if actual_score_col_name in df_sem.columns: new_feature_names[actual_score_col_name] = f'avg_score_{score_type_key.lower()}_sem{sem_hist}'
        if 'attendance_percentage' in df_sem.columns: new_feature_names['attendance_percentage'] = f'avg_kehadiran_sem{sem_hist}'
        new_feature_names['course_id'] = f'jumlah_mk_sem{sem_hist}'
        sem_agg_features.rename(columns=new_feature_names, inplace=True)
        if 'difficulty_level' in df_sem.columns:
            difficulty_summary = df_sem.groupby('stu_id')['difficulty_level'].value_counts().unstack(fill_value=0)
            difficulty_summary.columns = [f'jumlah_{col.lower()}_sem{sem_hist}' for col in difficulty_summary.columns]
            difficulty_summary.reset_index(inplace=True)
            sem_agg_features = pd.merge(sem_agg_features, difficulty_summary, on='stu_id', how='left')
        
        # Fitur Dosen: Jumlah dosen unik
        dosen_sem = pd.merge(df_sem[['stu_id', 'course_id']], course_instructor_df[course_instructor_df['semester_id'] == sem_hist], on='course_id', how='left')
        dosen_unik_sem = dosen_sem.groupby('stu_id')['instructor_id'].nunique().reset_index(name=f'jumlah_dosen_unik_sem{sem_hist}')
        sem_agg_features = pd.merge(sem_agg_features, dosen_unik_sem, on='stu_id', how='left')
        
        temp_features_for_sem = pd.merge(temp_features_for_sem, sem_agg_features, on='stu_id', how='left')
    
    # Pastikan semua kolom fitur ada, isi 0 jika tidak ada data untuk mahasiswa/semester itu
    for score_type_key in ASSESSMENT_COLS_MAP.keys():
        col_name = f'avg_score_{score_type_key.lower()}_sem{sem_hist}'
        if col_name not in temp_features_for_sem.columns: temp_features_for_sem[col_name] = 0.0
    if f'avg_kehadiran_sem{sem_hist}' not in temp_features_for_sem.columns: temp_features_for_sem[f'avg_kehadiran_sem{sem_hist}'] = 0.0
    if f'jumlah_mk_sem{sem_hist}' not in temp_features_for_sem.columns: temp_features_for_sem[f'jumlah_mk_sem{sem_hist}'] = 0
    for diff_level in ['easy', 'medium', 'hard', 'unknown']:
        col_name = f'jumlah_{diff_level}_sem{sem_hist}'
        if col_name not in temp_features_for_sem.columns: temp_features_for_sem[col_name] = 0
    if f'jumlah_dosen_unik_sem{sem_hist}' not in temp_features_for_sem.columns: temp_features_for_sem[f'jumlah_dosen_unik_sem{sem_hist}'] = 0
    student_features_list.append(temp_features_for_sem)

features_df = student_features_list[0]
for i in range(1, len(student_features_list)):
    if len(student_features_list[i].columns) > 1 : features_df = pd.merge(features_df, student_features_list[i], on='stu_id', how='left')
features_df = pd.merge(features_df, student_df, on='stu_id', how='left') # Tambah demografi
df_target_sem3 = enrollment_details[enrollment_details['semester_id'] == 3]
if df_target_sem3.empty: print("❌ Tidak ada data S3 untuk target."); exit()
ip_sem3_per_mahasiswa = df_target_sem3.groupby('stu_id')['ip_mk'].mean().reset_index()
ip_sem3_per_mahasiswa.rename(columns={'ip_mk': 'TARGET_IP_SEMESTER_3'}, inplace=True)
final_ml_df = pd.merge(features_df, ip_sem3_per_mahasiswa, on='stu_id', how='inner')
final_ml_df.fillna(0, inplace=True)
if final_ml_df.empty: print("❌ Tidak ada data untuk training."); exit()
print(f"✅ Dataset final untuk training (Target IP Sem 3): {len(final_ml_df)} mahasiswa")

# --- 6. PERSIAPAN DATA UNTUK MODEL XGBOOST ---
print("\n🤖 Persiapan data untuk model XGBoost...")
cols_to_drop = ['stu_id', 'TARGET_IP_SEMESTER_3'] 
categorical_features = ['gender', 'dept_id'] 
X = final_ml_df.drop(columns=cols_to_drop)
y = final_ml_df['TARGET_IP_SEMESTER_3']
X = X.loc[:, (X != 0).any(axis=0)] 
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
actual_categorical_features = [col for col in categorical_features if col in X.columns]
final_feature_columns = X.columns.tolist() 

if not actual_categorical_features:
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_features)],remainder='passthrough')
else:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first' if len(actual_categorical_features) > 1 else None), actual_categorical_features)
        ], remainder='passthrough')

# --- 7. SPLIT DATA DAN TRAINING MODEL XGBOOST ---
print("\n🚀 Training model XGBoost...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
if X_train.empty: print("❌ X_train kosong."); exit()

xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))])
param_grid = {
    'regressor__n_estimators': [50, 100, 150], 'regressor__max_depth': [3, 4, 5],
    'regressor__learning_rate': [0.01, 0.05, 0.1], 'regressor__subsample': [0.7, 0.8, 0.9]
} # Grid lebih luas sedikit
cv_folds = min(3, len(X_train) // 2 if len(X_train) > 1 else 1)
if cv_folds < 2: cv_folds = 2 # Minimal CV 2 jika data sangat sedikit
print(f"   Menggunakan CV={cv_folds} untuk GridSearchCV karena ukuran X_train = {len(X_train)}")

grid_search = GridSearchCV(xgb_pipeline, param_grid, cv=cv_folds, scoring='r2', verbose=1, n_jobs=-1, error_score='raise')
best_model_pipeline = None
try:
    grid_search.fit(X_train, y_train)
    best_model_pipeline = grid_search.best_estimator_
    print(f"\n🏆 Parameter XGBoost Terbaik: {grid_search.best_params_}")
    print(f"   Skor R2 CV Terbaik dari GridSearchCV: {grid_search.best_score_:.4f}")
except ValueError as ve:
    print(f"❌ Error saat GridSearchCV: {ve}")
    print("   Mencoba melatih dengan parameter default XGBoost saja...")
    try:
        X_train_processed = preprocessor.fit_transform(X_train)
        if X_train_processed.shape[1] == 0: print("   ❌ Tidak ada fitur tersisa setelah preprocessing."); exit()
        xgb_default = XGBRegressor(objective='reg:squarederror', random_state=42)
        xgb_default.fit(X_train_processed, y_train)
        # Buat pipeline manual untuk konsistensi
        best_model_pipeline = Pipeline(steps=[('preprocessor', preprocessor.fit(X_train)), # preprocessor di-fit di X_train
                                              ('regressor', xgb_default)])
        print("   Berhasil melatih XGBoost dengan parameter default.")
    except Exception as e_fallback:
        print(f"   ❌ Gagal melatih XGBoost default: {e_fallback}"); exit()

if best_model_pipeline is None: print("❌ Model terbaik tidak berhasil dilatih."); exit()

# --- 8. EVALUASI MODEL ---
print("\n📈 Evaluasi model...")
y_pred_train = best_model_pipeline.predict(X_train)
y_pred_test = best_model_pipeline.predict(X_test)
y_pred_test = np.clip(y_pred_test, 0, 4)
print(f"✅ Hasil evaluasi (TRAIN):")
print(f"   R2 Score: {r2_score(y_train, y_pred_train):.4f}, MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")
print(f"✅ Hasil evaluasi (TEST):")
r2_test = r2_score(y_test, y_pred_test); mae_test = mean_absolute_error(y_test, y_pred_test)
print(f"   R2 Score: {r2_test:.4f}, MAE: {mae_test:.4f}")
if r2_test < 0: print("   ⚠️ PERINGATAN: R2 Score TEST negatif.")
elif r2_test < 0.1: print("   ⚠️ PERINGATAN: R2 Score TEST sangat rendah.")

# --- 9. PENYIMPANAN MODEL ---
MODEL_DIR = 'ml_model_ip_sem3_with_dosen_feature' # Folder baru
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILENAME = os.path.join(MODEL_DIR, 'xgb_ip_sem3_dosen_pipeline.pkl')
with open(MODEL_FILENAME, 'wb') as f: pickle.dump(best_model_pipeline, f)
original_feature_columns = X_train.columns.tolist()
COLUMNS_FILENAME = os.path.join(MODEL_DIR, 'original_columns_ip_sem3_dosen.pkl')
with open(COLUMNS_FILENAME, 'wb') as f: pickle.dump(original_feature_columns, f)
ASSESSMENT_MAP_FILENAME = os.path.join(MODEL_DIR, 'assessment_cols_map.pkl')
with open(ASSESSMENT_MAP_FILENAME, 'wb') as f: pickle.dump(ASSESSMENT_COLS_MAP, f)

print(f"\n✅ Model Pipeline, Kolom Asli, dan Assessment Map disimpan di '{MODEL_DIR}'")
print(f"🎉 Training Selesai!")
print(f"   Untuk prediksi IP Semester 4, gunakan model ini dengan data Semester 2 & 3.")