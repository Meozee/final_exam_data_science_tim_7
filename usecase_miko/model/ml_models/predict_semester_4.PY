from matplotlib import pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils import resample # Untuk bootstrapping
import pickle
import os
import warnings
import seaborn as sns

warnings.filterwarnings('ignore')

print("🚀 Melatih Model Prediksi IP (Target: IP Sem 3) dengan Pendekatan Robust & Evaluasi Bootstrapping 🚀")

# --- 1. Koneksi ke Database (Sama seperti sebelumnya) ---
DB_USER = "postgres"
DB_PASSWORD = "DBmiko"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "dbexam"
try:
    db_engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    print("✅ Koneksi database berhasil")
except Exception as e:
    print(f"❌ Koneksi gagal: {e}"); exit()

# --- 2. Fungsi-Fungsi Helper (Sama seperti sebelumnya) ---
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

# --- 3. AMBIL DATA DASAR (Sama seperti sebelumnya) ---
print("\n📊 Mengambil data dari database...")
try:
    student_df = pd.read_sql("SELECT stu_id, gender, dept_id FROM student", db_engine)
    enrollment_df = pd.read_sql("SELECT enroll_id, stu_id, course_id, semester_id FROM enrollment WHERE semester_id IN (1, 2, 3)", db_engine)
    assessment_df = pd.read_sql("SELECT enroll_id, assessment_type, score FROM assessment", db_engine)
    attendance_df = pd.read_sql("SELECT enroll_id, attendance_percentage FROM attendance", db_engine)
    course_difficulty_df = pd.read_sql("SELECT course_id, difficulty_level FROM course_difficulty", db_engine)
    print("✅ Data dasar berhasil diambil.")
except Exception as e:
    print(f"❌ Gagal mengambil data dasar: {e}"); exit()

# --- 4. PREPROCESSING DATA DASAR & HITUNG IP_KOMPREHENSIF_MK (Sama seperti sebelumnya) ---
print("\n⚙️ Preprocessing data dasar & Hitung IP Komprehensif per MK...")
assessment_pivot = assessment_df.pivot_table(index='enroll_id', columns='assessment_type', values='score', aggfunc='mean').reset_index()
assessment_pivot.columns = ['enroll_id' if col == 'enroll_id' else f"score_{col.lower().replace(' ', '_').replace('-', '_')}" for col in assessment_pivot.columns]
# PENTING: Sesuaikan nama kolom ini dengan output pivot Anda!
ASSESSMENT_COLS_MAP = {
    'Mid': 'score_midterm',     # GANTI 'score_midterm' dengan nama kolom Mid aktual Anda
    'Final': 'score_final',     # GANTI 'score_final' dengan nama kolom Final aktual Anda
    'Project': 'score_project'  # GANTI 'score_project' dengan nama kolom Project aktual Anda
}
print("Kolom skor terdeteksi setelah pivot (pastikan ASSESSMENT_COLS_MAP sesuai):")
detected_score_cols = [col for col in assessment_pivot.columns if col.startswith('score_')]
print(detected_score_cols)
# Pastikan semua kolom yang di-map ada, atau handle jika tidak ada
for key_map, val_map in ASSESSMENT_COLS_MAP.items():
    if val_map not in assessment_pivot.columns:
        print(f"Peringatan: Kolom skor untuk '{key_map}' ('{val_map}') tidak ditemukan di pivot assessment. Akan diisi 0 saat perhitungan NA.")
        # Buat kolom placeholder jika tidak ada agar fungsi calculate_na_mk tidak error
        assessment_pivot[val_map] = 0 


attendance_processed = attendance_df.groupby('enroll_id')['attendance_percentage'].mean().reset_index()
enrollment_details = pd.merge(enrollment_df, assessment_pivot, on='enroll_id', how='left')
enrollment_details = pd.merge(enrollment_details, attendance_processed, on='enroll_id', how='left')
enrollment_details = pd.merge(enrollment_details, course_difficulty_df, on='course_id', how='left')
enrollment_details['difficulty_level'].fillna('Unknown', inplace=True)
enrollment_details['na_mk'] = enrollment_details.apply(lambda row: calculate_na_mk(row, ASSESSMENT_COLS_MAP), axis=1)
enrollment_details['ip_mk'] = enrollment_details['na_mk'].apply(na_to_ip_mk)

# --- 5. FEATURE ENGINEERING & TARGET (Sama seperti sebelumnya) ---
print("\n🔧 Membuat fitur historis (Semester 1 & 2) dan target (IP Sem 3)...")
all_students_with_sem3_ip = enrollment_details[enrollment_details['semester_id'] == 3]['stu_id'].unique()
if len(all_students_with_sem3_ip) == 0:
    print("❌ Tidak ada mahasiswa dengan data di Semester 3. Training tidak bisa dilanjutkan."); exit()
base_features_df = pd.DataFrame({'stu_id': all_students_with_sem3_ip})
student_features_list = [base_features_df]
for sem_hist in [1, 2]:
    df_sem = enrollment_details[enrollment_details['semester_id'] == sem_hist]
    temp_features_for_sem = base_features_df[['stu_id']].copy()
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
        temp_features_for_sem = pd.merge(temp_features_for_sem, sem_agg_features, on='stu_id', how='left')
    for score_type_key in ASSESSMENT_COLS_MAP.keys():
        col_name = f'avg_score_{score_type_key.lower()}_sem{sem_hist}'
        if col_name not in temp_features_for_sem.columns: temp_features_for_sem[col_name] = 0.0
    if f'avg_kehadiran_sem{sem_hist}' not in temp_features_for_sem.columns: temp_features_for_sem[f'avg_kehadiran_sem{sem_hist}'] = 0.0
    if f'jumlah_mk_sem{sem_hist}' not in temp_features_for_sem.columns: temp_features_for_sem[f'jumlah_mk_sem{sem_hist}'] = 0
    for diff_level in ['easy', 'medium', 'hard', 'unknown']:
        col_name = f'jumlah_{diff_level}_sem{sem_hist}'
        if col_name not in temp_features_for_sem.columns: temp_features_for_sem[col_name] = 0
    student_features_list.append(temp_features_for_sem)
features_df = student_features_list[0]
for i in range(1, len(student_features_list)):
    if len(student_features_list[i].columns) > 1 : features_df = pd.merge(features_df, student_features_list[i], on='stu_id', how='left')
features_df = pd.merge(features_df, student_df, on='stu_id', how='left')
df_target_sem3 = enrollment_details[enrollment_details['semester_id'] == 3]
if df_target_sem3.empty: print("❌ Tidak ada data S3 untuk target."); exit()
ip_sem3_per_mahasiswa = df_target_sem3.groupby('stu_id')['ip_mk'].mean().reset_index()
ip_sem3_per_mahasiswa.rename(columns={'ip_mk': 'TARGET_IP_SEMESTER_3'}, inplace=True)
final_ml_df = pd.merge(features_df, ip_sem3_per_mahasiswa, on='stu_id', how='inner')
final_ml_df.fillna(0, inplace=True)
if final_ml_df.empty: print("❌ Tidak ada data untuk training."); exit()
print(f"✅ Dataset final untuk training (Target IP Sem 3): {len(final_ml_df)} mahasiswa")

# --- 6. PERSIAPAN DATA UNTUK MODEL ---
print("\n🤖 Persiapan data untuk model ...")
cols_to_drop = ['stu_id', 'TARGET_IP_SEMESTER_3'] 
categorical_features = ['gender', 'dept_id'] 
X = final_ml_df.drop(columns=cols_to_drop)
y = final_ml_df['TARGET_IP_SEMESTER_3']
X = X.loc[:, (X != 0).any(axis=0)] # Hapus kolom yang semua nilainya 0
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
actual_categorical_features = [col for col in categorical_features if col in X.columns]

# Simpan kolom fitur final setelah drop kolom yg semua nilainya 0
# Ini akan menjadi kolom yang diharapkan oleh preprocessor
final_feature_columns = X.columns.tolist() 

if not actual_categorical_features:
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_features)],remainder='passthrough')
else:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first' if len(actual_categorical_features) > 1 else None), actual_categorical_features)
        ], remainder='passthrough')

# --- 7. SPLIT DATA ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
if X_train.empty: print("❌ X_train kosong."); exit()

# --- 8. TRAINING & EVALUASI MODEL ---
print("\n🚀 Training & Evaluasi Model...")

# A. RidgeCV (Model Sederhana dengan CV Internal)
print("\n   --- Menguji RidgeCV ---")
ridge_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', RidgeCV(alphas=np.logspace(-6, 6, 13), store_cv_values=True))])
ridge_pipeline.fit(X_train, y_train)
ridge_alpha = ridge_pipeline.named_steps['regressor'].alpha_
y_pred_ridge = ridge_pipeline.predict(X_test)
y_pred_ridge = np.clip(y_pred_ridge, 0, 4)
r2_ridge = r2_score(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
print(f"   RidgeCV - Alpha Terbaik: {ridge_alpha:.4f}")
print(f"   RidgeCV - R2 Test: {r2_ridge:.4f}, MAE Test: {mae_ridge:.4f}")

# B. RandomForestRegressor dengan GridSearchCV
print("\n   --- Menguji RandomForestRegressor dengan GridSearchCV ---")
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', RandomForestRegressor(random_state=42))])
param_grid_rf = {'regressor__n_estimators': [50, 100], 'regressor__max_depth': [5, 10, None], 'regressor__min_samples_split': [2, 5]}
grid_search_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=min(3, len(X_train)//5 if len(X_train)>9 else 1), scoring='r2', verbose=0, n_jobs=-1) # Adjusted CV
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_
print(f"   RandomForest - Parameter Terbaik: {grid_search_rf.best_params_}")
print(f"   RandomForest - Skor R2 CV Terbaik: {grid_search_rf.best_score_:.4f}")
y_pred_rf = best_rf_model.predict(X_test)
y_pred_rf = np.clip(y_pred_rf, 0, 4)
r2_rf = r2_score(y_test, y_pred_rf); mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f"   RandomForest - R2 Test: {r2_rf:.4f}, MAE Test: {mae_rf:.4f}")

# C. XGBoost dengan GridSearchCV
print("\n   --- Menguji XGBoost dengan GridSearchCV ---")
xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))])
param_grid_xgb = {'regressor__n_estimators': [50, 100], 'regressor__max_depth': [3, 5],'regressor__learning_rate': [0.05, 0.1]}
grid_search_xgb = GridSearchCV(xgb_pipeline, param_grid_xgb, cv=min(3, len(X_train)//5 if len(X_train)>9 else 1), scoring='r2', verbose=0, n_jobs=-1) # Adjusted CV
grid_search_xgb.fit(X_train, y_train)
best_xgb_model = grid_search_xgb.best_estimator_
print(f"   XGBoost - Parameter Terbaik: {grid_search_xgb.best_params_}")
print(f"   XGBoost - Skor R2 CV Terbaik: {grid_search_xgb.best_score_:.4f}")
y_pred_xgb = best_xgb_model.predict(X_test)
y_pred_xgb = np.clip(y_pred_xgb, 0, 4)
r2_xgb = r2_score(y_test, y_pred_xgb); mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
print(f"   XGBoost - R2 Test: {r2_xgb:.4f}, MAE Test: {mae_xgb:.4f}")

# --- 9. MEMILIH MODEL TERBAIK KESELURUHAN ---
models_performance = {
    "RidgeCV": {"model": ridge_pipeline, "r2_cv": r2_ridge, "r2_test": r2_ridge, "mae_test": mae_ridge, "params": {'alpha': ridge_alpha}}, # r2_cv untuk ridge adalah r2_test nya sendiri
    "RandomForest": {"model": best_rf_model, "r2_cv": grid_search_rf.best_score_, "r2_test": r2_rf, "mae_test": mae_rf, "params": grid_search_rf.best_params_},
    "XGBoost": {"model": best_xgb_model, "r2_cv": grid_search_xgb.best_score_, "r2_test": r2_xgb, "mae_test": mae_xgb, "params": grid_search_xgb.best_params_}
}
# Pilih berdasarkan R2 CV terbaik, tapi jika semua negatif, R2 test tertinggi bisa jadi pertimbangan (hati-hati)
best_model_name = None
# Jika semua CV score negatif, pilih yang paling mendekati 0. Jika ada yg positif, pilih yg tertinggi.
# Atau, bisa juga prioritaskan R2 test jika CV semua buruk, tapi ini kurang ideal.
# Untuk sekarang, kita pilih berdasarkan CV score yang paling tinggi (paling tidak negatif).
best_cv_r2_overall = -np.inf
for name, perf in models_performance.items():
    if perf['r2_cv'] > best_cv_r2_overall:
        best_cv_r2_overall = perf['r2_cv']
        best_model_name = name

chosen_model_pipeline = models_performance[best_model_name]['model']
chosen_model_params = models_performance[best_model_name]['params']
chosen_r2_cv = models_performance[best_model_name]['r2_cv']
chosen_r2_test = models_performance[best_model_name]['r2_test']
chosen_mae_test = models_performance[best_model_name]['mae_test']

print("\n" + "="*50)
print("🏆 MODEL TERBAIK KESELURUHAN (Berdasarkan R2 CV Tertinggi) 🏆")
print(f"   Model Pilihan: {best_model_name}")
print(f"   Parameter Terbaik: {chosen_model_params}")
print(f"   Skor R2 CV Terbaik: {chosen_r2_cv:.4f}")
print(f"   R2 Score (Test): {chosen_r2_test:.4f}")
print(f"   MAE (Test): {chosen_mae_test:.4f}")
if chosen_r2_test < 0: print("   ⚠️ PERINGATAN: R2 Score TEST model terbaik negatif.")
elif chosen_r2_test < 0.1: print("   ⚠️ PERINGATAN: R2 Score TEST model terbaik sangat rendah.")


# --- 10. EVALUASI BOOTSTRAPPING UNTUK MODEL TERBAIK ---
print("\n🔄 Melakukan Evaluasi Bootstrapping untuk Model Terbaik...")
n_bootstraps = 100 # Kurangi jika terlalu lama, idealnya 1000
bootstrap_maes = []
bootstrap_r2s = []

if X_train.shape[0] < 2 : # Cek apakah X_train punya cukup sampel untuk resample
    print("   ⚠️ Jumlah sampel X_train terlalu sedikit untuk bootstrapping (<2). Skipping bootstrap.")
else:
    for i in range(n_bootstraps):
        X_bs, y_bs = resample(X_train, y_train, random_state=i)
        if X_bs.empty: continue # Skip jika bootstrap sample kosong
        
        # Perlu melatih ulang model terbaik pada data bootstrap
        # Kita perlu membuat ulang pipeline dengan parameter terbaik
        # Ini agak rumit jika model terbaik adalah RidgeCV karena alpha-nya sudah ditentukan.
        # Untuk RF dan XGB, kita bisa ambil regressornya dan parameter terbaiknya.
        
        current_best_regressor_name = best_model_name
        if 'RidgeCV' in current_best_regressor_name:
            # RidgeCV sudah melakukan CV sendiri, bootstrapping mungkin kurang memberi info baru utk alpha
            # Kita bisa fit pipeline yang sudah ada
            model_to_bootstrap = chosen_model_pipeline 
            model_to_bootstrap.fit(X_bs, y_bs)
        else: # Untuk RF atau XGBoost
            # Ambil langkah preprocessor dari pipeline model terbaik
            preprocessor_step = chosen_model_pipeline.named_steps['preprocessor']
            
            # Buat instance regressor baru dengan parameter terbaik
            if 'RandomForest' in current_best_regressor_name:
                regressor_instance = RandomForestRegressor(**{k.replace('regressor__',''):v for k,v in chosen_model_params.items()}, random_state=42)
            elif 'XGBoost' in current_best_regressor_name:
                regressor_instance = XGBRegressor(**{k.replace('regressor__',''):v for k,v in chosen_model_params.items()}, objective='reg:squarederror', random_state=42)
            else: # Fallback, seharusnya tidak terjadi
                print(f"   Model {current_best_regressor_name} tidak didukung untuk bootstrap manual regressor. Skipping.")
                break 
            
            model_to_bootstrap = Pipeline(steps=[('preprocessor', preprocessor_step), ('regressor', regressor_instance)])
            model_to_bootstrap.fit(X_bs, y_bs)

        y_pred_bs_test = model_to_bootstrap.predict(X_test)
        y_pred_bs_test = np.clip(y_pred_bs_test, 0, 4)
        bootstrap_maes.append(mean_absolute_error(y_test, y_pred_bs_test))
        bootstrap_r2s.append(r2_score(y_test, y_pred_bs_test))

    if bootstrap_maes and bootstrap_r2s:
        print(f"   Hasil Bootstrap MAE (Test): {np.mean(bootstrap_maes):.4f} ± {np.std(bootstrap_maes):.4f}")
        print(f"   Hasil Bootstrap R2 (Test) : {np.mean(bootstrap_r2s):.4f} ± {np.std(bootstrap_r2s):.4f}")
        # Plot distribusi MAE
        plt.figure(figsize=(10,4))
        sns.histplot(bootstrap_maes, kde=True)
        plt.title('Distribusi MAE dari Bootstrapping')
        plt.xlabel('Mean Absolute Error (Test Set)')
        plt.ylabel('Frekuensi')
        plt.show()
    else:
        print("   Tidak bisa melakukan bootstrapping (mungkin karena data terlalu sedikit).")


# --- 11. PENYIMPANAN MODEL TERBAIK AKHIR ---
MODEL_DIR = 'ml_model_ip_sem3_robust_eval'
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILENAME = os.path.join(MODEL_DIR, 'best_ip_sem3_predictor_pipeline.pkl')
with open(MODEL_FILENAME, 'wb') as f: pickle.dump(chosen_model_pipeline, f)

# Kolom asli SEBELUM preprocessing (yang akan jadi input form)
original_feature_columns = X_train.columns.tolist() # Ambil dari X_train sebelum diproses pipeline
COLUMNS_FILENAME = os.path.join(MODEL_DIR, 'original_columns_for_sem3_pred.pkl')
with open(COLUMNS_FILENAME, 'wb') as f: pickle.dump(original_feature_columns, f)

ASSESSMENT_MAP_FILENAME = os.path.join(MODEL_DIR, 'assessment_cols_map.pkl')
with open(ASSESSMENT_MAP_FILENAME, 'wb') as f: pickle.dump(ASSESSMENT_COLS_MAP, f)

print(f"\n✅ Model Pipeline Terbaik ({best_model_name}), Kolom Asli, dan Assessment Map telah disimpan di folder '{MODEL_DIR}'")
print(f"🎉 Training Selesai!")
print(f"   Untuk prediksi IP Semester 4, Anda akan menggunakan model ini dengan memberikan")
print(f"   data Semester 2 dan Semester 3 sebagai input (sesuai struktur fitur yang dilatih).")