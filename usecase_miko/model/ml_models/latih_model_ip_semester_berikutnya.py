import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import os

print("Mulai pelatihan model prediksi IP semester berikutnya...")

# --- KONEKSI DATABASE ---
DB_USER = "postgres"
DB_PASSWORD = "DBmiko"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "dbexam"

try:
    db_engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    print("✅ Koneksi database berhasil.")
except Exception as e:
    print(f"❌ Koneksi gagal: {e}")
    exit()

# --- FUNGSI KONVERSI GRADE KE POINT ---
def grade_to_point(grade):
    try:
        grade = float(grade)
        if grade >= 85: return 4.0
        elif grade >= 75: return 3.0
        elif grade >= 65: return 2.0
        elif grade >= 55: return 1.0
        else: return 0.0
    except:
        return 0.0

# --- AMBIL DATA ---
try:
    enrollment_df = pd.read_sql("SELECT enroll_id, stu_id, semester_id, grade FROM enrollment WHERE grade IS NOT NULL", db_engine)
    student_df = pd.read_sql("SELECT stu_id, gender, dept_id FROM student", db_engine)
    attendance_df = pd.read_sql("SELECT enroll_id, attendance_percentage FROM attendance", db_engine)
    print("✅ Data berhasil diambil.")
except Exception as e:
    print(f"❌ Gagal mengambil data: {e}")
    exit()

# Tambah kolom SKS dan hitung kualitas nilai
enrollment_df['sks'] = 3
enrollment_df['points'] = enrollment_df['grade'].apply(grade_to_point)
enrollment_df['quality_points'] = enrollment_df['points'] * enrollment_df['sks']

# IP per semester
ip_per_semester_df = enrollment_df.groupby(['stu_id', 'semester_id']).agg(
    total_quality_points=('quality_points', 'sum'),
    total_sks_taken=('sks', 'sum')
).reset_index()
ip_per_semester_df['ip_semester'] = (ip_per_semester_df['total_quality_points'] / ip_per_semester_df['total_sks_taken']).clip(0, 4)

# Attendance
attendance_df = attendance_df.merge(enrollment_df[['enroll_id', 'stu_id', 'semester_id']].drop_duplicates(), on='enroll_id', how='left')
attendance_per_semester_df = attendance_df.groupby(['stu_id', 'semester_id'])['attendance_percentage'].mean().reset_index()
attendance_per_semester_df.rename(columns={'attendance_percentage': 'avg_attendance_semester'}, inplace=True)

# --- DATASET TRAINING ---
training_instances = []
for stu_id in ip_per_semester_df['stu_id'].unique():
    student_ip_history = ip_per_semester_df[ip_per_semester_df['stu_id'] == stu_id].sort_values(by='semester_id')
    student_attendance_history = attendance_per_semester_df[attendance_per_semester_df['stu_id'] == stu_id].sort_values(by='semester_id')
    
    if len(student_ip_history) < 2:
        continue  # Tidak cukup data

    for i in range(len(student_ip_history) - 1):
        sem_n = student_ip_history.iloc[i]
        sem_np1 = student_ip_history.iloc[i + 1]

        sem_id = sem_n['semester_id']
        ip_sem = sem_n['ip_semester']

        semesters_up_to_n = student_ip_history[student_ip_history['semester_id'] <= sem_id]
        ipk_kumulatif = semesters_up_to_n['total_quality_points'].sum() / semesters_up_to_n['total_sks_taken'].sum()
        
        att_up_to_n = student_attendance_history[student_attendance_history['semester_id'] <= sem_id]
        avg_att_kumulatif = att_up_to_n['avg_attendance_semester'].mean() if not att_up_to_n.empty else 0
        
        stu_info = student_df[student_df['stu_id'] == stu_id]
        if stu_info.empty:
            continue

        training_instances.append({
            'ipk_saat_ini': ipk_kumulatif,
            'ip_semester_lalu': ip_sem,
            'rata_rata_kehadiran_kumulatif': avg_att_kumulatif,
            'departemen': stu_info['dept_id'].values[0],
            'gender': stu_info['gender'].values[0],
            'semester_ke_input': sem_id,
            'TARGET_IP_SEMESTER_BERIKUTNYA': sem_np1['ip_semester']
        })

# --- DATAFRAME FINAL ---
ml_df = pd.DataFrame(training_instances).dropna()
if ml_df.empty:
    print("❌ Dataset kosong setelah pemrosesan.")
    exit()

print(f"📊 Jumlah data training: {len(ml_df)}")

# --- PREPROCESSING ---
df_encoded = pd.get_dummies(ml_df, columns=['departemen', 'gender', 'semester_ke_input'], drop_first=True)
X = df_encoded.drop('TARGET_IP_SEMESTER_BERIKUTNYA', axis=1)
y = df_encoded['TARGET_IP_SEMESTER_BERIKUTNYA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- PIPELINE ---
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(f"📈 Evaluasi Model:")
print(f" - R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f" - MAE: {mean_absolute_error(y_test, y_pred):.4f}")

# --- SIMPAN MODEL ---
model_dir = 'ml_models'
os.makedirs(model_dir, exist_ok=True)

MODEL_FILENAME = os.path.join(model_dir, 'prediksi_ip_semester_berikutnya_rf.pkl')
COLUMNS_FILENAME = os.path.join(model_dir, 'prediksi_ip_semester_berikutnya_rf_columns.pkl')

with open(MODEL_FILENAME, 'wb') as f: pickle.dump(pipeline, f)
with open(COLUMNS_FILENAME, 'wb') as f: pickle.dump(X.columns.tolist(), f)

print(f"✅ Model disimpan ke: {MODEL_FILENAME}")
print(f"✅ Kolom disimpan ke: {COLUMNS_FILENAME}")
