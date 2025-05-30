import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

print("Mulai melatih model baru untuk input manual...")

# --- KONEKSI DATABASE (SESUAIKAN) ---
DB_USER = "postgres"
DB_PASSWORD = "DBmiko"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "dbexam"
db_engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# --- FUNGSI BANTU ---
def grade_to_point(grade):
    if grade >= 85: return 4.0
    if grade >= 75: return 3.0
    if grade >= 65: return 2.0
    if grade >= 55: return 1.0
    return 0.0

# --- PENGAMBILAN & PENGOLAHAN DATA ---
try:
    enrollment_df = pd.read_sql("SELECT * FROM enrollment", db_engine)
    student_df = pd.read_sql("SELECT * FROM student", db_engine)
    attendance_df = pd.read_sql("SELECT * FROM attendance", db_engine)
    course_difficulty_df = pd.read_sql("SELECT * FROM course_difficulty", db_engine)
    print("Data mentah berhasil diambil.")
except Exception as e:
    print(f"Gagal mengambil data: {e}")
    exit()

# Gabungkan data
df_merged = enrollment_df.merge(student_df, on='stu_id')
df_merged = df_merged.merge(attendance_df, on='enroll_id')
df_merged = df_merged.merge(course_difficulty_df, on='course_id', how='left')

# --- Periksa Kolom Setelah Penggabungan ---
print(f"Kolom setelah penggabungan: {df_merged.columns.tolist()}")

# --- Mengisi Nilai yang Hilang di 'difficulty_level' dengan Nilai Default ---
df_merged['difficulty_level'] = df_merged['difficulty_level'].fillna('medium')  # Misal kita isi dengan 'medium'

# Pastikan kolom 'difficulty_level' telah terisi dengan nilai default
print(f"Jumlah nilai yang hilang di kolom 'difficulty_level': {df_merged['difficulty_level'].isnull().sum()}")

# Hitung IP per semester
df_merged['sks'] = 3  # Asumsi semua SKS = 3
df_merged['points'] = df_merged['grade'].apply(grade_to_point)
df_merged['quality_points'] = df_merged['points'] * df_merged['sks']
ip_per_semester = df_merged.groupby(['stu_id', 'semester_id']).agg(
    total_quality_points=('quality_points', 'sum'),
    total_sks=('sks', 'sum')
).reset_index()
ip_per_semester['ip_semester'] = ip_per_semester['total_quality_points'] / ip_per_semester['total_sks']

# Rata-rata kehadiran per semester
attendance_per_semester = df_merged.groupby(['stu_id', 'semester_id'])['attendance_percentage'].mean().reset_index()

# --- REKAYASA FITUR UNTUK INPUT MANUAL ---
instances = []
for index, row in ip_per_semester.iterrows():
    current_student_id = row['stu_id']
    current_semester_id = row['semester_id']
    
    previous_semesters_ip = ip_per_semester[(
        ip_per_semester['stu_id'] == current_student_id) &
        (ip_per_semester['semester_id'] < current_semester_id)
    ]
    
    if not previous_semesters_ip.empty:
        ipk_kumulatif = previous_semesters_ip['ip_semester'].mean()
        ip_semester_lalu = previous_semesters_ip[previous_semesters_ip['semester_id'] == current_semester_id - 1]['ip_semester'].values[0]
        
        # Kehadiran semester lalu
        prev_attendance_df = attendance_per_semester[(
            attendance_per_semester['stu_id'] == current_student_id) &
            (attendance_per_semester['semester_id'] == current_semester_id - 1)
        ]
        attendance_semester_lalu = prev_attendance_df['attendance_percentage'].values[0] if not prev_attendance_df.empty else 0
    else:  # Kasus semester pertama
        ipk_kumulatif = 0
        ip_semester_lalu = 0
        attendance_semester_lalu = 0
        
    student_info = student_df[student_df['stu_id'] == current_student_id]
    gender = student_info['gender'].values[0]
    departemen = student_info['dept_id'].values[0]
    
    # Akses kolom dengan cara yang lebih aman
    course_difficulty = row.get('difficulty_level', 'medium')  # Gunakan .get() untuk mengakses kolom dengan aman
    
    instances.append({
        'ipk_sekarang': ipk_kumulatif,
        'ip_semester_lalu': ip_semester_lalu,
        'attendance_percentage': attendance_semester_lalu,
        'departemen': departemen,
        'gender': gender,
        'course_difficulty': course_difficulty,
        'TARGET_IP_SEMESTER_DEPAN': row['ip_semester']
    })

ml_df = pd.DataFrame(instances).dropna()

# --- PERSIAPAN, TRAINING, & PENYIMPANAN MODEL ---
df_processed = pd.get_dummies(ml_df, columns=['departemen', 'gender', 'course_difficulty'], drop_first=True)

X = df_processed.drop('TARGET_IP_SEMESTER_DEPAN', axis=1)
y = df_processed['TARGET_IP_SEMESTER_DEPAN']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Coba model RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Evaluasi Model - Mean Squared Error: {mse:.4f}")

# Simpan model yang baru
MODEL_FILENAME = 'ml_models/student_peer_group_predictor_rf.pkl'
COLUMNS_FILENAME = 'ml_models/student_peer_group_model_columns_rf.pkl'

with open(MODEL_FILENAME, 'wb') as f:
    pickle.dump(model, f)

with open(COLUMNS_FILENAME, 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print(f"Model disimpan ke: {MODEL_FILENAME}")
print(f"Kolom model disimpan ke: {COLUMNS_FILENAME}")
