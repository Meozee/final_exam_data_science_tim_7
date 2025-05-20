import psycopg2
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import os
import sys

# ---------- 1. Koneksi ke PostgreSQL ----------
try:
    conn = psycopg2.connect(
        dbname="dbexam",
        user="postgres",
        password="DBmiko",
        host="localhost",
        port="5432"
    )
except Exception as e:
    print("❌ Gagal koneksi ke database:", e)
    sys.exit(1)

# ---------- 2. Query gabungan (tanpa quiz_score) ----------
query = """
SELECT
    e.enroll_id,
    att.attendance_percentage,
    MAX(CASE WHEN ass.assessment_type = 'Midterm' THEN ass.score END)  AS midterm_score,
    MAX(CASE WHEN ass.assessment_type = 'Project' THEN ass.score END)  AS project_score,
    e.grade
FROM enrollment  e
LEFT JOIN attendance  att ON e.enroll_id = att.enroll_id
LEFT JOIN assessment  ass ON e.enroll_id = ass.enroll_id
GROUP BY e.enroll_id, att.attendance_percentage, e.grade
HAVING COUNT(ass.assessment_id) >= 2
"""

# ---------- 3. Load ke DataFrame ----------
df = pd.read_sql(query, conn)
conn.close()

print("📊 Data loaded:")
print(df.head(), "\n")
print(df.info(), "\n")

# ---------- 4. Validasi dan pembersihan ----------
if df.empty:
    print("❌ DataFrame kosong — periksa tabel atau query SQL.")
    sys.exit(1)

df.dropna(inplace=True)

if len(df) < 5:
    print("❌ Data terlalu sedikit untuk training (kurang dari 5 baris).")
    sys.exit(1)

# ---------- 5. Siapkan fitur & target ----------
feature_cols = ['attendance_percentage', 'midterm_score', 'project_score']
X = df[feature_cols]
y = df['grade']

# ---------- 6. Train‑test split & training ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ---------- 7. Evaluasi ----------
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))   # kompatibel lintas versi

print("✅ Model trained successfully!")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# ---------- 8. Simpan model ----------
os.makedirs("ml_models", exist_ok=True)
model_path = "ml_models/miko_grade_predictor.pkl"
joblib.dump(model, model_path)

print(f"✅ Model saved to: {model_path}")
