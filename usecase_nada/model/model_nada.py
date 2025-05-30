# usecase_nada/ml_development/MLmodel.py

import os
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine # Tambahkan ini
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Hapus import yang berhubungan dengan Django jika tidak ingin menggunakannya untuk konfigurasi
# from django.conf import settings
# from django.db import connection

# --- Konfigurasi Database (Mirip Usecase Miko) ---
DB_USER = "postgres"
DB_PASSWORD = "DBmiko" # Ganti dengan password Anda yang sebenarnya jika berbeda
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "dbexam"
DB_CONNECTION_STRING = f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'


# --- Konfigurasi Mapping Implisit Course ke Karir ---
COURSE_CAREER_MAPPING = {
    1: 'Cyber Security', 2: 'Network Administrator', 3: 'Database Administrator',
    4: 'Software Developer', 5: 'IT Consultant', 6: 'System Analyst',
    7: 'Web Developer', 8: 'Mobile Developer', 9: 'UI/UX Designer',
    10: 'Cloud Engineer', 11: 'Data Scientist', 12: 'Business Analyst',
    13: 'Data Engineer', 14: 'Financial Analyst', 15: 'Accounting',
    16: 'Marketing Specialist', 17: 'Human Resources', 18: 'Project Manager',
    19: 'Operations Manager', 20: 'Supply Chain Manager',
}

# --- Fungsi untuk Membuat Dummy Data (Fallback) ---
def create_dummy_training_data():
    print("🔄 Creating dummy training data as a fallback...")
    dummy_data = {
        'stu_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135],
        'gender': ['female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female'],
        'dept_id': [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1],
        'course_id_1': [11, 13, 11, 15, 1, 11, 13, 15, 1, 11, 15, 11, 13, 15, 1, 11, 13, 15, 1, 11, 4, 7, 10, 14, 18, 20, 5, 12, 16, 17, 19, 3, 6, 9, 2],
        'grade_c1': [85, 75, 90, 60, 88, 72, 95, 65, 80, 70, 40, 85, 70, 50, 92, 78, 60, 80, 75, 55, 90, 85, 88, 75, 92, 80, 78, 70, 65, 80, 72, 90, 82, 75, 68],
        'attendance_c1': [90, 80, 95, 70, 92, 85, 98, 75, 88, 82, 50, 90, 78, 60, 95, 80, 70, 85, 88, 65, 95, 90, 92, 80, 95, 85, 80, 75, 70, 85, 78, 92, 88, 80, 72],
        'course_id_2': [13, 11, 13, 11, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 12, 8, 5, 10, 19, 16, 18, 4, 7, 3, 6, 9, 2, 14, 17],
        'grade_c2': [70, 80, 75, 50, 80, 65, 85, 45, 75, 60, 30, 78, 65, 40, 85, 70, 55, 70, 68, 50, 75, 80, 70, 65, 80, 72, 60, 78, 55, 68, 58, 80, 70, 60, 70],
        'attendance_c2': [80, 85, 82, 60, 88, 75, 90, 55, 82, 70, 40, 85, 70, 50, 90, 75, 65, 80, 75, 60, 85, 90, 80, 75, 90, 82, 70, 85, 65, 78, 68, 90, 80, 70, 75],
        'course_id_3': [15, 15, 15, 13, 15, 15, 15, 11, 15, 15, 13, 15, 15, 11, 15, 15, 15, 11, 15, 15, 7, 14, 17, 18, 20, 5, 12, 16, 19, 4, 8, 10, 3, 6, 9],
        'grade_c3': [60, 65, 55, 30, 70, 50, 75, 20, 65, 40, 20, 60, 50, 30, 70, 55, 40, 60, 58, 45, 70, 60, 65, 50, 75, 68, 50, 55, 45, 60, 52, 70, 62, 58, 48],
        'attendance_c3': [70, 75, 65, 40, 80, 60, 85, 30, 75, 50, 30, 70, 60, 40, 80, 65, 50, 70, 68, 55, 80, 70, 75, 60, 85, 78, 60, 65, 55, 70, 62, 80, 72, 68, 58],
        'actual_career': [
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
    df = None
    db_engine = None # Inisialisasi db_engine

    # --- 1. Koneksi ke Database & Ekstraksi Data ---
    print("\n--- Tahap 1: Koneksi & Ekstraksi Data dari Database ---")
    try:
        db_engine = create_engine(DB_CONNECTION_STRING)
        print(f"✅ Koneksi database via SQLAlchemy berhasil: {DB_CONNECTION_STRING}")
        
        sql_query = """
        SELECT
            s.stu_id,
            s.gender,
            s.dept_id,
            e.course_id,
            e.grade, 
            a.attendance_percentage
        FROM
            public.student s
        JOIN
            public.enrollment e ON s.stu_id = e.stu_id
        JOIN
            public.attendance a ON e.enroll_id = a.enroll_id
        ORDER BY s.stu_id, e.semester_id, e.course_id;
        """
        raw_df = pd.read_sql_query(sql_query, db_engine)
        print(f"✅ Successfully loaded {len(raw_df)} rows from PostgreSQL using SQLAlchemy.")

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
                        student_features[f'grade_c{i+1}'] = int(group['grade'].iloc[i])
                        student_features[f'attendance_c{i+1}'] = int(group['attendance_percentage'].iloc[i])
                    else:
                        student_features[f'course_id_{i+1}'] = 0
                        student_features[f'grade_c{i+1}'] = 0
                        student_features[f'attendance_c{i+1}'] = 0
                
                main_course_id = group['course_id'].iloc[0] if not group.empty else None
                student_features['actual_career'] = COURSE_CAREER_MAPPING.get(main_course_id, 'Other')
                processed_data.append(student_features)
            
            df_processed = pd.DataFrame(processed_data)
            print(f"📊 Data aggregated. {len(df_processed)} students processed.")
            
            df = df_processed[df_processed['actual_career'] != 'Other']
            print(f"🔎 After filtering 'Other' careers, {len(df)} rows remaining.")
            
            if df.empty:
                print("⚠️ Warning: Processed data is empty after filtering 'Other' careers.")
                df = create_dummy_training_data()
                print("🔄 Using dummy data because real data became empty after processing.")

    except Exception as e:
        print(f"❌ Error during database connection or data loading using SQLAlchemy: {e}")
        print("🔄 Falling back to dummy data for training due to an error.")
        df = create_dummy_training_data()
    finally:
        if db_engine:
            db_engine.dispose() # Tutup koneksi engine SQLAlchemy
            print("ℹ️ SQLAlchemy engine disposed.")


    if df is None or df.empty:
        print("❌ FATAL: DataFrame is None or empty even after trying dummy data. Cannot proceed.")
        return

    print(f"\nTotal data to be used for training: {len(df)} rows.")
    print("Sample of training data (first 5 rows):")
    print(df.head())

    # --- 3. Praproses Fitur ---
    print("\n--- Tahap 3: Praproses Fitur ---")
    categorical_to_convert = ['course_id_1', 'course_id_2', 'course_id_3', 'dept_id']
    for col in categorical_to_convert:
        if col in df.columns:
            df[col] = df[col].astype(str)
        else:
            print(f"⚠️ Warning: Categorical column '{col}' not found. May cause errors in preprocessor.")

    numeric_features = [col for col in df.columns if 'grade_c' in col or 'attendance_c' in col]
    if 'gender' not in df.columns:
        print("⚠️ Warning: Column 'gender' not found. Using categorical features list without 'gender'.")
        categorical_features = [col for col in categorical_to_convert if col in df.columns]
    else:
         categorical_features = [col for col in categorical_to_convert if col in df.columns] + ['gender']

    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]

    if not numeric_features and not categorical_features:
        print("❌ FATAL: No numeric or categorical features identified. Cannot proceed.")
        return
    
    print(f"Numeric Features: {numeric_features}")
    print(f"Categorical Features: {categorical_features}")

    transformers_list = []
    if numeric_features:
        transformers_list.append(('num', StandardScaler(), numeric_features))
    if categorical_features:
        transformers_list.append(('cat', OneHotEncoder(handle_unknown='ignore', drop='first' if len(categorical_features) > 1 else None), categorical_features))

    if not transformers_list:
        print("❌ FATAL: No transformers could be created (no numeric or categorical features).")
        return
        
    preprocessor = ColumnTransformer(transformers=transformers_list, remainder='drop')

    # --- 4. Bangun Pipeline Model ML ---
    print("\n--- Tahap 4: Membangun dan Melatih Model ---")
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', LogisticRegression(max_iter=1000, solver='liblinear', random_state=42))])
    
    feature_columns = numeric_features + categorical_features
    if not all(feature in df.columns for feature in feature_columns):
        print(f"❌ FATAL: One or more feature columns are missing from the DataFrame. Expected: {feature_columns}")
        print(f"Available columns in DataFrame: {df.columns.tolist()}")
        return

    if 'actual_career' not in df.columns:
        print(f"❌ FATAL: Target column 'actual_career' is missing from the DataFrame.")
        return

    X = df[feature_columns]
    y = df['actual_career']

    if X.empty or y.empty:
        print("❌ FATAL: Features (X) or target (y) are empty. Cannot train model.")
        return

    try:
        model_pipeline.fit(X, y)
        print("✅ Model training complete.")
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        return

    # --- 5. Simpan Model ---
    print("\n--- Tahap 5: Menyimpan Model ---")
    try:
        # Model akan disimpan relatif terhadap direktori skrip ini
        model_dir_base = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(model_dir_base, 'ml_models_career_standalone') # Nama folder yang berbeda
        os.makedirs(model_dir, exist_ok=True)
        print(f"Model directory: {model_dir}")

        model_path = os.path.join(model_dir, 'nada_career_predictor_standalone.pkl')
        with open(model_path, 'wb') as file:
            pickle.dump(model_pipeline, file)
        print(f"✅ ML Model saved to: {model_path}")

        features_path = os.path.join(model_dir, 'nada_career_predictor_features_standalone.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump(feature_columns, f)
        print(f"✅ Feature names saved to: {features_path}")
        print("\n🎉 Proses Selesai! 🎉")

    except IOError as e:
        print(f"❌ Error: Could not write model or feature file. Check folder permissions or path. Detail: {e}")
    except Exception as e:
        print(f"❌ Error during model saving: {e}")


if __name__ == '__main__':
    print("🏁 Menjalankan skrip sebagai program utama (standalone)...")
    # Karena kita tidak menggunakan Django lagi untuk koneksi DB atau settings,
    # bagian konfigurasi Django di sini tidak lagi diperlukan.
    train_and_save_career_prediction_model()