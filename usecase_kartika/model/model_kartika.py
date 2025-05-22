import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

class AtRiskModel:
    """
    Model untuk memprediksi siswa berisiko menggunakan Random Forest Classifier
    """
    def __init__(self):
        """Inisialisasi model dan preprocessor"""
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            verbose=1  # Menampilkan progress training di terminal
        )
        self.preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
    
    def load_data(self):
        """
        Membuat dataset dummy untuk simulasi
        Returns:
            pd.DataFrame: Dataset berisi fitur dan target
        """
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'attendance': np.random.normal(75, 15, n_samples).clip(0, 100),
            'midterm': np.random.normal(65, 20, n_samples).clip(0, 100),
            'project': np.random.normal(70, 15, n_samples).clip(0, 100),
            'historical_grade': np.random.normal(72, 18, n_samples).clip(0, 100),
        }
        
        # Menghitung probabilitas risiko
        risk_prob = (
            0.3*(100-data['attendance'])/100 +
            0.3*(100-data['midterm'])/100 +
            0.2*(100-data['project'])/100 +
            0.2*(100-data['historical_grade'])/100
        )
        data['at_risk'] = (risk_prob > 0.5).astype(int)
        
        return pd.DataFrame(data)
    
    def train(self):
        """Melatih model dengan data yang ada"""
        print("\n=== MEMULAI TRAINING MODEL ===")
        df = self.load_data()
        X = df[['attendance', 'midterm', 'project', 'historical_grade']]
        y = df['at_risk']
        
        # Split data training dan testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Preprocessing data
        print("\n=== PREPROCESSING DATA ===")
        X_train = self.preprocessor.fit_transform(X_train)
        X_test = self.preprocessor.transform(X_test)
        
        # Training model
        print("\n=== TRAINING RANDOM FOREST ===")
        self.model.fit(X_train, y_train)
        
        # Evaluasi model
        print("\n=== EVALUASI MODEL ===")
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        # Menyimpan model
        joblib.dump(self.model, 'model/at_risk_model.pkl')
        joblib.dump(self.preprocessor, 'model/preprocessor.pkl')
        print("\nModel berhasil disimpan!")
    
    def predict(self, attendance, midterm, project, historical_grade):
        """
        Memprediksi risiko siswa berdasarkan input
        
        Args:
            attendance: Persentase kehadiran
            midterm: Nilai ujian tengah semester
            project: Nilai proyek
            historical_grade: Nilai historis
            
        Returns:
            dict: Hasil prediksi beserta probabilitas
        """
        try:
            # Load model yang sudah disimpan
            model = joblib.load('model/at_risk_model.pkl')
            preprocessor = joblib.load('model/preprocessor.pkl')
            
            # Membuat dataframe dari input
            input_data = pd.DataFrame([[
                attendance, midterm, project, historical_grade
            ]], columns=['attendance', 'midterm', 'project', 'historical_grade'])
            
            # Preprocessing dan prediksi
            processed_data = preprocessor.transform(input_data)
            prediction = model.predict(processed_data)
            probability = model.predict_proba(processed_data)
            
            # Format hasil
            result = {
                'prediction': 'High Risk' if prediction[0] == 1 else 'Low Risk',
                'probability': float(probability[0][1]),
                'features': {
                    'Attendance': attendance,
                    'Midterm': midterm,
                    'Project': project,
                    'Historical Grade': historical_grade
                }
            }
            
            # Menampilkan hasil di terminal
            print("\n=== HASIL PREDIKSI ===")
            for key, value in result['features'].items():
                print(f"{key}: {value}")
            print(f"\nPrediksi: {result['prediction']}")
            print(f"Probabilitas Risiko: {result['probability']:.2%}")
            
            return result
            
        except Exception as e:
            print(f"\nERROR: {str(e)}")
            return {'error': str(e)}


class CourseRecommender:
    """
    Sistem rekomendasi mata kuliah untuk siswa
    """
    def __init__(self):
        """Inisialisasi recommender system"""
        self.course_features = None
        self.student_features = None
    
    def load_data(self):
        """
        Membuat data dummy untuk simulasi
        Returns:
            tuple: (DataFrame course, DataFrame interactions)
        """
        np.random.seed(42)
        # Data mata kuliah
        courses = pd.DataFrame({
            'course_id': range(1, 21),
            'difficulty': np.random.choice(['Easy', 'Medium', 'Hard'], 20),
            'department': np.random.choice([1, 2], 20)
        })
        
        # Data interaksi siswa-mata kuliah
        interactions = []
        for stu_id in range(1, 201):
            for course_id in range(1, 21):
                if np.random.random() > 0.7:
                    interactions.append({
                        'stu_id': stu_id,
                        'course_id': course_id,
                        'grade': np.random.normal(70, 15).clip(0, 100)
                    })
        
        return courses, pd.DataFrame(interactions)
    
    def recommend(self, student_id):
        """
        Memberikan rekomendasi mata kuliah untuk siswa tertentu
        
        Args:
            student_id: ID siswa
            
        Returns:
            list: Daftar rekomendasi course_id
        """
        courses, interactions = self.load_data()
        
        # Mata kuliah yang sudah diambil
        taken = interactions[interactions['stu_id'] == student_id]['course_id'].unique()
        
        # Rekomendasi mata kuliah yang belum diambil
        recommendations = [c for c in range(1, 21) if c not in taken][:5]
        
        # Menampilkan hasil di terminal
        print(f"\nRekomendasi untuk student {student_id}:")
        for course_id in recommendations:
            print(f"- Course {course_id}")
        
        return recommendations
    