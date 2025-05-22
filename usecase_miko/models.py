from django.db import models
import joblib
import pandas as pd
from django.db import connection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Model untuk menyimpan riwayat prediksi
class PredictionLog(models.Model):
    attendance_percentage = models.IntegerField()
    midterm_score = models.IntegerField()
    project_score = models.IntegerField()
    difficulty_level = models.CharField(
        max_length=10,
        choices=[
            ('Easy', 'Easy'),
            ('Medium', 'Medium'),
            ('Hard', 'Hard')
        ]
    )
    predicted_grade = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediksi: {self.predicted_grade} pada {self.created_at}"


# Fungsi untuk mengambil data dari database
def get_data_from_db():
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT 
                a.attendance_percentage,
                MAX(CASE WHEN ass.assessment_type = 'Midterm' THEN ass.score END) AS midterm_score,
                MAX(CASE WHEN ass.assessment_type = 'Project' THEN ass.score END) AS project_score,
                cd.difficulty_level,
                e.grade
            FROM public.enrollment e
            JOIN public.attendance a ON e.enroll_id = a.enroll_id
            JOIN public.assessment ass ON e.enroll_id = ass.enroll_id
            JOIN public.course_difficulty cd ON e.course_id = cd.course_id
            GROUP BY e.enroll_id, a.attendance_percentage, cd.difficulty_level, e.grade;
        """)
        columns = [col[0] for col in cursor.description]
        data = cursor.fetchall()

    df = pd.DataFrame(data, columns=columns)
    df['difficulty_level'] = df['difficulty_level'].map({'Easy': 0, 'Medium': 1, 'Hard': 2})
    return df


# Fungsi untuk melatih model dan menyimpannya
def train_and_save_model():
    df = get_data_from_db()

    X = df[['attendance_percentage', 'midterm_score', 'project_score', 'difficulty_level']]
    y = df['grade']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Trained - RMSE: {rmse:.2f}, R²: {r2:.2f}")

    # Simpan model
    joblib.dump(model, 'usecase_miko/model/student_grade_model2.pkl')


# Fungsi untuk memprediksi nilai siswa berdasarkan input
def predict_student_grade(attendance_percentage, midterm_score, project_score, difficulty_level):
    try:
        model = joblib.load('usecase_miko/model/student_grade_model2.pkl')
    except FileNotFoundError:
        train_and_save_model()
        model = joblib.load('usecase_miko/model/student_grade_model2.pkl')

    difficulty_map = {'Easy': 0, 'Medium': 1, 'Hard': 2}
    difficulty_encoded = difficulty_map.get(difficulty_level, 0)

    prediction = model.predict([[
        attendance_percentage,
        midterm_score,
        project_score,
        difficulty_encoded
    ]])

    return round(prediction[0], 2)


# Fungsi opsional: simpan prediksi ke database
def save_prediction_to_db(attendance_percentage, midterm_score, project_score, difficulty_level, predicted_grade):
    PredictionLog.objects.create(
        attendance_percentage=attendance_percentage,
        midterm_score=midterm_score,
        project_score=project_score,
        difficulty_level=difficulty_level,
        predicted_grade=predicted_grade
    )