from cmath import sqrt
import joblib
import pandas as pd
from django.shortcuts import render
from django.db import connection
from .forms import StudentPerformanceForm


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


def train_and_save_model():
    df = get_data_from_db()

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib

    X = df[['attendance_percentage', 'midterm_score', 'project_score', 'difficulty_level']]
    y = df['grade']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Model Trained - RMSE: {rmse:.2f}, R²: {r2:.2f}")

    joblib.dump(model, 'usecase_miko/model/student_grade_model.pkl')


def predict_student_performance(request):
    prediction = None
    if request.method == 'POST':
        form = StudentPerformanceForm(request.POST)
        if form.is_valid():
            attendance = int(form.cleaned_data['attendance_percentage'])
            midterm = int(form.cleaned_data['midterm_score'])
            project = int(form.cleaned_data['project_score'])
            difficulty = {'Easy': 0, 'Medium': 1, 'Hard': 2}[form.cleaned_data['difficulty_level']]

            try:
                model = joblib.load('usecase_miko/model/student_grade_model.pkl')
            except FileNotFoundError:
                train_and_save_model()
                model = joblib.load('usecase_miko/model/student_grade_model.pkl')

            prediction = model.predict([[attendance, midterm, project, difficulty]])[0]

    else:
        form = StudentPerformanceForm()

    context = {
        'form': form,
        'prediction': round(prediction, 2) if prediction else None
    }
    return render(request, 'usecase_miko/predict.html', context)