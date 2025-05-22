import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def load_data():
    # Contoh query gabungan dari PostgreSQL
    query = """
    SELECT 
        a.attendance_percentage,
        MAX(CASE WHEN ass.assessment_type = 'Midterm' THEN ass.score END) AS midterm_score,
        MAX(CASE WHEN ass.assessment_type = 'Project' THEN ass.score END) AS project_score,
        cd.difficulty_level,
        e.grade
    FROM enrollment e
    JOIN attendance a ON e.enroll_id = a.enroll_id
    JOIN assessment ass ON e.enroll_id = ass.enroll_id
    JOIN course_difficulty cd ON e.course_id = cd.course_id
    GROUP BY e.enroll_id, a.attendance_percentage, cd.difficulty_level, e.grade;
    """
    
    # Ganti dengan koneksi DB jika ada
    data = pd.read_sql(query, "postgresql://user:password@localhost/dbname")
    
    # Encode difficulty_level
    data['difficulty_level'] = data['difficulty_level'].map({'Easy': 0, 'Medium': 1, 'Hard': 2})
    
    return data

def train_and_save_models(df):
    X = df[['attendance_percentage', 'midterm_score', 'project_score', 'difficulty_level']]
    y = df['grade']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {"RMSE": rmse, "MAE": mae, "R²": r2}
        
        joblib.dump(model, f'model/{name.lower().replace(" ", "_")}_model.pkl')
    
    return results

if __name__ == "__main__":
    df = load_data()
    results = train_and_save_models(df)
    
    print("Model Evaluation Results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")