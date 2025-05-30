import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.decomposition import PCA
import pickle
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("🚀 ENHANCED Model IP Semester Berikutnya - ADVANCED OPTIMIZATION")

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
    if grade is None: return 0.0
    try:
        grade_float = float(grade)
        if grade_float >= 85: return 4.0
        elif grade_float >= 75: return 3.0
        elif grade_float >= 65: return 2.0
        elif grade_float >= 55: return 1.0
        else: return 0.0
    except (ValueError, TypeError):
        return 0.0

# --- ENHANCED FEATURE ENGINEERING ---
def create_advanced_features(df):
    """Membuat fitur-fitur yang lebih informatif dengan teknik advanced"""
    # Basic trend features
    df['ip_trend'] = df.groupby('stu_id')['ip_semester'].diff().fillna(0)
    df['ip_consistency'] = df.groupby('stu_id')['ip_semester'].transform(
        lambda x: x.expanding().std().fillna(0)
    )
    
    # Advanced trend analysis
    df['ip_trend_acceleration'] = df.groupby('stu_id')['ip_trend'].diff().fillna(0)
    df['ip_volatility'] = df.groupby('stu_id')['ip_semester'].transform(
        lambda x: x.rolling(window=3, min_periods=1).std().fillna(0)
    )
    
    # Performance momentum
    df['positive_trend_count'] = df.groupby('stu_id')['ip_trend'].transform(
        lambda x: (x > 0).rolling(window=3, min_periods=1).sum()
    )
    
    # Seasonal patterns
    df['semester_type'] = df['semester_id'] % 2  # Ganjil/Genap
    
    # Performance categories with more granularity
    df['performance_category'] = pd.cut(df['ip_semester'], 
                                        bins=[0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 
                                        labels=['Very_Poor', 'Poor', 'Below_Avg', 'Average', 'Good', 'Excellent'])
    
    # Exponential weighted moving average (lebih sensitif terhadap data terbaru)
    df['ip_ewm'] = df.groupby('stu_id')['ip_semester'].transform(
        lambda x: x.ewm(span=3).mean()
    )
    
    return df

# --- AMBIL DATA ---
try:
    enrollment_df = pd.read_sql("SELECT enroll_id, stu_id, semester_id, grade FROM enrollment WHERE grade IS NOT NULL", db_engine)
    student_df = pd.read_sql("SELECT stu_id, gender, dept_id FROM student", db_engine)
    attendance_df = pd.read_sql("SELECT enroll_id, attendance_percentage FROM attendance", db_engine)
    print("✅ Data berhasil diambil.")
except Exception as e:
    print(f"❌ Gagal mengambil data: {e}")
    exit()

# --- ENHANCED PREPROCESSING ---
enrollment_df['sks'] = 3
enrollment_df['points'] = enrollment_df['grade'].apply(grade_to_point)
enrollment_df['quality_points'] = enrollment_df['points'] * enrollment_df['sks']

ip_per_semester_df = enrollment_df.groupby(['stu_id', 'semester_id']).agg(
    total_quality_points=('quality_points', 'sum'),
    total_sks_taken=('sks', 'sum'),
    mata_kuliah_count=('enroll_id', 'count'),
    avg_grade=('grade', 'mean'),
    grade_std=('grade', 'std'),  # Tambahan: variabilitas nilai
    min_grade=('grade', 'min'),   # Tambahan: nilai terburuk
    max_grade=('grade', 'max')    # Tambahan: nilai terbaik
).reset_index()

ip_per_semester_df['ip_semester'] = (
    ip_per_semester_df['total_quality_points'] / ip_per_semester_df['total_sks_taken']
).fillna(0).clip(0, 4)

# Fill NaN values untuk grade_std
ip_per_semester_df['grade_std'] = ip_per_semester_df['grade_std'].fillna(0)

ip_per_semester_df = create_advanced_features(ip_per_semester_df)

# Enhanced attendance features
attendance_df = attendance_df.merge(
    enrollment_df[['enroll_id', 'stu_id', 'semester_id']].drop_duplicates(), 
    on='enroll_id', how='left'
)
attendance_per_semester_df = attendance_df.groupby(['stu_id', 'semester_id']).agg(
    avg_attendance_semester=('attendance_percentage', 'mean'),
    attendance_consistency=('attendance_percentage', 'std'),
    min_attendance=('attendance_percentage', 'min'),
    max_attendance=('attendance_percentage', 'max')
).reset_index()
attendance_per_semester_df = attendance_per_semester_df.fillna(0)

# --- ENHANCED DATASET CREATION ---
training_instances = []
for stu_id in ip_per_semester_df['stu_id'].unique():
    student_ip_history = ip_per_semester_df[ip_per_semester_df['stu_id'] == stu_id].sort_values(by='semester_id')
    student_attendance_history = attendance_per_semester_df[attendance_per_semester_df['stu_id'] == stu_id].sort_values(by='semester_id')
    
    if len(student_ip_history) < 3:
        continue

    for i in range(2, len(student_ip_history)):
        current_data = student_ip_history.iloc[i-1]
        target_data = student_ip_history.iloc[i]
        historical_data = student_ip_history.iloc[:i]
        
        # Basic features
        ipk_kumulatif = (historical_data['total_quality_points'].sum() / 
                       historical_data['total_sks_taken'].sum()) if historical_data['total_sks_taken'].sum() > 0 else 0
        
        # Enhanced historical analysis
        ip_trend_avg = historical_data['ip_trend'].mean()
        ip_std = historical_data['ip_semester'].std() if len(historical_data) > 1 else 0
        ip_last_3_avg = historical_data['ip_semester'].tail(3).mean() if len(historical_data) >= 3 else historical_data['ip_semester'].mean()
        ip_last_2_avg = historical_data['ip_semester'].tail(2).mean() if len(historical_data) >= 2 else historical_data['ip_semester'].mean()
        
        # Performance trajectory
        ip_first_half = historical_data['ip_semester'].iloc[:len(historical_data)//2].mean() if len(historical_data) >= 4 else ipk_kumulatif
        ip_second_half = historical_data['ip_semester'].iloc[len(historical_data)//2:].mean() if len(historical_data) >= 4 else ipk_kumulatif
        improvement_trend = ip_second_half - ip_first_half
        
        # Recovery patterns
        low_performance_count = (historical_data['ip_semester'] < 2.5).sum()
        recovery_after_low = 0
        if len(historical_data) >= 2:
            for j in range(1, len(historical_data)):
                if historical_data.iloc[j-1]['ip_semester'] < 2.5 and historical_data.iloc[j]['ip_semester'] >= 2.5:
                    recovery_after_low += 1
        
        # Attendance features
        attendance_current = student_attendance_history[
            student_attendance_history['semester_id'] == current_data['semester_id']
        ]
        avg_attendance = attendance_current['avg_attendance_semester'].iloc[0] if not attendance_current.empty else 0
        attendance_consistency = attendance_current['attendance_consistency'].iloc[0] if not attendance_current.empty else 0
        min_attendance = attendance_current['min_attendance'].iloc[0] if not attendance_current.empty else 0
        
        # Student demographics
        student_info = student_df[student_df['stu_id'] == stu_id]
        if student_info.empty:
            continue
            
        gender = student_info['gender'].values[0]
        departemen = student_info['dept_id'].values[0]
        
        # Academic load features
        total_courses_taken = historical_data['mata_kuliah_count'].sum()
        avg_courses_per_semester = total_courses_taken / len(historical_data)
        
        training_instances.append({
            # Basic features
            'ipk_kumulatif': ipk_kumulatif,
            'ip_semester_terakhir': current_data['ip_semester'],
            'ip_semester_2_terakhir': historical_data['ip_semester'].iloc[-2] if len(historical_data) >= 2 else current_data['ip_semester'],
            
            # Trend features
            'ip_trend_rata': ip_trend_avg,
            'ip_std_deviasi': ip_std,
            'ip_3_semester_terakhir': ip_last_3_avg,
            'ip_2_semester_terakhir': ip_last_2_avg,
            'improvement_trend': improvement_trend,
            
            # Performance patterns
            'mata_kuliah_count': current_data['mata_kuliah_count'],
            'avg_grade_terakhir': current_data['avg_grade'],
            'grade_consistency': current_data['grade_std'],
            'min_grade_semester': current_data['min_grade'],
            'max_grade_semester': current_data['max_grade'],
            
            # Recovery & resilience
            'low_performance_count': low_performance_count,
            'recovery_after_low': recovery_after_low,
            
            # Attendance
            'kehadiran_semester_ini': avg_attendance,
            'konsistensi_kehadiran': attendance_consistency,
            'min_kehadiran': min_attendance,
            
            # Academic load
            'total_courses_taken': total_courses_taken,
            'avg_courses_per_semester': avg_courses_per_semester,
            
            # Context
            'semester_ke': current_data['semester_id'],
            'semester_type': current_data['semester_type'],
            'departemen': departemen,
            'gender': gender,
            
            # Advanced features
            'ip_volatility': current_data['ip_volatility'],
            'positive_trend_count': current_data['positive_trend_count'],
            'ip_ewm': current_data['ip_ewm'],
            
            'TARGET_IP_SEMESTER_BERIKUTNYA': target_data['ip_semester']
        })

if not training_instances:
    print("❌ Tidak ada data yang cukup untuk training.")
    exit()

ml_df = pd.DataFrame(training_instances).dropna()
print(f"📊 Jumlah data training: {len(ml_df)}")
print(f"📊 Target distribution:")
print(ml_df['TARGET_IP_SEMESTER_BERIKUTNYA'].describe())

# --- FEATURE SELECTION & PREPROCESSING ---
categorical_cols = ['departemen', 'gender', 'semester_type']
df_processed = pd.get_dummies(ml_df, columns=categorical_cols, drop_first=True)

X = df_processed.drop('TARGET_IP_SEMESTER_BERIKUTNYA', axis=1)
y = df_processed['TARGET_IP_SEMESTER_BERIKUTNYA']

# Remove extreme outliers (lebih konservatif)
valid_idx = (y >= 0) & (y <= 4.0) & (np.abs(y - y.median()) <= 3 * y.std())
X = X[valid_idx]
y = y[valid_idx]

print(f"📊 Data setelah cleaning outliers: {len(X)}")

# Stratified split dengan lebih banyak bins untuk distribusi yang lebih baik
try:
    y_binned = pd.cut(y, bins=10, duplicates='drop')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_binned)
except:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- BASELINE ---
print("\n" + "="*50)
print("📊 BASELINE PERFORMA")
y_pred_baseline = X_test['ip_semester_terakhir']
baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
baseline_r2 = r2_score(y_test, y_pred_baseline)
print(f"   Baseline MAE: {baseline_mae:.4f}")
print(f"   Baseline R2:  {baseline_r2:.4f}")
print("="*50 + "\n")

# --- ADVANCED MODELS WITH HYPERPARAMETER TUNING ---
def get_models_with_params():
    return {
        'ElasticNet': {
            'model': ElasticNet(random_state=42),
            'params': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        },
        'Ridge': {
            'model': Ridge(random_state=42),
            'params': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            }
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'C': [0.1, 1.0, 10.0],
                'epsilon': [0.01, 0.1, 0.2],
                'kernel': ['rbf', 'linear']
            }
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 4, 6],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            }
        }
    }

scalers = {
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler(),
    'QuantileTransformer': QuantileTransformer(n_quantiles=100, random_state=42),
    'NoScaler': None
}

# Feature selection methods
def apply_feature_selection(X_train, X_test, y_train, method='selectk', k=15):
    if method == 'selectk':
        selector = SelectKBest(score_func=f_regression, k=min(k, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        selected_features = X_train.columns[selector.get_support()].tolist()
        return X_train_selected, X_test_selected, selector, selected_features
    elif method == 'pca':
        pca = PCA(n_components=min(k, X_train.shape[1]))
        X_train_selected = pca.fit_transform(X_train)
        X_test_selected = pca.transform(X_test)
        return X_train_selected, X_test_selected, pca, [f'PC{i+1}' for i in range(pca.n_components_)]
    else:
        return X_train, X_test, None, X_train.columns.tolist()

# --- COMPREHENSIVE MODEL EVALUATION ---
best_cv_score = -np.inf
best_model = None
best_scaler = None
best_config = None
best_feature_selector = None
best_selected_features = None

models_with_params = get_models_with_params()

print("🔍 COMPREHENSIVE MODEL SEARCH WITH HYPERPARAMETER TUNING")
print("="*70)

results = []

for model_name, model_config in models_with_params.items():
    print(f"\n🧪 Testing {model_name}...")
    
    for scaler_name, scaler in scalers.items():
        for fs_method in ['none', 'selectk']:  # Feature selection methods
            try:
                # Apply scaling
                if scaler is not None:
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    X_train_final = pd.DataFrame(X_train_scaled, columns=X_train.columns)
                    X_test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns)
                else:
                    X_train_final = X_train.copy()
                    X_test_final = X_test.copy()
                
                # Apply feature selection
                if fs_method != 'none':
                    X_train_fs, X_test_fs, fs_selector, selected_features = apply_feature_selection(
                        X_train_final, X_test_final, y_train, method=fs_method, k=15
                    )
                else:
                    X_train_fs, X_test_fs = X_train_final, X_test_final
                    fs_selector, selected_features = None, X_train.columns.tolist()
                
                # Hyperparameter tuning with GridSearchCV
                grid_search = GridSearchCV(
                    model_config['model'], 
                    model_config['params'],
                    cv=5, 
                    scoring='r2',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train_fs, y_train)
                
                best_model_grid = grid_search.best_estimator_
                cv_mean_r2 = grid_search.best_score_
                
                # Test performance
                y_pred = best_model_grid.predict(X_test_fs)
                y_pred = np.clip(y_pred, 0, 4)
                
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                config_name = f"{model_name}+{scaler_name}+{fs_method}"
                print(f"   {config_name}: CV_R2={cv_mean_r2:.4f}, Test_R2={r2:.4f}, MAE={mae:.4f}")
                
                results.append({
                    'config': config_name,
                    'cv_r2': cv_mean_r2,
                    'test_r2': r2,
                    'test_mae': mae,
                    'best_params': grid_search.best_params_
                })
                
                # Update best model
                if cv_mean_r2 > best_cv_score:
                    best_cv_score = cv_mean_r2
                    best_model = best_model_grid
                    best_scaler = scaler
                    best_feature_selector = fs_selector
                    best_selected_features = selected_features
                    best_config = {
                        'model_name': model_name,
                        'scaler_name': scaler_name,
                        'fs_method': fs_method,
                        'best_params': grid_search.best_params_,
                        'cv_r2': cv_mean_r2,
                        'test_r2': r2,
                        'test_mae': mae
                    }
                    
            except Exception as e:
                print(f"   ❌ Error with {model_name}+{scaler_name}+{fs_method}: {str(e)[:100]}")

# --- FINAL RESULTS ---
print("\n" + "="*70)
print("🏆 FINAL RESULTS")
print("="*70)

# Sort results by CV R2
results_df = pd.DataFrame(results).sort_values('cv_r2', ascending=False)
print("\nTop 10 Configurations:")
print(results_df.head(10).to_string(index=False))

print(f"\n🥇 BEST MODEL CONFIGURATION:")
print(f"   Model: {best_config['model_name']}")
print(f"   Scaler: {best_config['scaler_name']}")
print(f"   Feature Selection: {best_config['fs_method']}")
print(f"   Best Parameters: {best_config['best_params']}")
print(f"   CV R2 Score: {best_config['cv_r2']:.4f}")
print(f"   Test R2 Score: {best_config['test_r2']:.4f}")
print(f"   Test MAE: {best_config['test_mae']:.4f}")

# Improvement analysis
improvement = best_config['test_r2'] - baseline_r2
print(f"\n📈 IMPROVEMENT ANALYSIS:")
print(f"   Baseline R2: {baseline_r2:.4f}")
print(f"   Best Model R2: {best_config['test_r2']:.4f}")
print(f"   Improvement: {improvement:.4f} ({(improvement/abs(baseline_r2)*100) if baseline_r2 != 0 else 'N/A'})")

if best_config['test_r2'] > 0.1:
    print("   ✅ Model shows meaningful predictive power!")
elif best_config['test_r2'] > 0.05:
    print("   ⚠️  Model shows some predictive power, but limited.")
else:
    print("   ❌ Model predictive power is very limited.")

# --- SAVE BEST MODEL ---
model_dir = 'ml_models_enhanced'
os.makedirs(model_dir, exist_ok=True)

# Retrain with full dataset
if best_scaler is not None:
    X_all_scaled = best_scaler.fit_transform(X)
    X_all_final = pd.DataFrame(X_all_scaled, columns=X.columns)
else:
    X_all_final = X.copy()

if best_feature_selector is not None:
    X_all_final = best_feature_selector.fit_transform(X_all_final, y)

best_model.fit(X_all_final, y)

# Save files
MODEL_FILENAME = os.path.join(model_dir, 'best_ip_prediction_model.pkl')
SCALER_FILENAME = os.path.join(model_dir, 'best_ip_prediction_scaler.pkl')
SELECTOR_FILENAME = os.path.join(model_dir, 'best_ip_prediction_selector.pkl')
CONFIG_FILENAME = os.path.join(model_dir, 'best_ip_prediction_config.pkl')

with open(MODEL_FILENAME, 'wb') as f:
    pickle.dump(best_model, f)
    
if best_scaler is not None:
    with open(SCALER_FILENAME, 'wb') as f:
        pickle.dump(best_scaler, f)

if best_feature_selector is not None:
    with open(SELECTOR_FILENAME, 'wb') as f:
        pickle.dump(best_feature_selector, f)

with open(CONFIG_FILENAME, 'wb') as f:
    pickle.dump(best_config, f)

print(f"\n✅ Best model saved to {model_dir}/")

# --- FEATURE IMPORTANCE ANALYSIS ---
if hasattr(best_model, 'feature_importances_'):
    print(f"\n🧠 FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    if best_feature_selector is not None and hasattr(best_feature_selector, 'get_support'):
        feature_names = [best_selected_features[i] for i in range(len(best_selected_features))]
    else:
        feature_names = X.columns.tolist()
    
    if len(feature_names) == len(best_model.feature_importances_):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("Top 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))

print(f"\n🎉 Enhanced training completed!")
print(f"💡 Recommendations:")
print(f"   - If R2 is still low, consider collecting more diverse features")
print(f"   - Academic performance prediction is inherently challenging")
print(f"   - Focus on features with highest importance for practical insights")