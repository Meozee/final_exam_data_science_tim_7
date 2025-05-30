# Data Science Project - Team 7  
A Django web app for educational data analysis and ML model predictions.  

## 🚀 How to Run  

### 📋 Prerequisites  
- Python 3.8+  
- PostgreSQL (for production)  
- pip  

### 🛠️ Installation & Setup  

```bash  
# 1️⃣ Clone the Repository  
git clone https://github.com/Meozee/final_exam_data_science_tim_7.git
cd final_exam_data_science_tim_7  
  
# 2️⃣ Create Virtual Environment & Activate  
python -m venv django_venv  
# On Windows:  
django_venv\Scripts\activate  
# On macOS/Linux:  
source django_venv/bin/activate  
  
# 3️⃣ Install Dependencies  
pip install -r requirements.txt  
# If no requirements.txt, install individually:  
pip install django psycopg2-binary pandas scikit-learn numpy lightgbm xgboost joblib  
  
# 4️⃣ Setup Database  
# --- Option 1: PostgreSQL (Production) ---  
# Install PostgreSQL and create database 'dbexam' with user:  
# Username: postgres  
# Password: your_password  
# Host: localhost  
# Port: 5432  
  
# Import SQL backup into PostgreSQL:  
psql -U postgres -d dbexam -f examcase01.sql  
  
# Update settings.py if needed:  
# DATABASES = {  
#     'default': {  
#         'ENGINE': 'django.db.backends.postgresql',  
#         'NAME': 'dbexam',  
#         'USER': 'postgres',  
#         'PASSWORD': 'your_password',  
#         'HOST': 'localhost',  
#         'PORT': '5432',  
#     }  
# }  
  
# --- Option 2: SQLite (Development) ---  
# Use included db.sqlite3 (no setup needed)  
  
# 5️⃣ Apply Migrations  
python manage.py makemigrations  
python manage.py migrate  
  
# 6️⃣ Create Superuser  
python manage.py createsuperuser  
  
# 7️⃣ Collect Static Files  
python manage.py collectstatic  
  
# 8️⃣ Run the Server  
python manage.py runserver  
