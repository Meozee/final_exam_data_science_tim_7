from django.db import connection
import pandas as pd

def get_student_performance_data():
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
    
    # Encode difficulty_level
    df['difficulty_level'] = df['difficulty_level'].map({'Easy': 0, 'Medium': 1, 'Hard': 2})
    
    return df