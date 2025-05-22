# usecase_xxx/views.py
from django.http import HttpResponse

def placeholder(request):
    return HttpResponse("This is a placeholder for {{ nama anggota }}'s app.")
from django.shortcuts import render
from django.http import JsonResponse
from .model.model_kartika import AtRiskModel, CourseRecommender
import json

def at_risk_form(request):
    return render(request, 'usecase_kartika/at_risk_form.html')

def at_risk_predict(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            model = AtRiskModel()
            result = model.predict(
                attendance=data['attendance'],
                midterm=data['midterm'],
                project=data['project'],
                historical_grade=data['historical_grade']
            )
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=400)

def at_risk_result(request):
    return render(request, 'usecase_kartika/at_risk_result.html')

def course_recommendation_form(request):
    return render(request, 'usecase_kartika/course_recommendation.html')

def get_recommendations(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            recommender = CourseRecommender()
            result = recommender.recommend(
                student_id=data['student_id'],
                department=data.get('department'),
                max_courses=data.get('max_courses', 5)
            )
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=400)

def recommendation_result(request):
    return render(request, 'usecase_kartika/recommendation_result.html')