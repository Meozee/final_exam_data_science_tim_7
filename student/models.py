# student/models.py

from django.db import models
from department.models import Department # Impor dari lokasi yang benar

class Student(models.Model):
    stu_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    gender = models.CharField(max_length=10)
    dob = models.DateField()
    dept = models.ForeignKey(Department, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return self.name