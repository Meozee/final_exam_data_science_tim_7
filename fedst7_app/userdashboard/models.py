from django.db import models


class Assessment(models.Model):
    assessment_id = models.AutoField(primary_key=True)
    enroll = models.ForeignKey('Enrollment', models.DO_NOTHING, blank=True, null=True)
    assessment_type = models.CharField(max_length=50)
    score = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'assessment'


class Attendance(models.Model):
    attendance_id = models.AutoField(primary_key=True)
    enroll = models.ForeignKey('Enrollment', models.DO_NOTHING, blank=True, null=True)
    attendance_percentage = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'attendance'


class AuthGroup(models.Model):
    name = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = 'auth_group'


class AuthGroupPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_group_permissions'
        unique_together = (('group', 'permission'),)


class AuthPermission(models.Model):
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING)
    codename = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'auth_permission'
        unique_together = (('content_type', 'codename'),)


class AuthUser(models.Model):
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.BooleanField()
    username = models.CharField(unique=True, max_length=150)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.CharField(max_length=254)
    is_staff = models.BooleanField()
    is_active = models.BooleanField()
    date_joined = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'auth_user'


class AuthUserGroups(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_groups'
        unique_together = (('user', 'group'),)


class AuthUserUserPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    permission = models.ForeignKey(AuthPermission, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_user_permissions'
        unique_together = (('user', 'permission'),)


class Course(models.Model):
    course_id = models.AutoField(primary_key=True)
    course_name = models.CharField(max_length=100)
    dept = models.ForeignKey('Department', models.DO_NOTHING, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'course'


class CourseDifficulty(models.Model):
    course = models.ForeignKey(Course, models.DO_NOTHING, blank=True, null=True)
    difficulty_level = models.CharField(max_length=50)

    class Meta:
        managed = False
        db_table = 'course_difficulty'


class CourseInstructor(models.Model):
    course_instructor_id = models.AutoField(primary_key=True)
    course = models.ForeignKey(Course, models.DO_NOTHING, blank=True, null=True)
    instructor = models.ForeignKey('Instructor', models.DO_NOTHING, blank=True, null=True)
    semester = models.ForeignKey('Semester', models.DO_NOTHING, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'course_instructor'


class CourseSemester(models.Model):
    course = models.ForeignKey(Course, models.DO_NOTHING, blank=True, null=True)
    semester = models.ForeignKey('Semester', models.DO_NOTHING, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'course_semester'


class Department(models.Model):
    dept_id = models.AutoField(primary_key=True)
    dept_name = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'department'


class DjangoAdminLog(models.Model):
    action_time = models.DateTimeField()
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.SmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'django_admin_log'


class DjangoContentType(models.Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'django_content_type'
        unique_together = (('app_label', 'model'),)


class DjangoMigrations(models.Model):
    id = models.BigAutoField(primary_key=True)
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'


class Enrollment(models.Model):
    enroll_id = models.AutoField(primary_key=True)
    stu = models.ForeignKey('Student', models.DO_NOTHING, blank=True, null=True)
    course = models.ForeignKey(Course, models.DO_NOTHING, blank=True, null=True)
    semester = models.ForeignKey('Semester', models.DO_NOTHING, blank=True, null=True)
    grade = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'enrollment'


class Instructor(models.Model):
    instructor_id = models.AutoField(primary_key=True)
    instructor_name = models.CharField(max_length=100)
    dept = models.ForeignKey(Department, models.DO_NOTHING, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'instructor'


class Semester(models.Model):
    semester_id = models.AutoField(primary_key=True)
    semester_name = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'semester'


class Student(models.Model):
    stu_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    gender = models.CharField(max_length=10)
    dob = models.DateField()
    dept = models.ForeignKey(Department, models.DO_NOTHING, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'student'
