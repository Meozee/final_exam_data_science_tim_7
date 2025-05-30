from django.test import TestCase
from django.contrib.auth.models import User
from .models import MLModel

class MLModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        
    def test_model_creation(self):
        model = MLModel.objects.create(
            name="Test Model",
            creator=self.user,
            model_type="classification",
            accuracy=0.95,
            description="Test description",
            use_case="Test use case"
        )
        self.assertEqual(str(model), "Test Model (classification)")