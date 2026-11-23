import pytest

@pytest.fixture
def normal():
    return {"Pregnancies": 1, "Glucose": 95, "BloodPressure": 70, "SkinThickness": 20,
            "Insulin": 80, "BMI": 23, "DiabetesPedigreeFunction": 0.3, "Age": 28}
