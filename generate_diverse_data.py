import pandas as pd
import random
from faker import Faker

# Initialize Faker
fake = Faker()

# Set the number of rows
num_rows = 54966

# Define weighted distributions
medical_conditions = {
    "Hypertension": 0.25,
    "Diabetes": 0.20,
    "Asthma": 0.15,
    "Heart Disease": 0.10,
    "Cancer": 0.05,
    "Injury": 0.10,
    "Infection": 0.10,
    "None": 0.05
}

admission_types = {
    "Emergency": 0.3,
    "Routine": 0.4,
    "Specialized": 0.2,
    "Critical Care": 0.1
}

test_results = {
    "Positive": 0.4,
    "Negative": 0.5,
    "Inconclusive": 0.1
}

medications = {
    "Paracetamol": 0.2,
    "Ibuprofen": 0.15,
    "Amoxicillin": 0.1,
    "Metformin": 0.1,
    "None": 0.45
}


# Function to generate age-specific disease
def get_medical_condition(age):
    if age < 18:
        return random.choices(["Asthma", "Injury", "Infection"], weights=[0.5, 0.3, 0.2])[0]
    elif 18 <= age <= 40:
        return random.choices(["Injury", "Infection", "Hypertension"], weights=[0.4, 0.3, 0.3])[0]
    elif 41 <= age <= 65:
        return random.choices(["Diabetes", "Hypertension", "Heart Disease"], weights=[0.3, 0.5, 0.2])[0]
    else:
        return random.choices(["Heart Disease", "Cancer", "Hypertension"], weights=[0.4, 0.3, 0.3])[0]


# Generate fake data
data = {
    "Name": [fake.name() for _ in range(num_rows)],
    "Age": [random.randint(0, 100) for _ in range(num_rows)],
    "Gender": [random.choices(["Male", "Female", "Other"], weights=[0.48, 0.48, 0.04])[0] for _ in range(num_rows)],
    "Blood_Type": [random.choice(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]) for _ in range(num_rows)],
    "Medical_Condition": [],
    "Date_of_Admission": [fake.date_between(start_date='-1y', end_date='today').isoformat() for _ in range(num_rows)],
    "Doctor": [fake.name() for _ in range(num_rows)],
    "Hospital": [fake.company() for _ in range(num_rows)],
    "Insurance_Provider": [fake.company() for _ in range(num_rows)],
    "Billing_Amount": [],
    "Room_Number": [random.randint(1, 1000) for _ in range(num_rows)],
    "Admission_Type": [random.choices(list(admission_types.keys()), weights=admission_types.values())[0] for _ in
                       range(num_rows)],
    "Discharge_Date": [fake.date_between(start_date='-1y', end_date='today').isoformat() for _ in range(num_rows)],
    "Medication": [random.choices(list(medications.keys()), weights=medications.values())[0] for _ in range(num_rows)],
    "Test_Results": [random.choices(list(test_results.keys()), weights=test_results.values())[0] for _ in
                     range(num_rows)]
}

# Populate condition and billing based on age and condition
for age in data["Age"]:
    condition = get_medical_condition(age)
    data["Medical_Condition"].append(condition)

    # Billing amount varies with condition
    if condition == "None":
        billing = random.uniform(50, 200)
    elif condition in ["Hypertension", "Diabetes"]:
        billing = random.uniform(200, 1000)
    elif condition in ["Heart Disease", "Cancer"]:
        billing = random.uniform(5000, 20000)
    else:
        billing = random.uniform(100, 5000)

    data["Billing_Amount"].append(round(billing, 2))

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV for PostgreSQL upload
df.to_csv('/tmp/medical_records.csv', index=False)

print("Synthetic healthcare dataset generated successfully!")
