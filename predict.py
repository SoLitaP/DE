import pickle
import pandas as pd
from model import clean_data

def load_model_with_encoders(file_path):
    """
    Load the trained model and label encoders from a .pkl file.
    
    Args:
        file_path (str): Path to the .pkl file.
    
    Returns:
        model: Trained machine learning model.
        label_encoders (dict): Dictionary of label encoders used in preprocessing.
    """
    with open(file_path, 'rb') as f:
        data_loaded = pickle.load(f)
    print(f"Model and encoders loaded from {file_path}")
    return data_loaded['model'], data_loaded['label_encoders']


def predict(raw_data, model, label_encoders):
    """
    Make a prediction on new data.
    
    Args:
        raw_data (dict): New data to make a prediction on.
        model: Trained machine learning model.
        label_encoders (dict): Dictionary of label encoders used in preprocessing.
        
    Returns:
        float: Predicted target value.
    """
    # Load the raw data into a DataFrame
    new_data = pd.DataFrame(raw_data)
    
    categorical_columns = ['Gender', 'Blood_Type', 'Medical_Condition', 'Admission_Type']
    for col in categorical_columns:
        le = label_encoders[col]
        new_data[col] = le.fit_transform(new_data[col])

    features = ['Age', 'Gender', 'Blood_Type', 'Medical_Condition', 'Billing_Amount', 'Admission_Type']

    # Make a prediction
    prediction = model.predict(new_data[features])
    return prediction

if __name__ == "__main__":
    # Load the model and encoders
    loaded_model, loaded_encoders = load_model_with_encoders('random_forest_model_with_encoders.pkl')

    print("Loaded model:", loaded_model)
    print("Loaded encoders:", loaded_encoders)

    # Load new dataset for prediction
    new_data = pd.read_csv('healthcare_dataset.csv')  # Replace with the path to your raw dataset file

    # Predict hospital stays
    predictions = predict(new_data, loaded_model, loaded_encoders)
    print(predictions)
    # # Add predictions to the dataset
    new_data['Predicted_Days_of_Stay'] = predictions

    # Export to CSV
    new_data.to_csv('predicted_hospital_stays.csv', index=False)

    print("Predictions saved to 'predicted_hospital_stays.csv'")

