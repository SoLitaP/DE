import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
def load_data(file_path):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)

# Clean data
def clean_data(df):
    """
    Clean and preprocess the dataset.
    
    Args:
        df (pd.DataFrame): Raw dataset.
        
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    # Convert dates to datetime
    df['Date_of_Admission'] = pd.to_datetime(df['Date_of_Admission'])
    df['Discharge_Date'] = pd.to_datetime(df['Discharge_Date'])
    
    # Swap dates if Discharge_Date is earlier than Date_of_Admission
    mask = df['Discharge_Date'] < df['Date_of_Admission']
    df.loc[mask, ['Date_of_Admission', 'Discharge_Date']] = df.loc[mask, ['Discharge_Date', 'Date_of_Admission']].values

    # Calculate days of stay
    df['Days_of_Stay'] = (df['Discharge_Date'] - df['Date_of_Admission']).dt.days
    
    # Remove rows with negative or missing days
    df = df[df['Days_of_Stay'] >= 0]
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['Gender', 'Blood_Type', 'Medical_Condition', 'Admission_Type']

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders

# Split data
def split_data(df, target, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    
    Args:
        df (pd.DataFrame): The cleaned dataset.
        target (str): Target column name.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed.
        
    Returns:
        tuple: Training and testing feature sets (X_train, X_test) and target sets (y_train, y_test).
    """
    features = ['Age', 'Gender', 'Blood_Type', 'Medical_Condition', 'Billing_Amount', 'Admission_Type']
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Train model
def train_model(X_train, y_train):
    """
    Train a Random Forest Regressor.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target values.
        
    Returns:
        RandomForestRegressor: Trained model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def export_model_with_encoders(model, label_encoders, file_path):
    """
    Export the trained model and label encoders to a .pkl file.
    
    Args:
        model: Trained machine learning model.
        label_encoders (dict): Dictionary of label encoders used in preprocessing.
        file_path (str): Path to save the .pkl file.
    """
    # Bundle the model and label encoders
    data_to_save = {
        'model': model,
        'label_encoders': label_encoders
    }
    with open(file_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"Model and encoders exported to {file_path}")



if __name__ == "__main__":
    # Load data
    data_path = 'diverse_data.csv'  # Replace with actual path to your data
    df = load_data(data_path)
    
    # Clean data
    df, label_encoders = clean_data(df)
    
    # Select features and target
    features = ['Age', 'Gender', 'Blood_Type', 'Medical_Condition', 'Billing_Amount', 'Admission_Type']
    target = 'Days_of_Stay'

    X = df[features]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = split_data(df, target='Days_of_Stay')
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred)}")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred)}")
    print(f"R^2 Score: {r2_score(y_test, y_pred)}")
    
    export_model_with_encoders(model, label_encoders, 'random_forest_model_with_encoders.pkl')


