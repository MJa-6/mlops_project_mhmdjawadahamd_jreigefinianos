import pandas as pd

def load_data(filepath: str):
    """Load dataset from a specified file path."""
    data = pd.read_csv(filepath)

    # One-hot encode categorical variables
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    return data