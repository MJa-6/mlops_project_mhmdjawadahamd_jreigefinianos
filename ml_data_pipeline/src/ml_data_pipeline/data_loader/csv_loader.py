import pandas as pd
from .base_loader import DataLoader


class CSVLoader(DataLoader):
    def load_data(self, file_path: str) -> pd.DataFrame:
        data = pd.read_csv(file_path)

        # One-hot encode categorical variables
        categorical_cols = [
            "Sex",
            "ChestPainType",
            "RestingECG",
            "ExerciseAngina",
            "ST_Slope",
        ]
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

        return data
