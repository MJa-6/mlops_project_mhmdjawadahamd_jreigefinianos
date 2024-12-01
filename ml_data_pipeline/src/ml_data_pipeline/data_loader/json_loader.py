import pandas as pd
from .base_loader import DataLoader


class JSONLoader(DataLoader):
    def load_data(self, file_path: str) -> pd.DataFrame:
        return pd.read_json(file_path)
