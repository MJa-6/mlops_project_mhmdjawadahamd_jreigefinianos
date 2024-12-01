from ml_data_pipeline.data_loader import get_data_loader

def test_csv_loader():
    loader = get_data_loader("csv")
    data = loader.load_data("data/heart.csv")
    assert not data.empty, "Data should not be empty"