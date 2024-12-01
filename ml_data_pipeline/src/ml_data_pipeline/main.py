import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from .config import load_config
from .data_loader import get_data_loader
from loguru import logger

logger.add("debug.log", format="{time} {level} {message}", level="DEBUG")


def main(config_path):
    # Load configuration using the given config path
    config = load_config(config_path)

    # Get the appropriate data loader based on the file type
    loader = get_data_loader(config.data_loader.file_type)
    data = loader.load_data(config.data_loader.file_path)
    logger.info("Data was loaded")
    print("Loaded Data:")
    print(data.head())

    # Prepare data for training
    X = data.drop("HeartDisease", axis=1)
    y = data["HeartDisease"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.model.random_state
    )

    # Initialize and train model
    model = LogisticRegression(random_state=config.model.random_state, max_iter=3000)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Model accuracy: {score:.2f}")
    logger.info(f"Model accuracy: {score:.2f}")

if __name__ == "__main__":
    # Setup argparse to read the path to the configuration file
    parser = argparse.ArgumentParser(
        description="Run the ML pipeline with specified configuration"
    )
    parser.add_argument("config_path", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    main(args.config_path)
