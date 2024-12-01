from pydantic import BaseModel
from omegaconf import OmegaConf


class DataLoaderConfig(BaseModel):
    file_path: str
    file_type: str


class ModelConfig(BaseModel):
    type: str
    random_state: int


class Config(BaseModel):
    data_loader: DataLoaderConfig
    model: ModelConfig


def load_config(config_path: str) -> Config:
    raw_config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(raw_config, resolve=True)

    return Config(**config_dict)
