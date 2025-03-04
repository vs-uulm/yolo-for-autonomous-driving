import argparse
from dataclasses import dataclass, field

import yaml
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict

@dataclass
class Config:
    directory: str
    model_names: List[str] = field(default_factory=list)
    pretrained: bool = False
    trained_from_scratch: bool = False
    plots: bool = True
    validate: bool = True

    def __post_init__(self):
        if not self.directory:
            raise ValueError("No directory specified.")
        if len(self.model_names) == 0:
            raise ValueError("No models specified.")
        if not self.pretrained and not self.trained_from_scratch:
            raise ValueError("No training mode specified.")


def parse_config_file_path() -> Path:
    """Parse command-line arguments to get the config file path.

    Returns:
        Path: Path to the config file path.
    """
    parser = argparse.ArgumentParser(description="Train YOLO models using a YAML configuration file.")
    parser.add_argument(
        "config",
        type=str,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    return Path(args.config)

def load_config_file(config_path: Path) -> Config:
    """Load configuration values from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict: Parsed configuration data.
    """
    with open(config_path, "r") as file:
        data = yaml.safe_load(file)
    return Config(**data)

def get_dataset_yaml_file(dataset: str) -> str:
    """Generate the path to the dataset YAML file.

    Args:
        dataset (str): Name of the dataset.

    Returns:
        str: Path to the dataset YAML file.
    """
    return f'./{dataset}/dataset.yaml'


def get_model_suffixes(pretrained: bool, trained_from_scratch: bool) -> List[str]:
    """Determine the suffixes for the model files based on training type.

    Args:
        pretrained (bool): Whether to use a pretrained model.
        trained_from_scratch (bool): Whether to train from scratch.

    Returns:
        List[str]: List of suffixes for the model files.
    """
    suffixes = []
    if trained_from_scratch:
        suffixes.append("yaml")
    if pretrained:
        suffixes.append("pt")
    return suffixes


def train_model(model_name: str, yaml_file: str, suffix: str, dataset: str, validate:bool) -> None:
    """Train a YOLO model with the given configuration and save the model.

    Args:
        model_name (str): Name of the YOLO model.
        yaml_file (str): Path to the dataset YAML file.
        suffix (str): Model file suffix (e.g., 'pt' or 'yaml').
        dataset (str): Dataset name for saving the model.
        validate (bool): Whether to validate the model.
    """
    try:
        model_path = Path(f"{model_name}.{suffix}")
        model = YOLO(model_path)
        run_name = f"{model_name}_{dataset}"
        if suffix == "pt":
            run_name += "_finetuned"
        model.train(data=yaml_file, name=run_name, epochs=600, device=[0, 1], patience=10, val=validate, plots=True)

        save_filename = run_name + ".pt"
        model.save(save_filename)
    except Exception as e:
        print(f"Failed to train model: model={model_name}, dataset={dataset}, error={e}")

def main():
    """Main function to load configuration and train models."""
    config_file_path = parse_config_file_path()
    config = load_config_file(config_file_path)
    yaml_file = get_dataset_yaml_file(config.directory)
    suffixes = get_model_suffixes(config.pretrained, config.trained_from_scratch)
    for model_name in config.model_names:
        for suffix in suffixes:
            train_model(model_name, yaml_file, suffix, config.directory, config.validate)


if __name__ == "__main__":
    main()
