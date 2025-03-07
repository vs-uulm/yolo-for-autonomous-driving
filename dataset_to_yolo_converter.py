# https://docs.voxel51.com/dataset_zoo/datasets.html#dataset-zoo-kitti
# https://docs.voxel51.com/integrations/ultralytics.html

import argparse
import math
from pathlib import Path

import fiftyone
import yaml
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.random as four

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Config:
    """Dataclass representing our configuration parameters"""
    database_uri: str
    dataset: str
    classes: List[str]
    train_label: str
    export_directory: Path
    source_directory: Optional[Path]
    train_split_ratio: float = 0.8
    validation_label: Optional[str] = None

def load_config_file_path() -> Path:
    """Load the file path to the configuration file from the cli"""
    parser = argparse.ArgumentParser(description="YOLOv5 Exporter")
    parser.add_argument(
        "config",
        type=str,
        help="Path to the YAML configuration file (default: config.yaml)"
    )
    args = parser.parse_args()
    return Path(args.config)

def load_config_file(path: str) -> Config:
    """
    Load configuration from a YAML file and return a strongly-typed Config object.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)

def load_dataset(dataset_name: str, source_directory: Path, input_label: str) -> fo.Dataset:
    """Loads a dataset from the FiftyOne dataset zoo.

    Args:
        dataset_name (str): The name of the dataset to load.
        source_directory (Path): The directory where the dataset is located.
        input_label (str): The specific split (e.g., "train", "validation", "test") to load.

    Returns:
        fo.Dataset: The loaded FiftyOne dataset.
    """
    fiftyone_dataset = foz.load_zoo_dataset(
        dataset_name,
        split=input_label,
        source_dir=source_directory,
    )
    return fiftyone_dataset

def split_balanced(training_split_ratio: float, dataset: fo.Dataset):
    """Splits a FiftyOne dataset into balanced training and validation sets.

    Args:
        training_split_ratio (float): The proportion of the dataset to be used for training.
        dataset (fo.Dataset): The FiftyOne dataset to be split.

    Returns:
        tuple: A tuple containing:
            - fo.DatasetView: The training dataset view.
            - fo.DatasetView: The validation dataset view.
    """
    print("splitting dataset into training and validation sets with ratio ", training_split_ratio)
    k = math.ceil(training_split_ratio * len(dataset))
    training_view = four.balanced_sample(dataset, k, "ground_truth.label")
    train_ids = training_view.values("id")
    validation_view = dataset.exclude(train_ids)

    return training_view, validation_view


def export_dataset_to_yolo(dataset: fo.Dataset, export_directory: Path, output_label: str, classes: List[str]):
    """
    Export a FiftyOne Zoo dataset to the YOLOv5 format.
    """
    print(f"Exporting {output_label} dataset to {export_directory}")
    dataset.export(
        export_dir=export_directory,
        data_path=f"{output_label}/images",
        labels_path=f"{output_label}/labels",
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        split=output_label,
        classes=classes,
    )

def main():

    config_path = load_config_file_path()
    config = load_config_file(config_path)
    fo.config.database_uri = config.database_uri
    already_splitted = config.train_label != config.validation_label and config.validation_label is not None
    if already_splitted:
        train_dataset = load_dataset(config.dataset, config.source_directory, config.train_label)
        validation_dataset = load_dataset(config.dataset, config.source_directory, config.validation_label)
        export_dataset_to_yolo(
            dataset=train_dataset,
            dataset_name=config.dataset,
            output_label="train",
            classes=config.classes
        )
        export_dataset_to_yolo(
            dataset=validation_dataset,
            dataset_name=config.dataset,
            output_label="val",
            classes=config.classes
        )
    if not already_splitted:
        dataset = load_dataset(config.dataset, config.source_directory, config.train_label)
        training_view, validation_view = split_balanced(config.train_split_ratio, dataset)
        export_dataset_to_yolo(
            dataset=training_view,
            export_directory=config.export_directory,
            output_label="train",
            classes=config.classes
        )
        export_dataset_to_yolo(
            dataset=validation_view,
            export_directory=config.export_directory,
            output_label="val",
            classes=config.classes
        )

if __name__ == "__main__":
    main()
