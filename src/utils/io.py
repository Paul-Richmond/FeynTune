import json
import logging
import os
from typing import Any, Dict, Optional

from datasets import DatasetDict, load_dataset
from omegaconf import DictConfig


def save_dict_to_json(data: Dict[Any, Any], directory: str, filename: str) -> None:
    """
    Save a dictionary to a JSON file in the specified directory.

    This function creates the target directory if it doesn't exist,
    ensures the filename has a .json extension, and saves the dictionary
    as a formatted JSON file.

    Args:
        data: Dictionary to be saved to JSON.
        directory: Target directory path where the file will be saved.
        filename: Name of the file (with or without .json extension).

    Returns:
        None

    Example:
        >>> data = {"name": "John", "age": 30}
        >>> save_dict_to_json(data, "output", "user_data")
    """
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Check filename has extension and add if not
    if not filename.endswith('.json'):
        filename += '.json'

    # Construct the full file path
    file_path = os.path.join(directory, filename)

    # Write the dictionary to a JSON file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

    logging.info(f"Dictionary saved to {file_path}.")


def load_json_file(filename: str) -> Optional[Dict[Any, Any]]:
    """
    Load and parse a JSON file into a Python dictionary.

    This function attempts to read and parse a JSON file, handling various
    potential errors that might occur during the process.

    Args:
        filename: Path to the JSON file to be loaded.

    Returns:
        Dict[Any, Any]: The parsed JSON data as a dictionary if successful.
        None: If any error occurs during file reading or JSON parsing.

    Raises:
        No exceptions are raised; all errors are logged and None is returned.

    Example:
        >>> data = load_json_file("user_data.json")
        >>> if data is not None:
        ...     print(data["name"])
    """
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        if data is not None:
            logging.info("JSON data loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error(f'File {filename} not found.')
    except json.JSONDecodeError:
        logging.error(f"Error: '{filename}' is not a valid JSON file.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
    return None


def batch_dataset(dataset, batch_size):
    """
    Split a Huggingface dataset into batches.

    Args:
    dataset (Dataset): The Huggingface dataset to split.
    batch_size (int): The size of each batch.

    Returns:
    List[Dataset]: A list of Dataset objects, each representing a batch.
    """
    total_samples = len(dataset)
    batches = []

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch = dataset.select(range(start_idx, end_idx))
        batches.append(batch)

    return batches


def load_dataset_splits(datasets_cfg: DictConfig) -> DatasetDict:
    """
    Load dataset splits based on the provided configuration.

    Args:
        datasets_cfg (DictConfig): Configuration for datasets.

    Returns:
        DatasetDict: A dictionary containing the loaded dataset splits.
    """
    datasets_: Dict[str, Any] = {}
    for split, split_value in datasets_cfg.splits.items():
        datasets_[split] = load_dataset(datasets_cfg.name, split=split_value)
    return DatasetDict(datasets_)
