import pandas as pd
import os
import configparser

class ConfigReader:
    """Reads and provides access to configuration parameters.

    Parses a configuration file and stores the parameters as a dictionary.
    Provides a `get` method to access individual parameters.
    """
    def __init__(self, config_path):
        self.config = self.read_config(config_path)

    def read_config(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        return {
            'csv_folder_path': config['DEFAULT']['csv_folder_path'],
            'pixel_size': float(config['DEFAULT']['pixel_size']),
            'time_between_frames': float(config['DEFAULT']['time_between_frames']),
            'save_filtered': config['DEFAULT'].getboolean('save_filtered'),
            'sigma': float(config['DEFAULT'].get('sigma', 0.0))
        }

    def get(self, key):
        return self.config.get(key)

class DataHandler:
    """Handles data loading and saving operations.

    Provides static methods for loading data from CSV files and saving DataFrames to CSV.
    """
    @staticmethod
    def load_data(csv_path, required_columns):
        df = pd.read_csv(csv_path)
        if not required_columns.issubset(df.columns):
            raise ValueError(f"The input {csv_path} must contain the following columns: {required_columns}")
        return df

    @staticmethod
    def save_csv(df, output_path, index=False):
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        df.to_csv(output_path, index=index)