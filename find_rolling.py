import argparse
import pandas as pd
import numpy as np
import os
import configparser
from pathlib import Path

class ConfigReader:
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

class TrackProcessor:
    def __init__(self, min_count=3):
        self.min_count = min_count

    def process(self, df):
        return df.groupby('track_id').apply(self.process_group).reset_index(drop=True)

    def process_group(self, group):
    
        group = group.sort_values(by='frame_y').reset_index(drop=True)
        group = self.compute_rolling(group)
        group = self.eliminate_short_sequences(group)
        group = self.add_preceding_point(group)
        group = self.label_sequences(group)
        return group

    def compute_rolling(self, group):
        group['rolling'] = (group['y'] - group['y'].shift(1)) < 0
        return group

    def eliminate_short_sequences(self, group):
        rolling_list = group['rolling'].tolist()
        count = 0

        # Track and eliminate short sequences
        for i in range(len(rolling_list)):
            if rolling_list[i]:
                count += 1
            else:
                # If the sequence length is less than `min_count`, reset them to `False`
                if count < self.min_count:
                    for j in range(1, count + 1):
                        rolling_list[i - j] = False
                count = 0
        
        # Handle the case where the sequence of `True`s is at the end
        if count < self.min_count:
            for j in range(1, count + 1):
                rolling_list[len(rolling_list) - j] = False
        
        group['rolling'] = rolling_list
        return group

    def add_preceding_point(self, group):
        group['rolling'] = (group['rolling'] | group['rolling'].shift(-1))
        return group

    def label_sequences(self, group):
        group['sequence_id'] = (group['rolling'] != group['rolling'].shift()).cumsum()
        return group

    def update(self, processed_df, statistics_df):
        # Select necessary columns from statistics_df
        statistics_filtered = statistics_df[['track_id', 'sequence_id', 'rolling_filtered']]
        
        # Merge processed_df with statistics_df based on track_id and sequence_id
        processed_df = processed_df.merge(statistics_filtered, on=['track_id', 'sequence_id'], how='left')
        
        return processed_df

class StatisticsCalculator:
    def __init__(self, pixel_size, time_between_frames):
        self.pixel_size = pixel_size
        self.time_between_frames = time_between_frames

    def calculate_statistics_group(self, group):
        sequence_stats = group.groupby('sequence_id').agg(
            rolling=('rolling', 'first'),
            track_id=('track_id', 'first'),
            avg_x=('x', 'mean'),
            avg_y=('y', 'mean'),
            var_x=('x', 'var'),
            var_y=('y', 'var'),
            x_0=('x', 'first'),
            x_1=('x', 'last'),
            y_0=('y', 'first'),
            y_1=('y', 'last'),
            frame_0=('frame_y', 'first'),
            frame_1=('frame_y', 'last')
        ).reset_index()

        sequence_stats['dx'] = (sequence_stats['x_1'] - sequence_stats['x_0']) * self.pixel_size
        sequence_stats['dy'] = (sequence_stats['y_1'] - sequence_stats['y_0']) * self.pixel_size
        sequence_stats['dt'] = (sequence_stats['frame_1'] - sequence_stats['frame_0']) * self.time_between_frames
        sequence_stats['dy/dt'] = sequence_stats['dy'] / sequence_stats['dt']

        return sequence_stats

    def calculate_statistics(self, df):
        return df.groupby('track_id').apply(self.calculate_statistics_group).reset_index(drop=True)

    def calculate_sigma(self, statistics_df):
        non_rolling_sequences = statistics_df[statistics_df['rolling'] == False]
        sigma_y = non_rolling_sequences['var_y'].mean()
        print(f"sigma_y: {sigma_y:.2f}")
        print(f"3 sigma_y: {3 * sigma_y:.2f}")
        
        return sigma_y

    def update_statistics_df(self, statistics_df, sigma_y):
        # Create the rolling_filtered column based on conditions
        statistics_df['rolling_filtered'] = False  # Initialize to False
        
        # Set rolling_filtered to True where rolling == True and abs(dy) > 3 * sigma_y
        statistics_df.loc[
            (statistics_df['rolling'] == True) & (abs(statistics_df['dy']) > 3 * sigma_y), 
            'rolling_filtered'
        ] = True
        
        return statistics_df

class SigmaDisplacementFilter:
    def __init__(self, sigma_y):
        self.sigma_y = sigma_y

    def update_statistics_df(self, statistics_df):
        # Create the rolling_filtered column based on conditions
        statistics_df['rolling_filtered'] = False  # Initialize to False
        
        # Set rolling_filtered to True where rolling == True and abs(dy) > 3 * sigma_y
        statistics_df.loc[
            (statistics_df['rolling'] == True) & (abs(statistics_df['dy']) > 3 * self.sigma_y), 
            'rolling_filtered'
        ] = True
        
        return statistics_df

    def update_processed_df(self, processed_df, statistics_df):
        # Select necessary columns from statistics_df
        statistics_filtered = statistics_df[['track_id', 'sequence_id', 'rolling_filtered']]
        
        # Merge processed_df with statistics_df based on track_id and sequence_id
        updated_processed_df = processed_df.merge(statistics_filtered, on=['track_id', 'sequence_id'], how='left')
        
        return updated_processed_df

    def update_both(self, processed_df, statistics_df):
        # Update statistics_df first
        statistics_df = self.update_statistics_df(statistics_df)
        
        # Then update processed_df based on the updated statistics_df
        processed_df = self.update_processed_df(processed_df, statistics_df)
        
        return processed_df, statistics_df

class RollingStatisticsAnalyzer:
    def calculate_platelet_rolling_statistics(self, statistics_df, data_handler, config):
        # Get rolling platelets
        rolling_platelets = self.get_rolling_platelets(statistics_df)
        
        # Calculate statistics
        rolling_data = self.calculate_rolling_data(statistics_df)
        stop_data = self.calculate_stop_data(statistics_df, rolling_platelets)

        # Select relevant columns for saving
        rolling_columns = ['track_id', 'rolling_velocity', 'rolling_time', 'travel_distance', 'x_0', 'x_1', 'y_0', 'y_1', 'frame_0', 'frame_1']
        stop_columns = ['track_id', 'stop_effective_velocity', 'stop_time', 'x_0', 'x_1', 'y_0', 'y_1', 'frame_0', 'frame_1']
        ratio_columns = ['track_id', 'rolling_ratio']

        # Save these statistics to CSV
        data_handler.save_csv(rolling_data[rolling_columns], config.get('rolling_csv'))
        data_handler.save_csv(stop_data[stop_columns], config.get('stop_csv'))


        # Calculate time ratios
        time_sums = self.calculate_time_ratios(rolling_data, stop_data)

        # Save time ratios to CSV
        data_handler.save_csv(time_sums[ratio_columns], config.get('ratios_csv'))

        # Summarize statistics
        summary_stats_df = self.summarize_statistics(rolling_data, stop_data, time_sums)

        data_handler.save_csv(summary_stats_df, config.get('summary_stats_csv'), index=True)
        print(f"Saved summary statistics to {config.get('summary_stats_csv')}")


    def get_rolling_platelets(self, statistics_df):
        return statistics_df.groupby('track_id')['rolling_filtered'].any().reset_index()

    def calculate_rolling_data(self, statistics_df):
        # Calculate rolling time for rolling_filtered == True
        rolling_data = statistics_df[statistics_df['rolling_filtered'] == True].copy()
        rolling_data['rolling_velocity'] = -rolling_data["dy/dt"]
        rolling_data['travel_distance'] = -rolling_data["dy"]
        rolling_data['rolling_time'] = rolling_data['dt']
        return rolling_data

    def calculate_stop_data(self, statistics_df, rolling_platelets):
        # Filter track_ids with rolling_filtered and calculate stop time for rolling_filtered == False
        rolling_track_ids = rolling_platelets['track_id']
        rolling_platelets_statistics = statistics_df[statistics_df['track_id'].isin(rolling_track_ids)]
        stop_stats = rolling_platelets_statistics[statistics_df['rolling_filtered'] == False].copy()
        stop_stats['stop_effective_velocity'] = -stop_stats["dy/dt"]
        stop_stats['stop_time'] = stop_stats['dt']
        return stop_stats

    def calculate_time_ratios(self, rolling_data, stop_data):
        # Step 1: Group rolling_time_stats by track_id and calculate the sum of rolling_time
        rolling_time_sum = rolling_data.groupby('track_id')['rolling_time'].sum().reset_index()
        rolling_time_sum.columns = ['track_id', 'total_rolling_time']

        # Step 2: Group stop_time_stats by track_id and calculate the sum of stop_time
        stop_time_sum = stop_data.groupby('track_id')['stop_time'].sum().reset_index()
        stop_time_sum.columns = ['track_id', 'total_stop_time']

        # Step 3: Join both results by track_id
        time_sums = rolling_time_sum.merge(stop_time_sum, on='track_id', how='outer')
        time_sums['total_rolling_time'] = time_sums['total_rolling_time'].fillna(0)
        time_sums['total_stop_time'] = time_sums['total_stop_time'].fillna(0)
        time_sums['rolling_ratio'] = np.where(
            (time_sums['total_rolling_time'] + time_sums['total_stop_time']) != 0,
            time_sums['total_rolling_time'] / (time_sums['total_rolling_time'] + time_sums['total_stop_time']),
            0
        )
        return time_sums

    def summarize_statistics(self, rolling_data, stop_data, time_sums):
        def calculate_standard_error(std_dev, count):
            return std_dev / np.sqrt(count) if count > 0 else np.nan
        summary_stats = {
            'rolling velocity': [
                rolling_data['rolling_velocity'].mean(),
                rolling_data['rolling_velocity'].std(),
                rolling_data['rolling_velocity'].count(),
                calculate_standard_error(rolling_data['rolling_velocity'].std(), rolling_data['rolling_velocity'].count())
            ],
            'rolling time': [
                rolling_data['rolling_time'].mean(),
                rolling_data['rolling_time'].std(),
                rolling_data['rolling_time'].count(),
                calculate_standard_error(rolling_data['rolling_time'].std(), rolling_data['rolling_time'].count())
            ],
            'travel_distance': [
                rolling_data['travel_distance'].mean(),
                rolling_data['travel_distance'].std(),
                rolling_data['travel_distance'].count(),
                calculate_standard_error(rolling_data['travel_distance'].std(), rolling_data['travel_distance'].count())
            ],
            'stop time': [
                stop_data['stop_time'].mean(),
                stop_data['stop_time'].std(),
                stop_data['stop_time'].count(),
                calculate_standard_error(stop_data['stop_time'].std(), stop_data['stop_time'].count())
            ],
            'rolling time ratio': [
                time_sums['rolling_ratio'].mean(),
                time_sums['rolling_ratio'].std(),
                time_sums.shape[0],
                calculate_standard_error(time_sums['rolling_ratio'].std(), time_sums.shape[0])
            ]
        }

        return pd.DataFrame.from_dict(summary_stats, orient='index', columns=['mean', 'std', 'count', 'standard_error'])


def save_additional_filtered_dfs(data_handler, processed_df, base_output_path):
    # 1. Save rolling == False
    not_rolling_df = processed_df[processed_df['rolling'] == False]
    not_rolling_path = base_output_path.replace(".csv", "_not_rolling.csv")
    data_handler.save_csv(not_rolling_df, not_rolling_path)
    print(f"Saved not rolling sub trajectories to {not_rolling_path}")

    # 2. Save rolling == True and rolling_filtered == False
    rolling_filtered_df = processed_df[(processed_df['rolling'] == True) & 
                                                  (processed_df['rolling_filtered'] == False)]
    rolling_filtered_path = base_output_path.replace(".csv", "_rolling_filtered.csv")
    data_handler.save_csv(rolling_filtered_df, rolling_filtered_path)
    print(f"Saved filtered sub trajectories to {rolling_filtered_path}")

    rolling_df = processed_df[processed_df['rolling_filtered'] == True]
    rolling_path = base_output_path.replace(".csv", "_rolling.csv")
    data_handler.save_csv(rolling_df, rolling_path)
    print(f"Saved rolling sub trajectories to {rolling_path}")

def create_output_directory(input_file):
    # Create the output directory based on the input file name (replacing extension with 'csv')
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = f"{base_name}_output"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return output_dir

def process_single_file(config, csv_path, output_dir):
    """Processes a single CSV file and saves the results.

    Args:
        config (ConfigReader): The configuration object.
        csv_path (str): Path to the input CSV file.
        output_dir (str): Path to the output directory.
    """
    # Define file paths inside the new directory
    csv_path_output = os.path.join(output_dir, 'processed_data.csv')
    csv_path_output_stats = os.path.join(output_dir, 'statistics.csv')
    rolling_csv = os.path.join(output_dir, 'rolling.csv')
    stop_csv = os.path.join(output_dir, 'stop.csv')
    ratios_csv = os.path.join(output_dir, 'ratios.csv')
    summary_stats_csv = os.path.join(output_dir, 'summary_stats.csv')

    data_handler = DataHandler()
    processor = TrackProcessor()
    statistics_calculator = StatisticsCalculator(
        config.get('pixel_size'), config.get('time_between_frames')
    )
    
    # Load and process data
    df = data_handler.load_data(csv_path, required_columns={'track_id', 'x', 'y', 'frame_y'})
    processed_df = processor.process(df)
    
    print("Processed tracks")

    # Calculate statistics
    statistics_df = statistics_calculator.calculate_statistics(processed_df)
    print("Calculated statistics")

    sigma_y = statistics_calculator.calculate_sigma(statistics_df)

    if config.get('sigma') > 0:
        sigma_y = config.get('sigma')
    
    print(f"Using sigma = {sigma_y}")
    
    # Use SigmaDisplacementFilter to update both DataFrames
    sigma_filter = SigmaDisplacementFilter(sigma_y)
    processed_df, statistics_df = sigma_filter.update_both(processed_df, statistics_df)
    print("Updated processed and statistics DataFrames using sigma")

    # Save processed data
    data_handler.save_csv(processed_df, csv_path_output)
    data_handler.save_csv(statistics_df, csv_path_output_stats)

    # Save additional DataFrames if save_filtered is True
    if config.get('save_filtered'):
        save_additional_filtered_dfs(data_handler, processed_df, csv_path_output)

    # Analyze rolling statistics and save to CSV
    rolling_analyzer = RollingStatisticsAnalyzer()
    rolling_analyzer.calculate_platelet_rolling_statistics(statistics_df, data_handler, {
        'rolling_csv': rolling_csv,
        'stop_csv': stop_csv,
        'ratios_csv': ratios_csv,
        'summary_stats_csv': summary_stats_csv
    })

    print(f"Saved all output files in directory: {output_dir}")

def main(config_path):
    config = ConfigReader(config_path)
    
    # Get the input file path
    folder_path = config.get('csv_folder_path')
    print(folder_path)
    for filename in os.listdir(folder_path):
        print(filename)
        if filename.endswith(".csv"):
            # Construct the full path to the CSV file
            csv_path = os.path.join(folder_path, filename)

            # Create the output directory for the current file
            output_dir = create_output_directory(csv_path)

            # Process the single file
            process_single_file(config, csv_path, output_dir)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--config", required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)

