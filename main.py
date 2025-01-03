import argparse
import pandas as pd
import os

from data_handling import ConfigReader, DataHandler
from track_segmentation import TrackSegmenter, SegmentStatisticsCalculator
from motion_analysis import MotionStatisticsAnalyzer, SigmaDisplacementFilter


class FileAnalyzer:
    """Analyzes platelet motion in a single CSV file.

    Loads track data from a CSV file, segments tracks into motion and stationary segments,
    calculates segment statistics, filters out spurious motion based on a sigma filter,
    and saves the processed data and motion statistics to separate CSV files.
    """
    def __init__(self, config, csv_path, output_dir):
        self.config = config
        self.data_handler = DataHandler()
        self.df = self._load_data(csv_path)
        self._define_output_paths(output_dir)
        self.track_segmenter = TrackSegmenter()
        self.segment_statistics_calculator = SegmentStatisticsCalculator(
            self.config.get('pixel_size'), self.config.get('time_between_frames')
        )
    
    def _load_data(self, csv_path):
        return self.data_handler.load_data(csv_path, required_columns={'track_id', 'x', 'y', 'frame_y'})

    def _define_output_paths(self, output_dir):
        print(f"Going to save all files to the directory: {output_dir}")  
        self.csv_path_final_counts = os.path.join(output_dir, 'final_counts.csv')
        self.csv_path_output_counts = os.path.join(output_dir, 'counts.csv')
        self.csv_path_output_trajectories_base = os.path.join(output_dir, 'trajectories.csv')
        self.csv_path_processed_data = os.path.join(output_dir, 'processed_data.csv')
        self.csv_path_statistics = os.path.join(output_dir, 'statistics.csv')
        self.csv_path_moving = os.path.join(output_dir, 'moving.csv')
        self.csv_path_stop = os.path.join(output_dir, 'stop.csv')
        self.csv_path_ratios = os.path.join(output_dir, 'ratios.csv')
        self.csv_path_summary_stats = os.path.join(output_dir, 'summary_stats.csv')

    def _filter_by_displacement(self):
        """Filters motion segments based on displacement using a sigma filter.

        Calculates the mean y variance for stationary segments and uses it (or the sigma value from the config, if provided) as a threshold.
        Marks segments with displacements below the threshold as not moving.
        """
        sigma_y = self.segment_statistics_calculator.calculate_sigma(self.per_segment_info_df)
        if self.config.get('sigma') > 0:
            print("Using sigma from the config")
            sigma_y = self.config.get('sigma')
        print(f"Using sigma = {sigma_y}")
        sigma_filter = SigmaDisplacementFilter(sigma_y)
        self.track_segments_df, self.per_segment_info_df = sigma_filter.filter_by_displacement(self.track_segments_df, self.per_segment_info_df)
        
        print("Updated processed and statistics DataFrames using sigma")

    def _save_additional_filtered_dfs(self):
        """Saves filtered DataFrames for motion analysis: non-moving, false positive motion, and moving segments.
        """
        not_moving_df = self.track_segments_df[self.track_segments_df['motion'] == False]
        not_moving_path = self.csv_path_output_trajectories_base.replace(".csv", "_not_moving.csv")
        self.data_handler.save_csv(not_moving_df, not_moving_path)
        print(f"Saved not moving sub trajectories to {not_moving_path}")
 

        moving_false_positive_df = self.track_segments_df[(self.track_segments_df['motion'] == True) & 
                                                    (self.track_segments_df['motion_final'] == False)]
        moving_false_positive_path = self.csv_path_output_trajectories_base.replace(".csv", "_moving_false_positive.csv")
        self.data_handler.save_csv(moving_false_positive_df, moving_false_positive_path)
        print(f"Saved 'false positive movement' sub trajectories to {moving_false_positive_path}")

        moving_df = self.track_segments_df[self.track_segments_df['motion_final'] == True]
        moving_path = self.csv_path_output_trajectories_base.replace(".csv", "_moving.csv")
        self.data_handler.save_csv(moving_df, moving_path)
        print(f"Saved moving sub trajectories to {moving_path}")

    def _save_segments_data(self):
        """Saves track segment data and segment statistics to CSV files.

        Optionally saves additional filtered DataFrames based on configuration.
        """
        self.data_handler.save_csv(self.track_segments_df, self.csv_path_processed_data)
        self.data_handler.save_csv(self.per_segment_info_df, self.csv_path_statistics)

        if self.config.get('save_filtered'):
            self._save_additional_filtered_dfs()


    def _analyze_movement_types(self):
        """Analyzes and saves statistics for different movement types (moving, bumped, adhered).
        """
        motion_analyzer = MotionStatisticsAnalyzer(self.csv_path_moving, self.csv_path_stop, self.csv_path_ratios, self.csv_path_summary_stats)
        time_sums = motion_analyzer.calculate_platelet_motion_statistics(self.per_segment_info_df, self.data_handler)
        
        moving_track_ids, bumped_track_ids, adhered_track_ids = motion_analyzer.identify_track_ids(time_sums)
        total_track_ids = moving_track_ids.union(bumped_track_ids, adhered_track_ids)

        final_counts = pd.DataFrame({
            "moving_count": [len(moving_track_ids)],
            "bumped_count": [len(bumped_track_ids)],
            "adhered_count": [len(adhered_track_ids)],
            "total_count": [len(total_track_ids)]
        })
        self.data_handler.save_csv(final_counts, self.csv_path_final_counts)

        counts_df = (
            self.df.groupby("frame_y")
            .apply(lambda group: pd.Series({
                "moving_count": group["track_id"].isin(moving_track_ids).sum(),
                "bumped_count": group["track_id"].isin(bumped_track_ids).sum(),
                "adhered_count": group["track_id"].isin(adhered_track_ids).sum(),
                "total_count": len(group)
            }))
            .reset_index()
        )
        self.data_handler.save_csv(counts_df, self.csv_path_output_counts)

    def analyze(self):
        """Performs the complete analysis of platelet motion in the input file.

        Segments tracks, calculates statistics, filters data, and saves the results.
        """       
        self.track_segments_df = self.track_segmenter.identify_motion_segments(self.df)
        print("Determined motion segments and stationary segments")

        self.per_segment_info_df = self.segment_statistics_calculator.calculate_statistics(self.track_segments_df)
        print("Calculated statistics for each segment of each track")


        self._filter_by_displacement()
        
        self._save_segments_data()
        self._analyze_movement_types()      


def create_output_directory(input_file):
    """Creates an output directory based on the input file name.

    The output directory name is derived from the input file's base name
    (without extension) with "_output" appended.

    Args:
        input_file (str): Path to the input CSV file.

    Returns:
        str: The path to the created output directory.
    """
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = f"{base_name}_output"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return output_dir



def main(config_path):
    """Processes all CSV files in the specified folder according to the configuration.
    Args:
        config_path (str): Path to the configuration file.
    """
    config = ConfigReader(config_path)
    folder_path = config.get('csv_folder_path')
    print("Processing files in folder:", folder_path)
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            print("Processing file:", filename)
            csv_path = os.path.join(folder_path, filename)
            output_dir = create_output_directory(csv_path)
            file_analyzer = FileAnalyzer(config, csv_path, output_dir)
            file_analyzer.analyze()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration file for script.")
    parser.add_argument("--config", required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)

