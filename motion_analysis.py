import numpy as np
import pandas as pd

CRIT_RR = 0.4
CRIT_TIME = 1

class SigmaDisplacementFilter:
    """Filters track segments based on displacement using a threshold.

    This class filters out segments with small displacements in the y-direction,
    likely representing noise or spurious movements. The filtering threshold is either
    a user-provided value (from the configuration) or, if not provided, calculated as a 
    multiple of the standard deviation (sigma) of the y-variance for stationary segments.
    """
    def __init__(self, sigma_y):
        self.sigma_y = sigma_y

    def _update_per_segment_info_df(self, per_segment_info_df):
        """Updates per-segment information with final motion classification.

        Adds a 'motion_final' column to the DataFrame. Segments are marked as 'motion_final' = True
        if they were initially marked as moving ('motion' = True) AND their absolute displacement in y
        exceeds 3 times the `self.sigma_y` threshold.

        Args:
            per_segment_info_df (pd.DataFrame): DataFrame with segment information.

        Returns:
            pd.DataFrame: The updated DataFrame with the 'motion_final' column.
        """
        # Apply final criteria for platelet on motion
        per_segment_info_df['motion_final'] = False
        per_segment_info_df.loc[
            (per_segment_info_df['motion'] == True) & (abs(per_segment_info_df['dy']) > 3 * self.sigma_y), 
            'motion_final'
        ] = True
        
        return per_segment_info_df

    def _update_track_segments_df(self, track_segments_df, per_segment_info_df):
        """Updates track segments DataFrame with the 'motion_final' classification.

        Merges the 'motion_final' column from the updated per_segment_info_df into the 
        track_segments_df, effectively applying the sigma-based filtering to the main track data.


        Args:
            track_segments_df (pd.DataFrame): DataFrame with track segments.
            per_segment_info_df (pd.DataFrame): DataFrame with updated motion information.

        Returns:
            pd.DataFrame: The updated track_segments_df with the 'motion_final' column.
        """
        # Select necessary columns from per_segment_info_df
        statistics_filtered = per_segment_info_df[['track_id', 'segment_id', 'motion_final']]
        
        # Merge track_segments_df with per_segment_info_df based on track_id and segment_id
        updated_track_segments_df = track_segments_df.merge(statistics_filtered, on=['track_id', 'segment_id'], how='left')
        
        return updated_track_segments_df

    def filter_by_displacement(self, track_segments_df, per_segment_info_df):
        """Filters segments based on y-displacement using the sigma threshold.

        Updates both the per-segment information and the main track segments DataFrame
        with the 'motion_final' classification after applying the sigma filter.

        Args:
            track_segments_df (pd.DataFrame): DataFrame with track segments.
            per_segment_info_df (pd.DataFrame): DataFrame with segment information.

        Returns:
            tuple: A tuple containing the updated `track_segments_df` and `per_segment_info_df`.
        """
        per_segment_info_df = self._update_per_segment_info_df(per_segment_info_df)
        # update track_segments_df based on the updated per_segment_info_df
        track_segments_df = self._update_track_segments_df(track_segments_df, per_segment_info_df)
        return track_segments_df, per_segment_info_df

class MotionStatisticsAnalyzer:
    """Analyzes and calculates statistics for platelet motion segments.

    This class calculates metrics for both in motion and stationary trajectory segments within
    platelet tracks, including velocity, duration, and travel distance.  It also
    calculates ratios of moving time to total time and summarizes overall motion statistics.
    """
    def __init__(self, csv_path_moving, csv_path_stop, csv_path_ratios, csv_path_summary_stats):
        self.csv_path_moving = csv_path_moving
        self.csv_path_stop = csv_path_stop
        self.csv_path_ratios = csv_path_ratios
        self.csv_path_summary_stats = csv_path_summary_stats
        
    def calculate_platelet_motion_statistics(self, per_segment_info_df, data_handler):
        """Calculates and saves motion statistics for platelet tracks.

        Identifies tracks with motion, calculates metrics for moving and stationary segments,
        computes time ratios, and summarizes overall statistics.

        Args:
            per_segment_info_df (pd.DataFrame): DataFrame with segment-level information.
            data_handler (DataHandler): Instance for saving data to CSV files.

        Returns:
            pd.DataFrame: DataFrame containing calculated time ratios for each track.
        """
        moving_platelets = self._get_tracks_with_motion_segments(per_segment_info_df)
        motion_segments_metrics = self._calculate_motion_segments_metrics(per_segment_info_df)
        stationary_segments_metrics = self._calculate_stationary_segments_metrics(per_segment_info_df, moving_platelets)

        motion_columns = ['track_id', 'motion_velocity', 'motion_time', 'travel_distance', 'x_0', 'x_1', 'y_0', 'y_1', 'frame_0', 'frame_1']
        stationary_columns = ['track_id', 'stop_effective_velocity', 'stop_time', 'x_0', 'x_1', 'y_0', 'y_1', 'frame_0', 'frame_1']
        ratio_columns = ['track_id', 'total_time', 'total_motion_time', 'total_stop_time', 'motion_ratio', 'avg_motion_velocity']
        data_handler.save_csv(motion_segments_metrics[motion_columns], self.csv_path_moving)
        data_handler.save_csv(stationary_segments_metrics[stationary_columns], self.csv_path_stop)

        time_sums = self._calculate_motion_time_per_track_statistics(motion_segments_metrics, stationary_segments_metrics)
        data_handler.save_csv(time_sums[ratio_columns], self.csv_path_ratios)

        summary_stats_df = self._summarize_motion_statistics(motion_segments_metrics, stationary_segments_metrics, time_sums)
        data_handler.save_csv(summary_stats_df, self.csv_path_summary_stats, index=True)
        print(f"Saved summary statistics to {self.csv_path_summary_stats}")
        return time_sums


    def _get_tracks_with_motion_segments(self, per_segment_info_df):
        """Identifies tracks containing at least one motion segment.

        Args:
            per_segment_info_df (pd.DataFrame): DataFrame with segment information.

        Returns:
            pd.DataFrame: DataFrame with 'track_id' and a boolean indicating presence of motion.
        """
        return per_segment_info_df.groupby('track_id')['motion_final'].any().reset_index()

    def _calculate_motion_segments_metrics(self, per_segment_info_df):
        """Calculates metrics for segments marked as moving.

         Calculates motion velocity, travel distance, and motion time for each moving segment.

        Args:
            per_segment_info_df (pd.DataFrame): DataFrame with segment information.

        Returns:
            pd.DataFrame: DataFrame with calculated motion metrics for moving segments.
        """
        motion_data = per_segment_info_df[per_segment_info_df['motion_final'] == True].copy()
        motion_data['motion_velocity'] = -motion_data["dy/dt"]
        motion_data['travel_distance'] = -motion_data["dy"]
        motion_data['motion_time'] = motion_data['dt']
        return motion_data

    def _calculate_stationary_segments_metrics(self, per_segment_info_df, moving_platelets):
        """Calculates metrics for stationary segments within tracks having motion.

        Args:
            per_segment_info_df (pd.DataFrame):  DataFrame with segment information.
            moving_platelets (pd.DataFrame): DataFrame indicating tracks with motion.

        Returns:
            pd.DataFrame: DataFrame with calculated metrics for stationary segments.
        """
        moving_track_ids = moving_platelets['track_id']
        moving_platelets_statistics = per_segment_info_df[per_segment_info_df['track_id'].isin(moving_track_ids)]
        stop_stats = moving_platelets_statistics[per_segment_info_df['motion_final'] == False].copy()
        stop_stats['stop_effective_velocity'] = -stop_stats["dy/dt"]
        stop_stats['stop_time'] = stop_stats['dt']
        return stop_stats

    def _calculate_motion_time_per_track_statistics(self, motion_data, stop_data):
        """Calculates total motion time, total stop time, total time, and motion ratio for each track.

        Args:
            motion_data (pd.DataFrame): DataFrame containing motion segment metrics.
            stop_data (pd.DataFrame): DataFrame containing stationary segment metrics.

        Returns:
            pd.DataFrame: DataFrame containing calculated time statistics per track.
        """
        motion_time_stats = motion_data.groupby('track_id').agg(
            total_motion_time=('motion_time', 'sum'),
            avg_motion_velocity=('motion_velocity', 'mean')  # Calculate average motion velocity
        ).reset_index()

        stop_time_sum = stop_data.groupby('track_id')['stop_time'].sum().reset_index()
        stop_time_sum.columns = ['track_id', 'total_stop_time']

        time_sums = motion_time_stats.merge(stop_time_sum, on='track_id', how='outer')
        time_sums['total_motion_time'] = time_sums['total_motion_time'].fillna(0)
        time_sums['total_stop_time'] = time_sums['total_stop_time'].fillna(0)
        time_sums['total_time'] = time_sums['total_motion_time'] + time_sums['total_stop_time']
        time_sums['motion_ratio'] = np.where(
            time_sums['total_time'] != 0,
            time_sums['total_motion_time'] / time_sums['total_time'],
            0
        )
        return time_sums
    
    def _summarize_motion_statistics(self, motion_data, stop_data, time_sums):
        """Calculates and summarizes overall motion statistics.

        Computes mean, standard deviation, count, and standard error for various motion metrics.

        Args:
            motion_data (pd.DataFrame): DataFrame containing motion segment metrics.
            stop_data (pd.DataFrame): DataFrame containing stationary segment metrics.
            time_sums (pd.DataFrame): DataFrame containing time statistics per track.


        Returns:
            pd.DataFrame: DataFrame summarizing overall motion statistics.
        """
        def calculate_standard_error(std_dev, count):
            return std_dev / np.sqrt(count) if count > 0 else np.nan
        summary_stats = {
            'motion velocity': [
                motion_data['motion_velocity'].mean(),
                motion_data['motion_velocity'].std(),
                motion_data['motion_velocity'].count(),
                calculate_standard_error(motion_data['motion_velocity'].std(), motion_data['motion_velocity'].count())
            ],
            'motion time': [
                motion_data['motion_time'].mean(),
                motion_data['motion_time'].std(),
                motion_data['motion_time'].count(),
                calculate_standard_error(motion_data['motion_time'].std(), motion_data['motion_time'].count())
            ],
            'travel_distance': [
                motion_data['travel_distance'].mean(),
                motion_data['travel_distance'].std(),
                motion_data['travel_distance'].count(),
                calculate_standard_error(motion_data['travel_distance'].std(), motion_data['travel_distance'].count())
            ],
            'stop time': [
                stop_data['stop_time'].mean(),
                stop_data['stop_time'].std(),
                stop_data['stop_time'].count(),
                calculate_standard_error(stop_data['stop_time'].std(), stop_data['stop_time'].count())
            ],
            'motion time ratio': [
                time_sums['motion_ratio'].mean(),
                time_sums['motion_ratio'].std(),
                time_sums.shape[0],
                calculate_standard_error(time_sums['motion_ratio'].std(), time_sums.shape[0])
            ]
        }

        return pd.DataFrame.from_dict(summary_stats, orient='index', columns=['mean', 'std', 'count', 'standard_error'])

    def identify_track_ids(self, time_sums):
        """Classifies tracks into moving, bumped, and adhered categories based on motion criteria.

        Args:
            time_sums (pd.DataFrame): DataFrame containing time statistics per track.

        Returns:
            tuple: Three sets containing track IDs for moving, bumped, and adhered tracks, respectively.
        """
        moving = time_sums[(time_sums['motion_ratio'] >= CRIT_RR)]
        moving_track_ids = set(moving['track_id'].unique().tolist())
        bumped = time_sums[(time_sums['total_time'] <= CRIT_TIME) & (time_sums['motion_ratio'] < CRIT_RR)]
        bumped_track_ids = set(bumped['track_id'].unique().tolist())
        adhered = time_sums[(time_sums['total_time'] > CRIT_TIME) & (time_sums['motion_ratio'] < CRIT_RR)]
        adhered_track_ids = set(adhered['track_id'].unique().tolist())
        
        return moving_track_ids, bumped_track_ids, adhered_track_ids