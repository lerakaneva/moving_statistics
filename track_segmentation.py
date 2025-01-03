class TrackSegmenter:
    """Segments platelet tracks into motion and stationary segments.

    This class analyzes platelet tracks and identifies segments of motion and non-motion
    based on changes in the y-coordinate over time. It filters out short motion segments
    to remove noise and refines segment boundaries for accurate motion analysis.
    """
    def __init__(self, min_count=3):
        """Initializes TrackSegmenter with a minimum motion segment length.

        Args:
            min_count (int): The minimum number of consecutive frames a platelet must 
                             be moving to be considered a true motion segment. Defaults to 3.
        """
        self.min_count = min_count

    def identify_motion_segments(self, df):
        """Identifies motion segments within platelet tracks.

        Processes each track independently, sorting by frame and then applying motion
        detection, filtering, and segmentation logic.

        Args:
            df (pd.DataFrame): DataFrame containing track data with 'track_id', 'x', 'y', and 'frame_y' columns.

        Returns:
            pd.DataFrame: The input DataFrame with added 'motion' and 'segment_id' columns.
        """
        return df.groupby('track_id').apply(self._process_track).reset_index(drop=True)

    def _process_track(self, group):
        """Processes a single track to identify and segment motion.

        Sorts the track by frame, identifies moments of motion, eliminates short motion segments
        (shorter than `self.min_count` frames), includes the frame preceding a motion segment
        as part of the motion, and finally numerates the identified segments.

        Args:
            group (pd.DataFrame): DataFrame containing data for a single track.

        Returns:
            pd.DataFrame: The processed DataFrame with added 'motion' and 'segment_id' columns.
        """
        group = group.sort_values(by='frame_y').reset_index(drop=True)
        group = self._compute_in_motion(group)
        group = self._eliminate_short_motion_segments(group)
        group = self._add_preceding_point(group)
        group = self._numerate_segments(group)
        return group

    def _compute_in_motion(self, group):
        """Determines if a platelet is in motion at each frame.

        Compares the current frame's y-coordinate with the previous frame's.  If the current
        y-coordinate is less than the previous (indicating motion in the direction of the flow), the platelet
        is marked as 'motion' (True) for that frame.

        Args:
            group (pd.DataFrame): DataFrame for a single track, sorted by 'frame_y'.

        Returns:
            pd.DataFrame: The DataFrame with a new 'motion' column (boolean).
        """
        group['motion'] = (group['y'] - group['y'].shift(1)) < 0
        return group

    def _eliminate_short_motion_segments(self, group):
        """Eliminates short motion segments based on min_count.

        Filters out motion segments shorter than `self.min_count` frames by marking them as
        stationary (False) in the 'motion' column.  This removes noise or spurious short movements.

        Args:
            group (pd.DataFrame): DataFrame with 'motion' column.

        Returns:
            pd.DataFrame: DataFrame with updated 'motion' column.
        """
        positions_in_motion_list = group['motion'].tolist()
        count = 0
        for i in range(len(positions_in_motion_list)):
            if positions_in_motion_list[i]:
                count += 1
            else:
                if count < self.min_count:
                    for j in range(1, count + 1):
                        positions_in_motion_list[i - j] = False
                count = 0
        if count < self.min_count:
            for j in range(1, count + 1):
                positions_in_motion_list[len(positions_in_motion_list) - j] = False
        group['motion'] = positions_in_motion_list
        return group

    def _add_preceding_point(self, group):
        """Includes the frame preceding a motion segment as moving.

        Modifies the 'motion' column to include the frame immediately before a detected motion segment
        as part of that motion segment. This ensures that the beginning of motion is captured correctly.

        Args:
            group (pd.DataFrame): DataFrame with the 'motion' column.

        Returns:
             pd.DataFrame: DataFrame with the updated 'motion' column.
        """
        group['motion'] = (group['motion'] | group['motion'].shift(-1))
        return group

    def _numerate_segments(self, group):
        """Assigns unique segment IDs.

        Adds a 'segment_id' column to the DataFrame. Each continuous sequence of motion or non-motion
        is assigned a unique integer ID.

        Args:
            group (pd.DataFrame): DataFrame with the 'motion' column.

        Returns:
            pd.DataFrame: DataFrame with the added 'segment_id' column.
        """
        group['segment_id'] = (group['motion'] != group['motion'].shift()).cumsum()
        return group

    def update(self, track_segments_df, per_segment_info_df):
        """Updates track segments DataFrame with motion analysis results.

        Combines `track_segments_df` with analysis data from  `per_segment_info_df`,
        specifically adding the 'motion_final' column.

        Args:
            track_segments_df (pd.DataFrame): DataFrame with track segment information.
            per_segment_info_df (pd.DataFrame): DataFrame containing motion analysis results.

        Returns:
            pd.DataFrame: The updated `track_segments_df`.
        """

        statistics_filtered = per_segment_info_df[['track_id', 'segment_id', 'motion_final']]
        track_segments_df = track_segments_df.merge(statistics_filtered, on=['track_id', 'segment_id'], how='left')
        return track_segments_df
    
class SegmentStatisticsCalculator:
    """Calculates statistics for track segments.

    This class computes various statistics for each segment within platelet tracks,
    including displacement, duration, and velocity. It uses pixel size and time between
    frames to convert measurements to physical units.
    """
    def __init__(self, pixel_size, time_between_frames):
        self.pixel_size = pixel_size
        self.time_between_frames = time_between_frames

    def _calculate_statistics_track(self, group):
        """Calculates statistics for segments within a single track.

        Groups the track data by 'segment_id' and computes statistics like initial and final
        coordinates, displacement, duration, and velocity for each segment.

        Args:
            group (pd.DataFrame): DataFrame containing track data for a single track.

        Returns:
            pd.DataFrame: DataFrame containing segment-level statistics.
        """

        segment_stats = group.groupby('segment_id').agg(
            motion=('motion', 'first'),
            track_id=('track_id', 'first'),
            var_y=('y', 'var'),
            x_0=('x', 'first'),
            x_1=('x', 'last'),
            y_0=('y', 'first'),
            y_1=('y', 'last'),
            frame_0=('frame_y', 'first'),
            frame_1=('frame_y', 'last')
        ).reset_index()

        segment_stats['dx'] = (segment_stats['x_1'] - segment_stats['x_0']) * self.pixel_size
        segment_stats['dy'] = (segment_stats['y_1'] - segment_stats['y_0']) * self.pixel_size
        segment_stats['dt'] = (segment_stats['frame_1'] - segment_stats['frame_0']) * self.time_between_frames
        segment_stats['dy/dt'] = segment_stats['dy'] / segment_stats['dt']

        return segment_stats

    def calculate_statistics(self, df):
        """Calculates statistics for all tracks in the DataFrame.

        Groups the input DataFrame by 'track_id' and applies the `_calculate_statistics_group`
        function to each track.

        Args:
            df (pd.DataFrame): DataFrame containing track data with 'track_id', 'x', 'y', 'frame_y', 'motion', and 'segment_id' columns.

        Returns:
            pd.DataFrame: DataFrame containing segment-level statistics for all tracks.
        """
        return df.groupby('track_id').apply(self._calculate_statistics_track).reset_index(drop=True)

    def calculate_sigma(self, per_segment_info_df):
        """Calculates the mean variance in y for stationary segments.

        This value is used as a threshold for filtering out small, spurious movements.

        Args:
            per_segment_info_df (pd.DataFrame): DataFrame containing segment statistics.

        Returns:
            float: The mean variance in the y-coordinate for stationary segments.
        """
        stationary_segments  = per_segment_info_df[per_segment_info_df['motion'] == False]
        sigma_y = stationary_segments['var_y'].mean()
        print(f"sigma_y = {sigma_y}")
        return sigma_y
