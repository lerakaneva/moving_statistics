# Platelet Motion Analysis

This project analyzes platelet motion from tracked CSV files.  It segments platelet tracks into moving and stationary segments, calculates various motion metrics (velocity, distance, duration), filters out spurious movements, and provides summary statistics.

## Features

* **Track Segmentation:** Identifies motion and stationary segments within individual platelet tracks.
* **Motion Metrics Calculation:** Computes velocity, travel distance, and duration for each segment.
* **Sigma-Based Filtering:**  Filters out noise and spurious movements based on a user-defined or automatically calculated threshold.
* **Summary Statistics:** Generates summary statistics for overall platelet motion behavior.

## Requirements

* Python 3.7+
* Pandas
* NumPy
* Configparser

## Installation

1. Clone the repository
2. Install required packages:
pip install -r requirements.txt

## Usage

1. **Configuration:** Create a configuration file (`config.ini`) with the following structure.  Adjust the values as needed for your data:

```ini
[DEFAULT]
csv_folder_path = /path/to/your/csv/files  # Replace with the actual path
pixel_size = 0.431 # Pixel size in micrometers
time_between_frames = 0.05  # Time between frames in seconds
save_filtered = True  # Save tracks with stationary, moving, and spurious segments (True/False)
sigma = 1.0 # Sigma for displacement filtering (0.0 to use calculated sigma)
```

2. **Run the analysis:**
```
python main.py --config config_example.ini
```

## Output

The analysis results are saved in separate CSV files within an output directory named `<input_file_name>_output` for each input CSV file. The output files include:

* `processed_data.csv`: Track data with segment information (motion/motion final labels, segment IDs).
* `statistics.csv`: Segment-level statistics (displacement, duration, velocity, etc.).
* `moving.csv`: Metrics for moving segments.
* `stop.csv`: Metrics for stationary segments of moving platelets.
* `ratios.csv`: Time ratios (moving time/total time) per track for moving platelets.
* `summary_stats.csv`: Overall summary statistics for motion metrics.
* `final_counts.csv` : Number of total tracks and a classification based on time ratios: moving, bumped, and adhered.
* `counts.csv`: Counts per frame for moving, bumped, adhered, and total tracks.
* `trajectories_moving.csv`: Sub-trajectories of the moving segments.
* `trajectories_not_moving.csv`: Sub-trajectories of non-moving segments.
* `trajectories_moving_false_positive.csv`: Sub-trajectories of initially identified moving segments filtered out by the sigma filter.


## Project Structure

* `data_handling.py`: Contains classes for configuration reading and data I/O.
* `track_segmentation.py`: Contains classes for track segmentation and segment statistics calculation.
* `motion_analysis.py`: Contains classes for motion analysis and sigma filtering.
* `main.py`: Main script to run the analysis.
