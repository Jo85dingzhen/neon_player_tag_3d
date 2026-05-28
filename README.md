# AprilTag 3D Coordinate Report

This folder contains a single Python script, `tag_3d.py`, for converting Pupil Labs Neon Head Pose Tracker AprilTag model data into a room-based 3D coordinate report.

## What It Does

`tag_3d.py` reads the Head Pose Tracker model CSV, converts marker positions from model units into centimeters, and generates an interactive HTML report named `coordinate_report.html`.

The coordinate system is:

- Origin: room corner at floor level, `(0, 0, 0)`
- X axis: rightward along the wall
- Y axis: upward from the floor
- Z axis: away from the wall into the room
- Unit: centimeters

## Required File

Before running the script, place the model CSV here:

```text
exports/000/head_pose_tracker_model.csv
```

The script uses this relative path, so the project can be moved to another computer without editing the Python file.

Expected folder layout:

```text
neon_player/
  tag_3d.py
  exports/
    000/
      head_pose_tracker_model.csv
```

## Install Dependencies

```bash
pip install pandas numpy
```

## Run

```bash
python tag_3d.py
```

After running, the script will create `coordinate_report.html` and open it in the default browser.

## Notes

Only `tag_3d.py` is required as the main code file. Generated reports, exported data, cache folders, and local environment files are ignored by `.gitignore`.
