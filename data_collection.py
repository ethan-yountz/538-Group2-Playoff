from pathlib import Path

import kagglehub


data_dir = Path(__file__).resolve().parent / "Data"
data_dir.mkdir(parents=True, exist_ok=True)

# Download latest version into the local Data folder.
path = kagglehub.dataset_download(
    "eoinamoore/historical-nba-data-and-player-box-scores",
    output_dir=str(data_dir),
)

print("Path to dataset files:", path)
