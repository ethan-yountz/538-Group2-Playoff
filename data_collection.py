from pathlib import Path


data_dir = Path(__file__).resolve().parent / "Data"
data_dir.mkdir(parents=True, exist_ok=True)


def download_kaggle_dataset() -> str:
    # Download the Kaggle source once into the project Data folder.
    import kagglehub

    path = kagglehub.dataset_download(
        "eoinamoore/historical-nba-data-and-player-box-scores",
        output_dir=str(data_dir),
    )
    return str(path)


def download_538_elo() -> Path:
    # Download and keep only seasons >= start_year from Neil-Paine's public NBA Elo history.
    start_year = 2020
    import csv
    from urllib.request import Request, urlopen

    url = "https://raw.githubusercontent.com/Neil-Paine-1/NBA-elo/main/nba_elo.csv"
    elo_path = data_dir / "nba_elo.csv"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    year_col = "season"

    with urlopen(request, timeout=120) as response:
        reader = csv.DictReader((line.decode("utf-8") for line in response))
        with elo_path.open("w", newline="", encoding="utf-8") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=reader.fieldnames or [])
            writer.writeheader()
            for row in reader:
                if row and int(row.get(year_col, "0")) >= start_year:
                    writer.writerow(row)
    return elo_path


# download_kaggle_dataset()
# download_538_elo()
