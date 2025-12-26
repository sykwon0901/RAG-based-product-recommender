# Data (not tracked by git)

This project requires two raw CSV files that are intentionally **excluded from git** to keep the repository lightweight.

## Required files

Place the following files under this `data/` directory:

- `amazon_face_core_meta.csv`
- `amazon_face_core_reviews.csv`

Expected structure:

data/
amazon_face_core_meta.csv
amazon_face_core_reviews.csv


## Download (Google Drive)

Download both files from the Google Drive folder below and put them into `data/`:

- Google Drive folder: https://drive.google.com/drive/folders/1gT7IrUfiR8UqWbUcVT4DZ4LlZJ2LMvGF

Access: anyone with the link can view/download (no edit permission needed).

## Notes

- Do **not** commit these raw data files to the repository.
- If you want to run the pipeline on a different machine, make sure the file names match exactly.
- If your environment uses a different data path, update the corresponding config or script parameters accordingly.
