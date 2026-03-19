"""
Merges the Mendeley dataset (7 classes) and the Student dataset (soul)
to produce the Merged_dataset.csv with 8 genres.
"""
import pandas as pd
import os

# --- 1. Mendeley dataset ---
mendeley_path = "dataset/Mendeley_cleaned.csv"
if not os.path.exists(mendeley_path):
    # Re-generate from raw if needed
    df_raw = pd.read_csv("dataset/music_dataset.csv")
    df_mendeley = df_raw[['artist_name', 'track_name', 'release_date', 'genre', 'lyrics']]
    df_mendeley.to_csv(mendeley_path, index=False)
    print(f"Re-created {mendeley_path}")
else:
    df_mendeley = pd.read_csv(mendeley_path)

# Keep only the required columns
df_mendeley = df_mendeley[['artist_name', 'track_name', 'release_date', 'genre', 'lyrics']]

# --- 2. Student dataset ---
student_path = "dataset/Student_dataset.csv"
df_student = pd.read_csv(student_path)
df_student = df_student[['artist_name', 'track_name', 'release_date', 'genre', 'lyrics']]

# --- 3. Merge ---
df_merged = pd.concat([df_mendeley, df_student], ignore_index=True)

# Drop rows with missing lyrics or genre
df_merged = df_merged.dropna(subset=['lyrics', 'genre'])
df_merged = df_merged[df_merged['lyrics'].str.strip() != '']

# Save
output_path = "dataset/Merged_dataset.csv"
df_merged.to_csv(output_path, index=False)

print(f"Merged dataset saved to {output_path}")
print(f"Total rows: {len(df_merged)}")
print(f"\nGenre distribution:\n{df_merged['genre'].value_counts()}")
