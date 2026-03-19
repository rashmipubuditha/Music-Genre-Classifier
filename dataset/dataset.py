import pandas as pd

df = pd.read_csv("dataset/music_dataset.csv")

df = df[['artist_name','track_name','release_date','genre','lyrics']]

df.to_csv("dataset/Mendeley_cleaned.csv",index=False)