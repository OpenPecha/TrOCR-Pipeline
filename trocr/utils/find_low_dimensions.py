import csv
import pandas as pd
from pathlib import Path

df = pd.read_csv("../tibetan-dataset/labels_new.csv", delimiter=",")

print(df.head())

anomolies = Path('../tibetan-dataset/Anomolies')
anomoly_files = list(anomolies.iterdir())
anomoly_file_names = set([x.name for x in anomoly_files])

filtered_df = df[df['ImageName'].isin(anomoly_file_names)]

filtered_df.to_csv('anomolies.csv', index=False)

Images_to_remove = filtered_df["ImageName"].unique()

updated_df = df[~df['ImageName'].isin(Images_to_remove)]

updated_df.to_csv("labels_new.csv", index=False)