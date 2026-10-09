"""Step 3 of the household set: make one contact sheet per category so that each photo can be labelled by eye.

Reads household_candidates.zip (the output of the Kaggle notebook of step 2) and writes one PNG per category to sheets/. A sheet shows up to 16 West
African photos (labelled WA1, WA2, ...) and 8 Western photos (WE1, WE2, ...), one photo per home first. The variant of each photo (clay pot, straw
broom, pit latrine and so on) is then noted in photos.csv, and the names of the variants in labels.csv.
"""
import os
import zipfile

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

tile = 230
columns = 6
os.makedirs("candidates", exist_ok=True)
os.makedirs("sheets", exist_ok=True)
zipfile.ZipFile("household_candidates.zip").extractall("candidates")
found = pd.read_csv("candidates/found.csv", dtype={"id": str})
font = ImageFont.load_default()


def pick(photos, count):
    photos = photos.sample(frac=1, random_state=3)
    one_per_home = photos.drop_duplicates(["topic", "place"])
    rest = photos[~photos.index.isin(one_per_home.index)]
    return pd.concat([one_per_home, rest]).head(count)


for category, photos in found.groupby("row"):
    west_african = pick(photos[photos["group"] == "west_african"], 16).assign(tag="WA")
    western = pick(photos[photos["group"] == "western"], 8).assign(tag="WE")
    chosen = pd.concat([west_african, western]).reset_index(drop=True)
    chosen["label"] = [f"{tag}{n + 1}" for n, tag in enumerate(chosen["tag"])]
    sheet = Image.new("RGB", (columns * tile, ((len(chosen) + columns - 1) // columns) * (tile + 22)), "white")
    draw = ImageDraw.Draw(sheet)
    for n, photo in chosen.iterrows():
        image = Image.open(f"candidates/{photo['id']}.jpg").convert("RGB")
        image.thumbnail((tile - 4, tile - 4))
        x, y = (n % columns) * tile, (n // columns) * (tile + 22)
        sheet.paste(image, (x + 2, y + 20))
        draw.text((x + 4, y + 4), f"{photo['label']}  {photo['country'][:12]}  {photo['topic'][:14]}", fill="black", font=font)
    sheet.save(f"sheets/category{int(category):02d}.png")
    chosen[["row", "group", "label", "id", "country", "place", "topic"]].to_csv(f"sheets/category{int(category):02d}.csv", index=False)
print(len(found), "photos,", found["row"].nunique(), "contact sheets in sheets/")
