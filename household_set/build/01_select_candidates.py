"""Step 1 of the household set: choose candidate Dollar Street photos for each of the 17 categories.

Reads only the metadata columns of the Hugging Face mirror of Dollar Street (iamshnoo/dollarstreet), so no image is downloaded. For each category it
picks up to 24 photos from West African homes (Burkina Faso, Côte d'Ivoire and Togo first, then Ghana, Nigeria and Liberia) and 12 photos from
Western homes, with at most 3 (or 2) photos per home. The list is written to candidates.csv and pasted into the Kaggle notebook of step 2.
"""
import collections
import re

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

topics = [
    "cooking_pots", "cooking", "stove_hob", "drinking_water", "get_water", "water_sources", "water_outlet", "kitchen_sink", "grains",
    "cooking_utensils", "plate_of_food", "table_with_food", "place_where_eating_dinner", "cleaning_equipment", "cleaning_floors",
    "armchair", "couch", "sitting_area", "bed", "toilet", "light_source_in_kitchen", "light_source_in_livingroom", "light_sources",
    "home", "roof", "front_door", "bike", "moped_motorcycle", "car", "everyday_shoes", "clothes", "radio", "tv", "phone", "toothbrush",
    "goats", "chickens", "pet",
]
west_african_first = ["Burkina Faso", "Cote d'Ivoire", "Togo"]
west_african_fallback = ["Ghana", "Nigeria", "Liberia"]
western = ["United States", "United Kingdom", "France", "Spain", "Netherlands", "Austria", "Denmark", "Sweden", "Switzerland", "Italy", "Canada"]

# category number -> (Dollar Street topics for West African homes, topics for Western homes)
categories = {
    1: (["cooking_pots"], ["cooking_pots"]),
    2: (["cooking"], ["stove_hob"]),
    3: (["drinking_water", "get_water"], ["drinking_water", "kitchen_sink"]),
    4: (["get_water", "water_outlet", "water_sources"], ["kitchen_sink", "water_outlet"]),
    5: (["grains", "cooking_utensils"], ["cooking_utensils", "grains"]),
    6: (["place_where_eating_dinner", "plate_of_food", "table_with_food"], ["table_with_food", "plate_of_food"]),
    7: (["cleaning_equipment", "cleaning_floors"], ["cleaning_equipment", "cleaning_floors"]),
    8: (["sitting_area", "armchair"], ["couch"]),
    9: (["bed"], ["bed"]),
    10: (["toilet"], ["toilet"]),
    11: (["light_source_in_kitchen", "light_source_in_livingroom"], ["light_source_in_kitchen", "light_source_in_livingroom", "light_sources"]),
    12: (["home"], ["home"]),
    13: (["roof"], ["roof"]),
    14: (["front_door"], ["front_door"]),
    15: (["bike", "moped_motorcycle"], ["car"]),
    16: (["everyday_shoes"], ["everyday_shoes"]),
    17: (["clothes"], ["clothes"]),
    18: (["radio", "phone"], ["tv"]),
    19: (["toothbrush"], ["toothbrush"]),
    20: (["goats", "chickens"], ["pet"]),
}


def read_metadata():
    fs = HfFileSystem()
    shards = collections.defaultdict(list)
    for path in fs.ls("datasets/iamshnoo/dollarstreet/data", detail=False):
        shards[re.match(r"^(.*)-\d{5}-of-\d{5}-", path.split("/")[-1]).group(1)].append(path)
    tables = []
    for topic in topics:
        for path in sorted(shards[topic]):
            with fs.open(path, "rb") as file:
                table = pq.ParquetFile(file).read(columns=["id", "country_name", "place", "income"]).to_pandas()
            tables.append(table.assign(topic=topic))
    return pd.concat(tables, ignore_index=True)


def take(photos, count, per_home):
    photos = photos.sample(frac=1, random_state=1)
    chosen, taken = [], collections.Counter()
    for photo in photos.itertuples():
        if taken[(photo.topic, photo.place)] < per_home:
            chosen.append(photo)
            taken[(photo.topic, photo.place)] += 1
        if len(chosen) == count:
            break
    return chosen


metadata = read_metadata()
candidates = []
for number, (west_african_topics, western_topics) in categories.items():
    photos = metadata[metadata["topic"].isin(west_african_topics)]
    chosen = {"west_african": take(photos[photos["country_name"].isin(west_african_first)], 24, 3)}
    if len(chosen["west_african"]) < 24:
        chosen["west_african"] += take(photos[photos["country_name"].isin(west_african_fallback)], 24 - len(chosen["west_african"]), 3)
    chosen["western"] = take(metadata[metadata["topic"].isin(western_topics) & metadata["country_name"].isin(western)], 12, 2)
    for group, rows in chosen.items():
        candidates += [{"row": number, "group": group, "id": r.id, "topic": r.topic, "country": r.country_name, "place": r.place, "income": float(r.income)} for r in rows]
pd.DataFrame(candidates).to_csv("candidates.csv", index=False)
print(len(candidates), "candidate photos")
