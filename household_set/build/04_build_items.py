"""Step 4 of the household set: write items.csv and the images, one row per photo, with the answers and the wrong options.

Reads photos.csv (the variant of each photo) and labels.csv (the names of each variant in English, French and Mooré, and the names that count as
a correct answer), and the photos extracted in step 3 (candidates/). For a West African photo the first wrong option is the Western look-alike,
for a Western photo it is a West African variant of the same category. The two other wrong options are names from other categories.
"""
import os
import random
import shutil

import pandas as pd

photos = pd.read_csv("photos.csv", dtype={"id": str})
labels = pd.read_csv("labels.csv").fillna("")
rng = random.Random(20261007)
by_variant = {label.variant: label for label in labels.itertuples()}


def same_name(first, second):
    return first.en == second.en or first.fr == second.fr or (first.mo and first.mo == second.mo)


rows = []
for photo in photos.itertuples():
    label = by_variant[photo.variant]
    western = next(r for r in labels.itertuples() if r.row == label.row and r.kind == "western")
    if photo.group == "west_african":
        wrong = [western]
    else:
        wrong = [rng.choice([r for r in labels.itertuples() if r.row == label.row and r.kind == "west_african"])]
    others = [r for r in labels.itertuples() if r.row != label.row and not any(same_name(r, t) for t in [label] + wrong)]
    rng.shuffle(others)
    others.sort(key=lambda r: r.mo == "")  # names that have a Mooré word come first
    for other in others:
        if len(wrong) == 3:
            break
        if not any(same_name(other, w) for w in wrong):
            wrong.append(other)
    own_english, own_french = set(label.syn_en.split("|")), set(label.syn_fr.split("|"))
    lookalike = [t for t in western.syn_en.split("|") if t not in own_english] + [t for t in western.syn_fr.split("|") if t not in own_french]
    rows.append(
        {
            "id": photo.id, "group": photo.group, "category": label.row_name, "variant": photo.variant, "country": photo.country,
            "english": label.en, "french": label.fr, "moore": label.mo,
            "accepted_english": label.syn_en, "accepted_french": label.syn_fr,
            "lookalike_terms": "|".join(lookalike) if photo.group == "west_african" else "",
            "wrong_english": "|".join(w.en for w in wrong), "wrong_french": "|".join(w.fr for w in wrong), "wrong_moore": "|".join(w.mo for w in wrong),
        }
    )

items = pd.DataFrame(rows)
shutil.rmtree("../images", ignore_errors=True)
os.makedirs("../images")
for photo_id in items["id"]:
    shutil.copy(f"candidates/{photo_id}.jpg", f"../images/{photo_id}.jpg")
items.to_csv("../items.csv", index=False)
print(len(items), items["group"].value_counts().to_dict(), items["category"].nunique(), "categories")
