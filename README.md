# moore-vision-blind-spot

Do small vision-language models recognise West African things, and can they name them in Mooré? This is the code for my entry to the Fatima Fellowship challenge. The results, the fine-tuned adapters and the written answers are in the public bucket [anyantudre/moore-vision-blind-spot](https://huggingface.co/buckets/anyantudre/moore-vision-blind-spot).

## What is here

- `notebooks/01_moore_dictionary_curation.ipynb` builds the dictionary photos from three Mooré dictionaries. It is the same notebook as in [moore-image-lexicon](https://github.com/anyantudre/moore-image-lexicon).
- `notebooks/02_blind_spot_evaluation_and_finetuning.ipynb` evaluates Qwen3-VL-4B-Instruct, Qwen3-VL-2B-Instruct and SmolVLM2-2.2B-Instruct on the dictionary test photos and on the household photos, then fine-tunes Qwen3-VL-4B with and without Mooré names and tests it again. Both notebooks have a `.py` version of the same code.
- `household_set/` has 222 household photos from Dollar Street with their names in English, French and Mooré (`items.csv`, `images/`). `household_set/build/` has the four steps that made it: the candidate list, the Kaggle notebook that cuts the photos out of Dollar Street, the contact sheets I labelled by eye, and the script that writes `items.csv`.
- `results/` and `figures/` are the outputs of the Kaggle run of notebook 2.

## Running it

Both notebooks run on Kaggle with a T4 GPU and a Hugging Face token in the secret `HF_TOKEN`. Notebook 2 loads the dataset and the household set from the Hub.

## Rights

The household photos are from Dollar Street (Gapminder and MLCommons), CC BY-SA 4.0. The dictionary photos come from dictionaries without an open licence, see the dataset card. The code is under the MIT licence.
