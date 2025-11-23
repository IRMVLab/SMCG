import os

with open(os.path.join(os.path.dirname(__file__), "gibson_category.txt"), "r") as f:
    lines = f.readlines()
GIBSON_CATEGORIES = [line.rstrip() for line in lines]

import pickle

with open("./utils/word2vec_pca.txt", "rb") as f:
    loaded_results = pickle.load(f)

CATEGORIES = {}
CATEGORIES["gibson"] = GIBSON_CATEGORIES
CATEGORIES["vec"] = loaded_results
