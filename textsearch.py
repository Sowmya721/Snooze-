# -*- coding: utf-8 -*-

from google.colab import files
uploaded = files.upload()


import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from IPython.display import display, HTML

# Load dataset (make sure file name matches the one you uploaded)
df = pd.read_csv("preprocessed_dataset.csv")

# Load model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# ----------------------------
# Step 1: Embed product data
# ----------------------------
def embed_product(product):
    text = f"{product['title']} {product['product_details']}"
    return model.encode(text)

print("Embedding all products...")

product_embeddings = [embed_product(row) for _, row in df.iterrows()]

print("Embedding complete!")

# ----------------------------
# Step 2: Generate keyword sets
# ----------------------------
def generate_keyword_sets(df):
    base_colors = set([
        "black", "white", "red", "blue", "green", "yellow", "pink", "purple", "grey", "gray", "brown",
        "orange", "beige", "maroon", "navy", "cream", "gold", "silver", "charcoal", "ivory", "mint"
    ])

    color_keywords = set()
    category_keywords = set()

    for text in df['title'].tolist() + df['product_details'].tolist():
        words = str(text).lower().split()
        for word in words:
            if word in base_colors:
                color_keywords.add(word)

    for cat in df['sub_category'].dropna().unique():
        for token in str(cat).lower().split():
            category_keywords.add(token.strip())

    return list(color_keywords), list(category_keywords)

color_keywords, category_keywords = generate_keyword_sets(df)

# ----------------------------
# Step 3: Parse user query
# ----------------------------
def parse_query(query):
    query = query.lower()
    tokens = query.split()

    color = next((token for token in tokens if token in color_keywords), None)
    sub_cat_token = next((token for token in tokens if token in category_keywords), None)

    sub_category = None
    if sub_cat_token:
        for cat in df['sub_category'].dropna().unique():
            if sub_cat_token in cat.lower():
                sub_category = cat
                break

    return {"color": color, "sub_category": sub_category}

# ----------------------------
# Step 4: Search products
# ----------------------------
def search_products(query, top_k=10):
    parsed = parse_query(query)
    query_vec = model.encode(query)
    sims = cosine_similarity([query_vec], product_embeddings)[0]

    results = []
    for idx in np.argsort(sims)[::-1]:
        row = df.iloc[idx]

        if parsed["sub_category"]:
            if parsed["sub_category"].lower() not in row["sub_category"].lower():
                continue

        if parsed["color"]:
            if parsed["color"] not in str(row["title"]).lower() and parsed["color"] not in str(row["product_details"]).lower():
                continue

        results.append({
            "title": row["title"],
            "price": row["selling_price"],
            "url": row["url"],
            "category": row["sub_category"],
            "score": round(float(sims[idx]), 3),
            "image": row["images"]  # Make sure this column exists in your CSV
        })

        if len(results) == top_k:
            break

    return results

# ----------------------------
# Step 5: Test the search
# ----------------------------
def test_search():
    test_queries = [
        "Blue formal shirt"
    ]

    for query in test_queries:
        print(f"\n🔍 Results for: '{query}'")
        results = search_products(query, top_k=5)
        for i, item in enumerate(results, 1):
            print(f"{i}. {item['title']}")
            print(f"   Price: ₹{item['price']} | Category: {item['category']}")
            print(f"   URL: {item['url']}")
            print(f"   Score: {item['score']}\n")

test_search()