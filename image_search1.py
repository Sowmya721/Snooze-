# -*- coding: utf-8 -*-
"""image.search

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1EvC28yxIjBJ5Phj-qa2GaMQg23R6ZEka
"""

import pandas as pd
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("/content/drive/MyDrive/datste/preprocessed_dataset_curatiled.csv")

# Optional: filter out rows without images
df = df[df['images'].notnull()]

def preprocess_image(img_url):
    try:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)
    except:
        return None

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_url):
    img = preprocess_image(img_url)
    if img is None:
        return None
    features = model.predict(img, verbose=0)
    return features.flatten()

image_features = []
valid_indices = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_urls = eval(row['images'])  # Ensure it's a list of URLs
    if img_urls:
        features = extract_features(img_urls[0])
        if features is not None:
            image_features.append(features)
            valid_indices.append(idx)

# Filter the original dataframe to only include valid images
df = df.iloc[valid_indices].reset_index(drop=True)
image_features = np.array(image_features)

def recommend_similar_images(input_image_path, top_k=5):
    # Load and extract features for input image
    img = Image.open(input_image_path).convert('RGB').resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    query_feature = model.predict(img_array, verbose=0).flatten()

    # Compute cosine similarity
    similarities = cosine_similarity([query_feature], image_features)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    print("Top Matches:")
    for idx in top_indices:
        print(f"Title: {df.iloc[idx]['title']}, Link: {df.iloc[idx]['url']}")
        img_urls = eval(df.iloc[idx]['images'])
        if img_urls:
            display_image(img_urls[0])

def display_image(img_url):
    try:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    except:
        print("Could not load image.")
        
def recommend_similar_images_from_upload(local_image_path, top_k=5):
    img = Image.open(local_image_path).convert('RGB').resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    query_feature = model.predict(img_array, verbose=0).flatten()

    similarities = cosine_similarity([query_feature], image_features)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        img_urls = eval(row["images"])
        results.append({
            "title": row["title"],
            "price": row["selling_price"],
            "url": row["url"],
            "image_url": img_urls[0] if img_urls else ""
        })
    return results


# Provide path to your test image (upload it or give path)
#![images](images.jpeg)
#image_path = "images.jpeg"
#recommend_similar_images("/content/drive/MyDrive/datste/images.jpeg", top_k=5)

