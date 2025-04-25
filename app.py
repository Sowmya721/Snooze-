from flask import Flask, render_template, request
from engine.textsearch import search_products
from image_search import recommend_similar_images_from_upload
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def home():
    text_results = []
    image_results = []
    
    if request.method == "POST":
        if 'text_query' in request.form:
            query = request.form['text_query']
            text_results = search_products(query, top_k=5)

        if 'image_file' in request.files:
            image = request.files['image_file']
            if image.filename != "":
                img_path = os.path.join(UPLOAD_FOLDER, image.filename)
                image.save(img_path)
                image_results = recommend_similar_images_from_upload(img_path, top_k=5)

    return render_template("index.html", text_results=text_results, image_results=image_results)

if __name__ == "__main__":
    app.run(debug=True)
