import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import numpy as np
from PIL import Image

import os
import similarity_search
# import gradio as gr
from flask import Flask, request, jsonify
# import cv2

# def transform_image(pillow_image):
#     data = np.asarray(pillow_image)
#     data = data / 255.0
#     data = data[np.newaxis, ..., np.newaxis]
#     # --> [1, x, y, 1]
#     data = tf.image.resize(data, [224, 224])
#     return data


# import numpy as np
# import cv2

def transform_image(pillow_image):
    data = np.asarray(pillow_image)
    data = data / 255.0
    data = data.resize((224, 224))
    
    return data

def get_similar_images(image):
    similar_image_ids = similarity_search.find(image)
    return similar_image_ids

def read_image(image_file):
    try:
        img = Image.open(image_file)
        img = img.convert('RGB')
        return img
    except Exception as e:
        raise ValueError('Failed to read {}: {}'.format(image_file, str(e)))


def predict(image):

    similar_image_ids = get_similar_images(image)

    return {"similar_image_ids" : similar_image_ids}

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            upload_dir = "uploads"  
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)

            file_path = os.path.join(upload_dir, file.filename)
            file.save(file_path)
            pillow_img = read_image(file_path)
            prediction = predict(pillow_img)
            data = {"prediction": prediction}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"


if __name__ == "__main__":
    app.run(debug=True)