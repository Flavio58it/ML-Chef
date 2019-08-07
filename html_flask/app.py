import os
import io
import numpy as np

import keras
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications.xception import (
    Xception, preprocess_input, decode_predictions)
from keras import backend as K

from flask import Flask, request, redirect, url_for, jsonify
from keras.models import load_model

from flask import Flask, render_template, redirect
from flask_pymongo import PyMongo

# , static_url_path='/static')
app = Flask(__name__, template_folder='./templates')

# Use flask_pymongo to set up mongo connection
app.config["MONGO_URI"] = "mongodb://localhost:27017/ML-Chef_db"
mongo = PyMongo(app)

model = None
graph = None

def prep_model():
    global model
    global graph
    model = Xception(weights="imagenet")
    graph = K.get_session().graph


# load_model()
prep_model()


def prepare_image(img):
    img = img_to_array(img)

    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    # return the processed image
    return img


@app.route('/', methods=['GET', 'POST'])
def index():
    data = {"success": False}

    # print(request.files and list(request.files.keys()))
    if (request.method == 'POST'
        and request.files.get('inputGroupFile01')
        and request.files.get('inputGroupFile02')
            and request.files.get('inputGroupFile03')):
        # read the file
        user_file = request.files['inputGroupFile01']

        # read the filename
        filename = user_file.filename

        # create a path to the uploads folder
        filepath = os.path.join('uploads', filename)

        user_file.save(filepath)

        # Load the saved image using Keras and resize it to the trained model size
        # format of 299x299 pixels
        image_size = (299, 299)
        im = keras.preprocessing.image.load_img(filepath,
                                                target_size=image_size,
                                                grayscale=False)

        # preprocess the image and prepare it for classification
        image = prepare_image(im)

        global graph
        with graph.as_default():
            preds = model.predict(image)
            results = decode_predictions(preds)
            # print the results
            print(results)

            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True
            first_image = data['predictions'][0]

            # read the file
        user_file = request.files['inputGroupFile02']

        # read the filename
        filename = user_file.filename

        # create a path to the uploads folder
        filepath = os.path.join('uploads', filename)

        user_file.save(filepath)

        # Load the saved image using Keras and resize it to the trained model size
        # format of 299x299 pixels
        image_size = (299, 299)
        im = keras.preprocessing.image.load_img(filepath,
                                                target_size=image_size,
                                                grayscale=False)

        # preprocess the image and prepare it for classification
        image = prepare_image(im)

        with graph.as_default():
            preds = model.predict(image)
            results = decode_predictions(preds)
            # print the results
            print(results)

            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True
            second_image = data["predictions"][0]

            # read the file
        user_file = request.files['inputGroupFile03']

        # read the filename
        filename = user_file.filename

        # create a path to the uploads folder
        filepath = os.path.join('uploads', filename)

        user_file.save(filepath)

        # Load the saved image using Keras and resize it to the trained model size
        # format of 299x299 pixels
        image_size = (299, 299)
        im = keras.preprocessing.image.load_img(filepath,
                                                target_size=image_size,
                                                grayscale=False)

        # preprocess the image and prepare it for classification
        image = prepare_image(im)

        with graph.as_default():
            preds = model.predict(image)
            results = decode_predictions(preds)
            # print the results
            print(results)

            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True
            third_image = data["predictions"][0]

            label1 = first_image['label']
            label2 = second_image['label']
            label3 = third_image['label']

            # Search database for recipes matching labels
            recipe = mongo.db.ML_Chef.find({'tag': [label1,label2,label3]})
            print(recipe)

            data_dict = {'first_image': first_image,
                         'second_image': second_image, 
                         'third_image': third_image}

            return render_template("index.html", data1=data_dict)

    return render_template("index.html", data1={})#, data1=data_dict)  # , data1 = data)


if __name__ == "__main__":
    app.run(debug=True)
