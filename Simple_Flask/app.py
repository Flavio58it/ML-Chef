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

app = Flask(__name__)

# Use flask_pymongo to set up mongo connection
app.config["MONGO_URI"] = "mongodb://localhost:27017/mars_app"
mongo = PyMongo(app)
app.config['UPLOAD_FOLDER'] = 'Uploads'


model = None
graph = None

@app.route("/")
def index():
    mars_dict = mongo.db.mars_collection.find_one()
    return render_template("index.html", mars=mars_dict)


@app.route("/scrape")
def scraper():
    mars_collection = mongo.db.mars_collection
    mars_data = scrape_mars.scrape()
    mars_collection.update({}, mars_data, upsert=True)
    return redirect("/", code=302)


if __name__ == "__main__":
    app.run(debug=True)

app = Flask(__name__)





def prep_model():
    global model
    global graph
    model = load_model("food_trained_cnn_v2.h5")
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
def upload_file():
    data = {"success": False}
    if request.method == 'POST':
        if request.files.get('file'):
            # read the file
            file = request.files['file']

            # read the filename
            filename = file.filename

            # create a path to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(filepath)

            # Load the saved image using Keras and resize it to the trained model size
            # format of 64x64 pixels
            image_size = (256, 256)
            im = keras.preprocessing.image.load_img(filepath,
                                                    target_size=image_size,
                                                    grayscale=False)

            # preprocess the image and prepare it for classification
            image = prepare_image(im)

            global graph
            with graph.as_default():
                #preds = model.predict(image)
                #x = image.img_to_array(img)
                #x = np.expand_dims(x, axis=0)
                #x = preprocess_input(x)
                class_index_to_class = {v: k for k,
                            v in training_set.class_indices.items()}
                index_max = np.argmax(model.predict(image)[0])
                prediction = class_index_to_class[index_max]
                #plt.imshow(img)
                print('Predicted: ', prediction, ', score: ', index_max)
                # print the results
                #print(results)

                data["predictions"] = []

                # loop over the results and add them to the list of
                # returned predictions
                for (imagenetID, label, prob) in results[0]:
                    r = {"label": label, "probability": float(prob)}
                    data["predictions"].append(r)

                # indicate that the request was a success
                data["success"] = True

        return jsonify(data)

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


if __name__ == "__main__":
    app.run(debug=True)
