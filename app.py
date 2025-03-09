from flask import Flask, render_template, request
from keras.saving import register_keras_serializable
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

def rescale_input(image):
    return image / 255.0

# Define custom objects for model loading
custom_objects = {'rescale_input': rescale_input}

model= load_model("/Users/Syedh/Desktop/Meow/models/model.h5", custom_objects=custom_objects)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html', btext="", atext="")

@app.route('/', methods=['POST'])
def predict():

    render_template("index.html", btext="", atext="")

    imageUpload= request.files['imageUpload']
    image_path = "./images/" + imageUpload.filename
    imageUpload.save(image_path)

    expected_size = (224, 224)  # Example expected size
    image = load_img(image_path, target_size=expected_size)  # Resize the image
    image_array = img_to_array(image)  # Convert to array
    image_array = tf.expand_dims(image_array, 0) 
    #image = tf.keras.utils.load_img(image_path, target_size=(244, 244))
    #imageArray = tf.keras.utils.img_to_array(image)
    #imageArray = tf.expand_dims(imageArray, 0)

    b = model.predict(image_array)
    a = tf.nn.softmax(b[0])

    most_probable_class_idx = tf.argmax(a).numpy()  # Get the index of the most probable class

    # Optional: Define class names if you have them
    class_names = ["Persian", "Bengal", "Maine_coon", "Siamese"]

    # Get the most probable class name
    most_probable_class_name = class_names[most_probable_class_idx]

    # Get the probability of the most probable class
    most_probable_class_prob = a[most_probable_class_idx].numpy()
    most_probable_class_prob=most_probable_class_prob*100

    return render_template('index.html', btext=most_probable_class_name, atext=most_probable_class_prob)


if __name__ == '__main__':
    app.run(port=3000, debug=True)