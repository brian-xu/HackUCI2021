import base64
import io
import optparse
import os

import cv2
import flask
import numpy as np
import requests
from PIL import Image

import forward

UPLOAD_FOLDER = '/tmp/demo'
ALLOWED_EXTENSIONS = {'bmp', 'jpeg', 'jpg', 'jpe', 'png', 'pbm', 'tif'}
classes = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3',
           'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious',
           'Apple Red Yellow 1', 'Apple Red Yellow 2', 'Apricot', 'Avocado', 'Avocado ripe', 'Banana',
           'Banana Lady Finger', 'Banana Red', 'Beetroot', 'Blueberry', 'Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2',
           'Carambula', 'Cauliflower', 'Cherry 1', 'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red',
           'Cherry Wax Yellow', 'Chestnut', 'Clementine', 'Cocos', 'Corn', 'Corn Husk', 'Cucumber Ripe',
           'Cucumber Ripe 2', 'Dates', 'Eggplant', 'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue', 'Grape Pink',
           'Grape White', 'Grape White 2', 'Grape White 3', 'Grape White 4', 'Grapefruit Pink', 'Grapefruit White',
           'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 'Lemon', 'Lemon Meyer', 'Limes',
           'Lychee', 'Mandarine', 'Mango', 'Mango Red', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry',
           'Nectarine', 'Nectarine Flat', 'Nut Forest', 'Nut Pecan', 'Onion Red', 'Onion Red Peeled', 'Onion White',
           'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach 2', 'Peach Flat', 'Pear', 'Pear 2', 'Pear Abate',
           'Pear Forelle', 'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams', 'Pepino',
           'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow', 'Physalis', 'Physalis with Husk',
           'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2', 'Plum 3', 'Pomegranate', 'Pomelo Sweetie',
           'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince', 'Rambutan', 'Raspberry',
           'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato 1', 'Tomato 2',
           'Tomato 3', 'Tomato 4', 'Tomato Cherry Red', 'Tomato Heart', 'Tomato Maroon', 'Tomato Yellow',
           'Tomato not Ripened', 'Walnut', 'Watermelon']

app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        byte_stream = io.BytesIO(requests.get(imageurl).content)
        image = cv2.imdecode(np.frombuffer(byte_stream.read(), np.uint8), 1)

    except:
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    result = app.predictor.predict(image)
    return flask.render_template(
        'index.html', has_result=True,
        result=[(data_uri_encoder(img), f"{classes[output]}") for img, output in result]
    )


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        imagefile = flask.request.files['imagefile']
        byte_stream = io.BytesIO()
        imagefile.save(byte_stream)
        pimg = Image.open(byte_stream)
        image = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

    except Exception as err:
        print(err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    result = app.predictor.predict(image)
    return flask.render_template(
        'index.html', has_result=True,
        result=[(data_uri_encoder(img), f"{classes[output]}") for img, output in result]
    )


def data_uri_encoder(img: np.array):
    image_pil = Image.fromarray(img)
    byte_stream = io.BytesIO()
    image_pil.save(byte_stream, format='png')
    data = base64.b64encode(byte_stream.getvalue()).decode("utf-8")
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return '.' in filename and filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS


def start_from_terminal(app):
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()

    app.predictor = forward.Predictor(opts.gpu)

    app.run(debug=opts.debug, host='127.0.0.1', port=opts.port)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
