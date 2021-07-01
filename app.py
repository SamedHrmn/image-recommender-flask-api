from flask import Flask, render_template, request, json, make_response
from werkzeug.utils import secure_filename
import os
from model_optimized import extract_feature_from_uploaded_image
import base64
from werkzeug.serving import WSGIRequestHandler
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(ROOT_DIR, "upload")
STATIC_FOLDER = os.path.join(ROOT_DIR, "static")
app = Flask(static_folder=STATIC_FOLDER, import_name=__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER


@app.route('/')
def main_page():
    return render_template('form.html')


@app.route('/prediction', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = img.convert('RGB')
        img = img.resize((224, 224))
        ref_image_path = app.config['UPLOAD_FOLDER'] + "\\" + "reference_image.jpg"
        img.save(ref_image_path, "JPEG", optimize=True)

        my_list = extract_feature_from_uploaded_image(image_path=ref_image_path)

        for f in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, f))

        prefix = 'static//fashion_clear//'
        new_list = (prefix + i for i in my_list)
        new_list = list(new_list)

        response_img = []
        encoded_img = []

        for i in range(0, len(new_list)):
            with open(str(new_list[i]), "rb") as im:
                response_img.append(im.read())

        for i in range(0, len(response_img)):
            encoded_img.append(base64.b64encode(response_img[i]).decode("ascii"))

        response = make_response(json.dumps(encoded_img))
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response


WSGIRequestHandler.protocol_version = "HTTP/1.1"

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
