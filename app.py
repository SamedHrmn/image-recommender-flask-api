from flask import Flask, render_template, request, redirect, url_for, json
from werkzeug.utils import secure_filename
import os
from model import get_images
from flask import send_file
import base64
from werkzeug.serving import WSGIRequestHandler

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
        my_list = get_images(UPLOAD_FOLDER + '\\' + filename, recommendation_threshold=0.50)
        prefix = 'static//'
        new_list = sum(my_list, [])
        new_list = (prefix + i for i in new_list)
        new_list = list(new_list)
        print("NEWLIST images: :", new_list)
        print("NEWLIST lenght : " + str(len(new_list)))
        json_obj = {}
        response_img = []
        encoded_img = []

        for i in range(0, len(new_list)):
            with open(str(new_list[i]), "rb") as im:
                response_img.append(im.read())

        for i in range(0 , len(response_img)):

            print("repsonse_img "+str(response_img[i]))

            encoded_img.append(base64.b64encode(response_img[i]).decode("ascii"))
        print("ENCODED LEN: " + str(len(encoded_img)))
        json_obj["img"] = encoded_img
        print("Json Obj : ", json_obj)
        return json.dumps(json_obj)
        # return render_template('form.html', images=new_list)


WSGIRequestHandler.protocol_version = "HTTP/1.1"

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
