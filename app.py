from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from model import get_images

UPLOAD_FOLDER = os.path.abspath("D://Projelerim//ML//image_recommender_flask//upload")
STATIC_FOLDER = os.path.abspath("D://Projelerim//ML//image_recommender_flask//static")
app = Flask(static_folder=STATIC_FOLDER, import_name=__name__)

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

filename = ""


@app.route('/')
def main_page():
    return render_template('form.html')



@app.route('/form', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        mylist = get_images(UPLOAD_FOLDER + '\\' + filename)
        prefix = 'static//'
        newlist = sum(mylist , [])
        newlist = (prefix + i for i in newlist)
        return render_template('form.html', images=newlist)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
