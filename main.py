import lib as r
from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates')


@app.route('/')
def home():
    return render_template('trial.html')


@app.route("/uploaded" , methods=['GET', 'POST'])
def uploader():
    
    if request.method=='POST':
        r.eraser()

        f = request.files['file1']
        file_name = secure_filename(f.filename)
        
        file_path = "static/input/" + file_name
        file_path_out = "static/output/" + file_name

        f.save(file_path)
        r.plotter(file_path, file_path_out)
        
        r.cal_psnr(file_path, file_path_out)
        
        return render_template('trial.html', file_path = file_path, file_path_out=file_path_out)


if __name__ == "__main__":
    app.run()