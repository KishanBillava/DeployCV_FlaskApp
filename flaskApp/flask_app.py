from logging import error
from flask import Flask, render_template
from flask import request 
import os
import pickle

#--------------- ML Libary ----------------# 
import numpy as np
import pandas as pd 
import scipy 
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# skimage
import skimage
import skimage.color
import skimage.transform
import skimage.feature
import skimage.io

# ------------------ ML Libary ------------------# 

app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')
MODEL_PATH = os.path.join(BASE_PATH,'static/models/')

#----------------------Load Model ---------------------------#

model_sgd_path = os.path.join(MODEL_PATH,'dsa_image_classification_sgd.pickle')
scalar_path = os.path.join(MODEL_PATH,'dsa_scalar.pickle')

model_sgd = pickle.load(open(model_sgd_path,'rb'))
scalar_sgd = pickle.load(open(scalar_path, 'rb'))

@app.errorhandler(404)
def error404():
    render_template("error.html")

@app.errorhandler(405)
def error405():
    render_template("error.html")

@app.errorhandler(500)
def error500():
    render_template("error.html")


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        
        upload_file  = request.files['image_name']
        filename = upload_file.filename
        print('file name uploaded is  ',filename)
        # know the extenstion of the filename .jpg  .png  .jpeg
        # 
        ext = filename.split('.')[-1]
        print('The extension of the filename = ', ext)
        if ext.lower() in ['png', 'jpg', 'jpeg']:
            # save the image 
            path_save  = os.path.join(UPLOAD_PATH,filename)
            upload_file.save(path_save)
            # send to pipeline model
            results = pipeline_model(path_save,scalar_sgd,model_sgd)

            hei = getheight(path_save)

            print(results)
            print(filename)
            return render_template('upload.html', fileupload=True,extention=False, data= results, image=filename, height=hei)


        else:
            print('Use only the Extenstion with .png .jpg .jpeg ')
            return render_template('upload.html', extention=True, fileupload=False)



    else:
        return render_template('upload.html', fileupload=False)


def getheight(path):
    img = skimage.io.imread(path)
    h,w,_ = img.shape
    ascept = h/w 
    given_width = 300
    height = given_width*ascept
    return height



def pipeline_model(path,scalar_transform,model_sgd):
    # pipeline model
    image  = skimage.io.imread(path)
    # transform image into 80x80
    image_resize = skimage.transform.resize(image, (80,80))
    image_scale = 255*image_resize
    image_transform = image_scale.astype(np.uint8)
    # rgb to gray
    gray = skimage.color.rgb2gray(image_transform)
    # hog feature 
    feature_vector = skimage.feature.hog(    gray,
                                        orientations=9,
                                        pixels_per_cell=(8, 8),
                                        cells_per_block=(2, 2),)
    
    #scaling
    scalex = scalar_transform.transform(feature_vector.reshape(1,-1))
    result  = model_sgd.predict(scalex)
    
    # decision 
    decision_value = model_sgd.decision_function(scalex).flatten()  # update model 
    labels = model_sgd.classes_
    # probability & confidence 
    z = scipy.stats.zscore(decision_value)
    prob_value = scipy.special.softmax(z)
    prob_value
    # top 5 value
    top5_prob_ind  = prob_value.argsort()[::-1][:5]
    top_lables = labels[top5_prob_ind]
    top_prob = prob_value[top5_prob_ind]
    # put in dict
    top_dict = dict()
    for key,value in zip(top_lables, top_prob):
        top_dict.update({key:np.round(value,2)})

    
    return top_dict


if __name__ == "__main__":
    app.run(debug=True)