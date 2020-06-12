#https://github.com/mohshawky5193/dog-breed-classifier/blob/master/web-app/web-app-classifier.py
#deployed at https://dog-breed-classifier-udacity.herokuapp.com/

import os
import io
from flask import Flask,request,jsonify,render_template
from fastai.basic_train import load_learner
from fastai.vision import open_image
import torch
from PIL import Image 

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def render_page():
    return render_template('cat-breed-detector.html')

@app.route('/uploadajax',methods=['POST'])
def upload_file():
    """
    retrieve the image uploaded and make sure it is an image file
    """
    file = request.files['file']
    image_extensions=['jpg', 'jpeg', 'png']
    
    if file.filename.split('.')[1] not in image_extensions:
        return jsonify('Please upload an appropriate image file')
    
    """
    Load the trained model in export.pkl 
    """
    learn = load_learner(path = ".")
    
    """
    Perform prediction
    """
    #image_bytes = file.read()
    #img = Image.open(io.BytesIO(image_bytes))
    
    img = open_image(file)
    
    pred_class,pred_idx,outputs = learn.predict(img)
    i = pred_idx.item()
    classes = ['Domestic Medium Hair', 'Persian', 'Ragdoll', 'Siamese', 'Snowshoe']
    prediction = classes[i]
    
    return jsonify(f'Your cat is a {prediction}')
    
if __name__ == '__main__':
    app.run(debug=False,port=os.getenv('PORT',5000))


