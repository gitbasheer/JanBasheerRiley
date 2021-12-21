import os
from uuid import uuid4

from flask import Flask, request, render_template, send_from_directory

app = Flask(__name__, static_folder= 'static')  # Need this because Flask sets up some paths behind the scenes
# app = Flask(__name__, static_folder="images")



APP_ROOT = os.path.dirname(os.path.abspath(__file__))

classes = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise'] # this is what we will see in html page

@app.route("/") # by this index function will converted into flask function
def index():
    return render_template("index.html")

@app.route("/modelinfo")
def modelInfo():
    return render_template("modelinfo.html")

@app.route("/model")
def model():
    return render_template("model.html")

@app.route("/model", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'test')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        img = upload.filename

        destination = "/".join([target, img])
        print ("Accept incoming file:", img)
        img2=img
        print ("Save it to:", destination)
        upload.save(destination)
        #import tensorflow as tf
        # import numpy as np
        # from keras.preprocessing import image
        #
        # from keras.models import load_model
        # new_model = load_model('cat_dog_classifier.h5')
        # new_model.summary()
        # test_image = image.load_img('images\\'+filename,target_size=(128,128))
        # test_image = image.img_to_array(test_image)
        # test_image = np.expand_dims(test_image, axis = 0)
        # result = new_model.predict(test_image)
        import numpy as np
        from tensorflow.keras.preprocessing import image
        from tensorflow.keras.models import load_model
        import matplotlib.pyplot as plt
        
        new_model = load_model('Emotion_CNN.h5')
        # new_model.summary()
        img = image.load_img('test/'+img, target_size=(48,48), color_mode="grayscale")
        img = np.array(img)
        img = np.expand_dims(img,axis = 0) #makes image shape (1,48,48)
        img = img.reshape(-1,48,48,1)
        result = new_model.predict(img)
        result = list(result[0])
        label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
        img_index = result.index(max(result))
        prediction =label_dict[img_index]
      
        
        # result1 = result[0]
        # for i in range(6):
        #
        #     if result1[i] == 1.:
        #         break;
        # prediction = classes[i]

    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("prediction.html",image_name=img2, text=prediction)

@app.route('/pediction/<img>')
def send_image(img):
    # print(send_from_directory("/test",img))

    return send_from_directory("test/",img)

if __name__ == "__main__":
    
    app.run(host='0.0.0.0', port=8080)

