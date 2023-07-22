from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2 

app = FastAPI()

saved_model_dir = "C://Users//a3141//Python//dep_folder//models//2"

model = tf.keras.models.load_model(saved_model_dir)
Class_names = ["Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Potato___Early_blight", "Potato___Late_blight", "healthy"]

#converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
#tflite_model = converter.convert()

# Save the model.
#with open('model.tflite', 'wb') as f:
#  f.write(tflite_model)

# model_lite = tf.keras.models.load_model('model.tflite')



@app.get("/ping")
async def ping(): 
    return "Hello, I am alive!"

def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image



@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image =  cv2.resize(image,(224,224))
    img_batch = np.expand_dims(image,0)
    prediction_batch = model.predict(img_batch)
    prediction = prediction_batch[0]
    max1 = 0
    m1 = -1
    max2 = 0
    m2 = -1
    i = 0
    flag = -1
    for prob in prediction:
        i = i+1
        if(max1==0 and max2==0):
            max1 = prob
            m1 = i
        else:
            if(max1<prob):
                max2 = max1
                m2 = m1
                max1 = prob
                m1 = i
            else:
                if(max2<prob):
                    max2 = prob
                    m2 = i
    if(max1>=0.8):
        flag = 0
        predicted_class = Class_names[np.argmax(prediction)]
        confidence = round(100 * (np.max(prediction)), 2)
    else:
        predicted_class = m1
        confidence = m2
    

    if flag==0:
        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }
    else:
        print('Please,Click the photo again!')
        return{
            'class': Class_names[predicted_class],
            'class': Class_names[confidence]
        }



if __name__ == "__main__":
    uvicorn.run(app,host = 'localhost',port=8000)

