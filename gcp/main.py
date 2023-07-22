from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

IMAGE_SIZE = 224

model = None
interpreter = None
input_index = None
output_index = None

class_names = ["Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight","Potato___Early_blight","Potato___Late_blight","healthy"]
BUCKET_NAME = "meri_bucket" # Here you need to put the name of your GCP bucket


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/model_2.h5",
            "/tmp/model_2.h5",
        )
        model = tf.keras.models.load_model("/tmp/model_2.h5")

    image = request.files["file"]

    image = np.array(
        Image.open(image).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)) # image resizing
    )

    #image = image/255 # normalize the image in 0 to 1 range

    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)
    prediction = predictions[0]

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
        predicted_class = class_names[np.argmax(prediction)]
        confidence = round(100 * (np.max(prediction)), 2)
    else:
        predicted_class1 = m1
        predicted_class2 = m2

    if flag==0:
        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'click_again': False
        }
    else:
        print('Please,Click the photo again!')
        return{
            'class': [class_names[predicted_class1], class_names[predicted_class2]],
            'confidence': float(0),
            'click_again': True
        }