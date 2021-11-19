import tensorflow as tf

# from keras.models import model_from_json
# json_file = open('bin_oralmodel.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
model = tf.keras.models.load_model('final_oralclassificationmodel')

# load weights into new model
# loaded_model.load_weights("bin_oralweights.h5")
print("Loaded model from disk")
model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

final_label=['Oral','NonOral']

import streamlit as st
st.write("""
         # Oral Vs Non-Oral prediction
         """
         )
st.write("This is a simple image classification web app to predict oralvsnonoral")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    size = (150,150)    
    image = ImageOps.fit(image_data, size)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(256,256))
    img = np.reshape(img,[1,256,256,3])
    prediction = model.predict(img)
    
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    st.write("This image most likely belongs to {} with a {:.2f} percent confidence.".format(final_label[np.argmax(prediction)], 100 * np.max(prediction)))
    
#     if np.argmax(prediction) == 0:
#         st.write("It is a paper!")
#     else: 
#         st.write("It is a rock!")
#     st.text("Probability (0: Oral, 1: NonOral")
#     st.write(prediction)