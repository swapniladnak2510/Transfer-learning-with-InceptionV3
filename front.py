import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
model=load_model('final.h5')
st.markdown("<h2 style='text-align: center;'>Detection of the pneumonia bacteria</h2>", unsafe_allow_html=True)
st.subheader('Input will be the front chest x-ray of suspected person')
st.markdown("<br><br>",unsafe_allow_html=True)
st.set_option('deprecation.showfileUploaderEncoding', False)
img=st.file_uploader('Drop or upload x-ray image here',types=['jpeg'])
st.markdown("<br><br>",unsafe_allow_html=True)
if (st.button('SUBMIT')) & (img is not None):
    img=Image.open(img)
    img=image.img_to_array(img)
    img=np.resize(img,(250,250,3))
    img=img/255
    img=np.expand_dims(img,axis=0)
    st.markdown(np.round(model.predict(img)))
