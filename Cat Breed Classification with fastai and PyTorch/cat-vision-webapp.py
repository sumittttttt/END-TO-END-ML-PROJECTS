import pathlib
from pathlib import Path

import streamlit as st
from fastai.vision.all import *
from fastai.vision.widgets import *

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

learn_inf = load_learner('resnet50.pkl')

st.header(' Welcome to Cat Vision!!🐈')
st.markdown('With the help of deep learning and resnet50 architecture we have created this Cat breed classification model where we can easily classify various breed of cats.')
st.subheader('We can classify this cat breeds!')
st.markdown('- Asian')
st.markdown('- Australian Mist')
st.markdown('- Bengal')
st.markdown('- British Longhair')
st.markdown('- Cyprus')
st.markdown('- Bombay')
st.markdown('- Japanese Bobtail')
st.markdown('- Russian Blue')
st.markdown('- Selkirk Rex')
st.markdown('- Turkish Vankedisi')

class Predict:
    def __init__(self, filename):
        self.learn_inference = load_learner(Path()/filename)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
            self.get_prediction()
    
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            return PILImage.create((uploaded_file))
        return None

    def display_output(self):
        st.image(self.img.to_thumb(500,500), caption='Uploaded Image')

    def get_prediction(self):

        if st.button('Classify'):
            pred, pred_idx, probs = self.learn_inference.predict(self.img)
            st.write(f'**Prediction**: {pred}')
            st.write(f'**Probability**: {probs[pred_idx]*100:.02f}%')
        else: 
            st.write(f'Click the button to classify') 

if __name__=='__main__':

    file_name='resnet50.pkl'
predictor = Predict(file_name)
