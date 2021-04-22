import streamlit as st
from fastai.vision.all import *
from utils import download_from_gdrive

def load_learner_(path):
    return load_learner(path)

def load_img(path):
    image = Image.open(path)
    w, h = image.size
    dim = (500, int((h*500)/w))
    return image.resize(dim)

st.markdown("# PotHole_Image Classifier")
st.markdown("Upload an image and the classifier will tell you whether its Perfect road Or Road with Pothole.")

with st.spinner('Downloading model...'):
    download_from_gdrive(file_id='1bH6n5fCJMo7EjIIQyM0shDbkNdCCm0w0',  dest_path='./export.pkl')
learn = load_learner_('export.pkl')

file_bytes = st.file_uploader("Upload a file", type=("png", "jpg", "jpeg", "jfif"))
if file_bytes:
    img = load_img(file_bytes)
    st.image(img)
    
    submit = st.button('Predict!')
    if submit:
        pred, pred_idx, probs = learn.predict(PILImage(img))
        st.markdown(f'Prediction: **{pred}**; Probability: **{probs[pred_idx]:.04f}**')
