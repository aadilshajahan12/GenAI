import cv2
import streamlit as st
import numpy as np
from keras.models import load_model
from tensorflow.keras.applications.nasnet import preprocess_input
import ollama

# Load your trained model
model = load_model('leaf_model.keras')
categories = ['Healthy','Mosaic','RedRot','Rust','Yellow']

# Remedy function using Ollama
def get_remedy(leaf):
    prompt = f"""You are an agro specialist specialised in sugarcane leaf diseases.
    Provide remedies for {leaf} which include treatment suggestions, preventive measures,
    and better farming practices."""
    res = ollama.generate(model='llama3.2:1b', prompt=prompt)
    return res['response']

# Prediction function
def predict_leaf(img):
    og = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    og = cv2.resize(og, (331, 331))
    og = preprocess_input(og)
    og_reshape = og.reshape((1, 331, 331, 3))
    pred = model.predict(og_reshape)
    res = pred.argmax()
    return categories[res]

# Streamlit UI
st.title("🌱 Sugarcane Leaf Disease Detector")

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input")
    option = st.radio("Choose input type:", ["Upload Image", "Live Video"])

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            st.image(img, channels="BGR", caption="Uploaded Image")
            leaf_class = predict_leaf(img)
            with col2:
                st.header("Output")
                st.subheader(f"Prediction: {leaf_class}")
                st.write(get_remedy(leaf_class))

    elif option == "Live Video":
        st.write("Press 'Start' to capture live feed from webcam.")
        run = st.checkbox("Start")
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)

        if run:
            ret, frame = cap.read()
            if ret:
                FRAME_WINDOW.image(frame, channels="BGR")
                leaf_class = predict_leaf(frame)
                with col2:
                    st.header("Output")
                    st.subheader(f"Prediction: {leaf_class}")
                    st.write(get_remedy(leaf_class))
        cap.release()
