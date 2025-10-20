import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

st.set_page_config(
    page_title="WebCIFAR10",
    page_icon="🖥",
    layout="centered",
    initial_sidebar_state="expanded"
    # menu_items={
    #     'Get Help': 'https://www.github.com/tezamarlevi',
    #     'Report a bug': "https://www.google.com/",
    #     'About': "Hi everybody, let me introduce myself, \
    #     I am Teza Marlevi Fajar and I am a junior data scientist \
    #     at Rockstar Automotive Company, on this \
    #     website I will make app for predict price. \
    #     This second time make website and i hope you enjoyed it"
    # }
)

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

@st.cache_resource
def load_cifar_model():
    model = load_model('cnn_cifar10_model.keras')  # Replace with your model path
    return model

# Load the model
model = load_cifar_model()

st.title("CIFAR-10 CNN Image Classifier")

# File uploader
uploaded_file = st.file_uploader("Choose a CIFAR-10 style image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess: Resize to 32x32, convert to RGB, normalize
    image = image.resize((32, 32))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_array = np.array(image).astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Predict
    predictions = model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
    
    # Display top 3 predictions
    top3_idx = np.argsort(predictions[0])[-3:][::-1]
    for i, idx in enumerate(top3_idx):
        st.write(f"{i+1}. {class_names[idx]}: {predictions[0][idx]*100:.2f}%")
else:
    st.info("Please upload an image to classify.")



# import streamlit as st
# import requests
# import pickle
# import io

# from io import StringIO
# from PIL import Image
# # import tensorflow
# from tensorflow.keras.models import load_model



# st.title("Web App: CIFAR - 10")
# # st.write("Hello, Streamlit!")
# st.set_page_config(
#     page_title="WebCIFAR10",
#     page_icon="🖥",
#     layout="centered",
#     initial_sidebar_state="expanded",
#     menu_items={
#         'Get Help': 'https://www.github.com/tezamarlevi',
#         'Report a bug': "https://www.google.com/",
#         'About': "Hi everybody, let me introduce myself, \
#         I am Teza Marlevi Fajar and I am a junior data scientist \
#         at Rockstar Automotive Company, on this \
#         website I will make app for predict price. \
#         This second time make website and i hope you enjoyed it"
#     }
# )


# st.header('Cyberbullying')
# st.write('Cyberbullying is bullying with the use of digital technologies. \
# It can take place on social media, messaging platforms, gaming platforms and \
# mobile phones. It is repeated behaviour, aimed at scaring, angering or shaming \
# those who are targeted.')

# st.write("Before making predictions, please enter your identity")

# st.text_input('Name')
# st.text_input('Email')
# st.text_input('Phone Number')
# st.write(" ")
# st.write(" ")
# st.write(" ")
# st.write("You can input your text, to checking that sentences include \
# bullying or not")
# tweet_text = st.text_input('Input Youre Tweet')


# uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
# if uploaded_file is not None:
#     # Open and display the uploaded image
#     image = Image.open(io.BytesIO(uploaded_file.read()))
#     st.image(image, caption=f"Uploaded {uploaded_file.name}", use_column_width=True)
#     st.write(f"File type: {uploaded_file.type}")
# else:
#     st.info("Please upload a JPEG, JPG, or PNG file.")

# @st.cache_resource # Use st.cache_resource for models to load once
# def load_my_model():
#     with open('cnn_cifar10_model.keras', 'rb') as f:
#         model = pickle.load(f) # or joblib.load('my_model.pkl')
#     return model

# model = load_my_model()

# loaded_model = load_model('/home/tezamarlevi/python-streamlit/cnn_cifar10_model.keras')

# # inference

# data = {'tweet_text': tweet_text }

# # URL = "http://127.0.0.1:5000/predict" # sebelum push backend
# URL = "https://cyberbullying-backend.herokuapp.com/predict" # setelah push backend

# # komunikasi
# r = requests.post(URL, json=data)
# res = r.json()

# if r.status_code == 200:
#     agree = st.checkbox("Let's Predict")
#     if agree:
#         st.title(res['result']['prediction_name'])
# elif r.status_code == 400:
#     st.title("ERROR")
#     st.write(res['message'])
# st.write('  ')      
# st.write('  ')
# st.write('  ')
# st.write('  ')
# st.write('  ')
# st.write('  ')
# slide = st.slider('Rate this website', 0, 5, 0)
# if slide == 0 :
#     st.caption('swipe to rate')
# else :
#     st.write('Thank You')

# st.write('  ')
# title = st.text_input('💬 Commenting app')
# if title :
#     st.write('Thank you for comment')