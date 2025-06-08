import cv2 as cv
import keras
import numpy as np
import streamlit as st

# Define CSS for light and dark modes
LIGHT_MODE_CSS = """
<style>
body {
    background-color: #ffffff !important;
    color: #000000 !important;
}
[data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}
[data-testid="stSidebar"] {
    background-color: #f0f0f0 !important;
    color: #000000 !important;
}
h1, h2, h3 {
    color: #2c3e50 !important;
}
p {
    color: #000000 !important;
}
.stAlert, .st-emotion-cache-1wivap2 {
    background-color: #fff !important;
    color: #000 !important;
}
</style>
"""

DARK_MODE_CSS = """
<style>
body {
    background-color: #1e1e1e !important;
    color: #ffffff !important;
}
[data-testid="stAppViewContainer"] {
    background-color: #1e1e1e !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] {
    background-color: #2e2e2e !important;
    color: #ffffff !important;
}
h1, h2, h3 {
    color: #ecf0f1 !important;
}
</style>
"""

# Add a sidebar toggle for dark mode
mode = st.sidebar.radio("Choose your theme:", ["Light Mode", "Dark Mode"])

# Apply CSS dynamically based on the selected mode
if mode == "Dark Mode":
    st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_MODE_CSS, unsafe_allow_html=True)

# Title and Description
st.title("🌱 Plant Disease Detection")
st.markdown(
    """
    ### Welcome to Plant Disease Detection App
    This tool uses cutting-edge **deep-learning techniques to identify plant diseases 
    from leaf images. It's trained on a dataset featuring different plant diseases**, 
    ensuring accurate and reliable predictions.
    """
)

st.info(
    """
    ⚠️ **Note:** Please upload clear leaf images of **Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, or Tomato**.
    Images of other plants may not yield accurate results.
    """
)

# Load the pre-trained model
model = keras.models.load_model('trained_model.h5')

# Define the list of class labels
label_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# File uploader for input images
uploaded_file = st.file_uploader("🌿 Upload a leaf image to detect disease:")
if uploaded_file is not None:
    # Read the uploaded image
    image_bytes = uploaded_file.read()
    img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
    
    # Preprocess the image
    normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)
    
    # Display the uploaded image
    st.image(image_bytes, caption="Uploaded Image", use_container_width=True)
    
    # Make a prediction
    predictions = model.predict(normalized_image)
    pred_index = np.argmax(predictions)
    confidence = predictions[0][pred_index] * 100
    
    # Display the result with error handling
    def format_label(label):
        # Remove plant name prefix and underscores, keep disease name readable
        if '___' in label:
            plant, disease = label.split('___', 1)
            disease = disease.replace('_', ' ')
            # Remove extra text in parentheses for clarity
            disease = disease.replace('(', '').replace(')', '')
            # Special handling for healthy
            if disease.lower() == 'healthy':
                return f"{plant} healthy"
            else:
                return f"{plant} {disease}"
        else:
            return label.replace('_', ' ')
    if pred_index < len(label_name):
        pretty_label = format_label(label_name[pred_index])
        if confidence >= 80:
            st.success(f"✅ **Prediction:** {pretty_label} (Confidence: {confidence:.2f}%)")
        else:
            st.warning("⚠️ The model is not confident in its prediction. Please try another image.")
    else:
        st.error("❌ Prediction index out of range. Please check your model and label list.")
else:
    st.info("Upload a leaf image to start detecting diseases!")
