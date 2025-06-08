# Plant Disease Detection

This project is a web application for detecting plant diseases from leaf images using a deep learning model. The app provides a simple interface for users to upload images and receive predictions about possible plant diseases.

## Live Demo
Try the application online: [https://plantdiseasecheck.streamlit.app/](https://plantdiseasecheck.streamlit.app/)

## Features
- Upload a leaf image and get an instant disease prediction
- Supports 38 different plant disease classes
- Light and dark mode UI
- User-friendly display of prediction results

## Setup
1. Clone this repository and navigate to the project directory.
2. Install the required dependencies:
   ```
   pip install -r requirement.txt
   ```
3. Ensure the trained model file (`trained_model.h5`) is present in the project directory.

## Usage
To run the web application, use:
```
streamlit run main.py
```
This will start a local server. Open the provided URL in your browser to use the app.

## Notes
- For best results, upload clear images of leaves from Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, or Tomato plants.
- Images of other plants may not yield accurate results.

## Project Structure
- `main.py` - Streamlit web app for plant disease detection
- `trained_model.h5` - Pre-trained Keras model
- `requirement.txt` - List of required Python packages

