best_model = "best.pt"


import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile

st.title("Corn Pest Detection")

st.write("Choose an image source:")
option = st.radio("Select input method", ("Upload Image", "Capture from Camera"))

image_data = None

if option == "Upload Image":
	uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
	if uploaded_file is not None:
		image_data = uploaded_file.read()

elif option == "Capture from Camera":
	camera_image = st.camera_input("Take a picture")
	if camera_image is not None:
		image_data = camera_image.getvalue()

if image_data is not None:
	# Save image data to a temp file
	with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
		tmp_file.write(image_data)
		tmp_path = tmp_file.name

	# Load model
	model = YOLO(best_model)

	# Run inference
	results = model(tmp_path)
	result = results[0]
	annotated_img = result.plot()

	# Convert annotated image (numpy array) to PIL Image for display
	annotated_pil = Image.fromarray(annotated_img)

	st.image(annotated_pil, caption="Detection Result", use_column_width=True)