best_model = "best.pt"


import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile

st.title("Corn Pest Detection")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
	# Save uploaded file to a temp location
	with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
		tmp_file.write(uploaded_file.read())
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