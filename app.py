import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to convert image to sketch
def convert_to_sketch(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted_gray_image = 255 - gray_image

    # Apply Gaussian blur to the inverted image
    blurred_image = cv2.GaussianBlur(inverted_gray_image, (21, 21), 0)

    # Invert the blurred image
    inverted_blurred_image = 255 - blurred_image

    # Create the sketch by dividing the grayscale image by the inverted blurred image
    sketch = cv2.divide(gray_image, inverted_blurred_image, scale=256.0)

    return sketch

# Streamlit app
def main():
    st.title("🎨 Image to Sketch Converter")
    st.write("Upload an image and convert it into a sketch!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Convert the image to a sketch
        sketch = convert_to_sketch(image_array)

        # Display the sketch
        st.image(sketch, caption="Sketch", use_column_width=True)

        # Add a download button for the sketch
        sketch_image = Image.fromarray(sketch)
        st.download_button(
            label="Download Sketch",
            data=sketch_image.tobytes(),
            file_name="sketch.png",
            mime="image/png",
        )

if __name__ == "__main__":
    main()
