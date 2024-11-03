import streamlit as st
import cv2
import numpy as np
from PIL import Image
import logging
from datetime import datetime
import os

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up detailed logging
log_filename = f'logs/polygon_extractor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

class ImageHandler:
    def __init__(self, uploaded_file):
        logging.info("Initializing ImageHandler")
        try:
            # Convert uploaded file to numpy array
            image = Image.open(uploaded_file)
            self.original_image = np.array(image)
            logging.info(f"Image loaded successfully. Shape: {self.original_image.shape}")
            logging.debug(f"Image dtype: {self.original_image.dtype}")
        except Exception as e:
            logging.error(f"Error loading image: {str(e)}")
            raise
    
    def extract_polygon(self, points):
        logging.info("Starting polygon extraction")
        try:
            # Create mask
            mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)
            logging.debug(f"Mask created with shape: {mask.shape}")
            
            # Extract region
            extracted = cv2.bitwise_and(self.original_image, self.original_image, mask=mask)
            logging.debug(f"Region extracted with shape: {extracted.shape}")
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(points)
            logging.debug(f"Bounding rectangle: x={x}, y={y}, w={w}, h={h}")
            
            # Crop the region
            cropped = extracted[y:y+h, x:x+w]
            logging.info(f"Cropped region shape: {cropped.shape}")
            
            return cropped, mask[y:y+h, x:x+w]
            
        except Exception as e:
            logging.error(f"Error in polygon extraction: {str(e)}")
            raise

def display_matrix_info(matrix, name):
    """Helper function to display matrix information"""
    logging.info(f"Displaying {name} matrix information")
    st.write(f"{name} Shape:", matrix.shape)
    st.write(f"{name} Data Type:", matrix.dtype)
    
    # Display statistics for each channel
    for i in range(matrix.shape[2]):
        channel_data = matrix[:,:,i]
        st.write(f"Channel {i} statistics:")
        st.write(f"  Min: {channel_data.min()}")
        st.write(f"  Max: {channel_data.max()}")
        st.write(f"  Mean: {channel_data.mean():.2f}")
        st.write(f"  Std: {channel_data.std():.2f}")

def main():
    try:
        st.title("3D Tensor Polygon Extraction")
        logging.info("Application started")
        
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            logging.info(f"File uploaded: {uploaded_file.name}")
            
            # Create an ImageHandler instance
            image_handler = ImageHandler(uploaded_file)
            
            # Display original image
            st.image(image_handler.original_image, caption='Original Image', use_column_width=True)
            
            # Display original tensor information
            st.subheader("Original Image Tensor Information")
            display_matrix_info(image_handler.original_image, "Original Image")
            
            if st.checkbox("Show Full Original Tensor"):
                st.write(image_handler.original_image)

            # Polygon drawing
            if st.button("Draw Polygon"):
                logging.info("Starting polygon drawing interface")
                img_copy = image_handler.original_image.copy()
                points = []

                def mouse_callback(event, x, y, flags, param):
                    nonlocal points, img_copy
                    if event == cv2.EVENT_LBUTTONDOWN:
                        points.append((x, y))
                        cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)
                        logging.debug(f"Point added at coordinates: ({x}, {y})")

                        if len(points) > 1:
                            cv2.polylines(img_copy, [np.array(points)], 
                                        isClosed=False, color=(255, 0, 0), thickness=2)
                        
                        cv2.imshow("Draw Polygon (Press 'q' when done)", img_copy)

                cv2.namedWindow("Draw Polygon (Press 'q' when done)")
                cv2.setMouseCallback("Draw Polygon (Press 'q' when done)", mouse_callback)

                while True:
                    cv2.imshow("Draw Polygon (Press 'q' when done)", img_copy)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cv2.destroyAllWindows()
                logging.info(f"Polygon drawing completed with {len(points)} points")

                # Extract and display polygon region
                if len(points) > 2:
                    points_np = np.array(points, dtype=np.int32)
                    extracted_region, mask = image_handler.extract_polygon(points_np)
                    
                    # Display extracted region
                    st.subheader("Extracted Region")
                    st.image(extracted_region, caption='Extracted Region', use_column_width=True)
                    
                    # Display extracted tensor information
                    st.subheader("Extracted Region Tensor Information")
                    display_matrix_info(extracted_region, "Extracted Region")
                    
                    if st.checkbox("Show Full Extracted Tensor"):
                        st.write(extracted_region)
                        
                    # Display mask
                    st.subheader("Extraction Mask")
                    st.image(mask, caption='Extraction Mask', use_column_width=True)
                    
                else:
                    logging.warning("Invalid polygon: less than 3 points provided")
                    st.warning("Please create a valid polygon with at least 3 points.")

    except Exception as e:
        logging.error(f"Unexpected error in main: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()