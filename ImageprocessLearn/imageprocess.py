import streamlit as st
import numpy as np
from PIL import Image
import cv2

# Function to blend the selected area with neighboring pixels
def blend_with_neighbors(image_array, vertices):
    result = image_array.copy()
    
    # Create a mask for the polygon
    mask = np.zeros(image_array.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [vertices], 255)  # Fill the polygon on the mask

    # Get the region of interest (ROI)
    roi = cv2.bitwise_and(result, result, mask=mask)

    # Replace pixels in the ROI with the mean of their neighbors
    for r in range(result.shape[0]):
        for c in range(result.shape[1]):
            if mask[r, c] == 255:
                neighbors = []
                if r > 0: neighbors.append(result[r - 1, c])  # Above
                if r < result.shape[0] - 1: neighbors.append(result[r + 1, c])  # Below
                if c > 0: neighbors.append(result[r, c - 1])  # Left
                if c < result.shape[1] - 1: neighbors.append(result[r, c + 1])  # Right
                
                if neighbors:
                    result[r, c] = np.mean(neighbors)

    return result

# Streamlit app
st.title("Interactive Polygon Background Removal Tool")

# Step 1: Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert('RGB')  # Convert to RGB
    image_array = np.array(image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Step 2: Draw polygon on the image using OpenCV
    st.write("To draw a polygon, save the image after editing it in OpenCV.")
    
    # Create a temporary path for the image
    temp_image_path = "temp_image.png"
    cv2.imwrite(temp_image_path, image_array)

    # Use OpenCV to open the image in a new window for drawing
    st.write("Click on the image to select vertices for the polygon.")
    st.write("Press 'c' to continue after drawing.")
    
    # Open the image for polygon drawing
    img = cv2.imread(temp_image_path)
    clone = img.copy()
    points = []

    def draw_polygon(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            if len(points) > 1:
                cv2.line(img, points[-1], points[-2], (0, 255, 0), 2)
            cv2.imshow("Draw Polygon", img)

    cv2.namedWindow("Draw Polygon")
    cv2.setMouseCallback("Draw Polygon", draw_polygon)

    while True:
        cv2.imshow("Draw Polygon", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):  # Press 'c' to continue
            break

    cv2.destroyAllWindows()

    # Convert points to numpy array
    vertices = np.array(points, dtype=np.int32)

    # Step 3: Button to process the image
    if st.button("Process Image"):
        processed_image = blend_with_neighbors(image_array, vertices)
        st.image(processed_image, caption='Processed Image', use_column_width=True, clamp=True)

# Add description and instructions
st.write("Upload an image and use OpenCV to draw a polygon on the image.")
