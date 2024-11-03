import numpy as np
from PIL import Image
import cv2  # Make sure to import cv2

class ImageHandler:
    def __init__(self, image_path):
        self.original_image = self.load_image(image_path)
        self.image_vector = self.image_to_vector(self.original_image)

    def load_image(self, image_path):
        # Load the image using PIL and convert to NumPy array
        image = Image.open(image_path)
        return np.array(image)

    def image_to_vector(self, image_array):
        # Return the image array as it is (keeping its 2D shape)
        return image_array  # Keep the original shape (height, width, channels)

    def get_image_vector(self):
        return self.image_vector

    def extract_polygon(self, points):
        # Create a mask and extract the polygon area from the original image
        mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        result = cv2.bitwise_and(self.original_image, self.original_image, mask=mask)
        return result, self.image_to_vector(result)
