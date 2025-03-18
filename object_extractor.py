import os
import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load pre-trained MobileNet SSD model and config for object detection
model_path = "models/MobileNetSSD_deploy.caffemodel"
config_path = "models/MobileNetSSD_deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_path, model_path)


def calculate_output_size(rect, margin_ratio=1.5):
    """Calculate dimensions with a margin around the main object."""
    width, height = rect[2] - rect[0], rect[3] - rect[1]
    margin_width = int(width * margin_ratio)
    margin_height = int(height * margin_ratio)

    return margin_width, margin_height


def standardise_image(
    image, output_path, output_size=(200, 200), margin_ratio=1.5, rect=None
):
    """Standardize and save object image with specified margin and size."""
    if rect is None:
        logging.warning("No object rectangle provided.")
        return

    x, y, w, h = rect
    width, height = w - x, h - y

    # Calculate margin dimensions
    margin_width = int(width * margin_ratio)
    margin_height = int(height * margin_ratio)

    # x, y should represent the center of the detected object
    x_center = int(x + width / 2)
    y_center = int(y + height / 2)

    # Calculate crop boundaries with margin
    x_start = max(0, x_center - margin_width // 2)
    y_start = max(0, y_center - margin_height // 2)
    x_end = min(image.shape[1], x_center + margin_width // 2)
    y_end = min(image.shape[0], y_center + margin_height // 2)

    # Crop the image
    cropped_img = image[y_start:y_end, x_start:x_end]

    # Resize the cropped image to the desired output size
    resized_img = cv2.resize(cropped_img, output_size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, resized_img)


def extract_objects(
    image_path,
    output_folder,
    output_size=(200, 200),
    margin_ratio=1.5,
    confidence_threshold=0.2,
):
    """Extract main object from image and save processed files."""
    image = cv2.imread(image_path)

    if image is None:
        logging.error(f"Could not read image: {image_path}")
        return

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net.setInput(blob)
    detections = net.forward()

    original_basename, file_extension = os.path.splitext(os.path.basename(image_path))

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            output_path = os.path.join(
                output_folder, f"{original_basename}_{i}{file_extension}"
            )
            standardise_image(
                image,
                output_path,
                output_size,
                margin_ratio,
                (startX, startY, endX, endY),
            )


def main(
    input_folder,
    output_folder,
    output_size=(300, 300),
    margin_ratio=1.5,
    allowed_extensions=(".png", ".jpg", ".jpeg"),
    confidence_threshold=0.2,
):
    """Process images in the input folder to detect main objects and output standardised images."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(allowed_extensions):
            input_path = os.path.join(input_folder, filename)
            extract_objects(
                input_path,
                output_folder,
                output_size,
                margin_ratio,
                confidence_threshold,
            )


if __name__ == "__main__":
    main("data/input", "data/output", margin_ratio=1)
