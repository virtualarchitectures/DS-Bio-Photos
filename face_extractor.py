import os
import cv2
import dlib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load dlib's face detector
detector = dlib.get_frontal_face_detector()


def calculate_output_size(face, margin_ratio=1.5):
    """Calculate dimensions with a margin around the face."""
    face_width, face_height = face.width(), face.height()
    margin_width = int(face_width * margin_ratio)
    margin_height = int(face_height * margin_ratio)

    return margin_width, margin_height


def standardise_image(
    image, output_path, output_size=(200, 200), margin_ratio=1.5, face=None
):
    """Standardize and save face image with specified margin and size."""
    if face is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if not faces:
            logging.warning("No face detected.")
            return
        face = faces[0]

    output_width, output_height = calculate_output_size(face, margin_ratio)
    x, y, w, h = (face.left(), face.top(), face.width(), face.height())

    # Define crop area with boundaries
    x_center, y_center = x + w // 2, y + h // 2
    x_start = max(0, x_center - output_width // 2)
    y_start = max(0, y_center - output_height // 2)
    x_end = min(image.shape[1], x_center + output_width // 2)
    y_end = min(image.shape[0], y_center + output_height // 2)

    cropped_img = image[y_start:y_end, x_start:x_end]

    # Resize image and save
    resized_img = cv2.resize(cropped_img, output_size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, resized_img)


def extract_faces(image_path, output_folder, output_size=(200, 200), margin_ratio=1.5):
    """Extract faces from image and save processed files."""
    image = cv2.imread(image_path)

    if image is None:
        logging.error(f"Could not read image: {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if not faces:
        logging.info(f"No face detected in {image_path}")
        return

    original_basename, file_extension = os.path.splitext(os.path.basename(image_path))

    for idx, face in enumerate(faces):
        output_path = os.path.join(
            output_folder, f"{original_basename}_{idx}{file_extension}"
        )
        standardise_image(image, output_path, output_size, margin_ratio, face)


def main(
    input_folder,
    output_folder,
    output_size=(300, 300),
    margin_ratio=1.5,
    allowed_extensions=(".png", ".jpg", ".jpeg"),
):
    """Process images in the input folder to detect faces and output standardised headshots."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(allowed_extensions):
            input_path = os.path.join(input_folder, filename)
            extract_faces(input_path, output_folder, output_size, margin_ratio)


if __name__ == "__main__":
    main("data/input", "data/output", margin_ratio=1)
