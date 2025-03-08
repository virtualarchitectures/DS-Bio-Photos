import os
import cv2
import dlib

# Load dlib's face detector
detector = dlib.get_frontal_face_detector()


def calculate_output_size(face, margin_ratio=1.5):
    # Calculate dimensions with a margin around the face
    face_width, face_height = face.width(), face.height()
    margin_width = int(face_width * margin_ratio)
    margin_height = int(face_height * margin_ratio)

    return margin_width, margin_height


def standardize_image(
    image, output_path, output_size=(200, 200), margin_ratio=1.5, face=None
):
    # If a face is not provided, detect the first face
    if face is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) == 0:
            print("No face detected.")
            return
        face = faces[0]

    # Calculate output size based on the face dimensions
    output_width, output_height = calculate_output_size(face, margin_ratio)

    # Define the region to crop (head and shoulders)
    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
    x_center = x + w // 2
    y_center = y + h // 2
    x_start = max(0, x_center - output_width // 2)
    y_start = max(0, y_center - output_height // 2)
    x_end = min(image.shape[1], x_center + output_width // 2)
    y_end = min(image.shape[0], y_center + output_height // 2)

    cropped_img = image[y_start:y_end, x_start:x_end]

    # Resize the image while maintaining the aspect ratio
    resized_img = cv2.resize(cropped_img, output_size, interpolation=cv2.INTER_AREA)

    # Save the processed image
    cv2.imwrite(output_path, resized_img)


def process_multiple_faces(
    image_path, output_folder, output_size=(200, 200), margin_ratio=1.5
):
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read image: {image_path}")
        return

    # Convert to grayscale (for face detection)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    if len(faces) == 0:
        print(f"No face detected in {image_path}")
        return

    # Process each detected face
    for idx, face in enumerate(faces):
        output_path = os.path.join(
            output_folder, f"face_{idx}_{os.path.basename(image_path)}"
        )
        standardize_image(image, output_path, output_size, margin_ratio, face)


# Example usage for multiple faces
input_folder = "data/input"
output_folder = "data/output"
output_size = (300, 300)
margin_ratio = 2

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each image to detect multiple faces
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        input_path = os.path.join(input_folder, filename)
        process_multiple_faces(input_path, output_folder, output_size, margin_ratio)
