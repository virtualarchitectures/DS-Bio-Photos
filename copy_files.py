import os
import shutil


def copy_files(directory_path="data/input", output_path="data/output", num_copies=2):
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Iterate over all files in the input directory
    for filename in os.listdir(directory_path):
        # Construct full file path
        file_path = os.path.join(directory_path, filename)

        if os.path.isfile(file_path):
            # Get file name and extension
            name, extension = os.path.splitext(filename)

            # Create the specified number of copies
            for i in range(1, num_copies + 1):
                # Construct copy filename
                copy_filename = f"{name}_{i}{extension}"
                copy_filepath = os.path.join(output_path, copy_filename)

                # Copy file to new location
                shutil.copyfile(file_path, copy_filepath)
                print(f"Copied {file_path} to {copy_filepath}")


if __name__ == "__main__":
    # Modify the number of copies as needed
    copy_files(num_copies=20)
