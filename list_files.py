import os

# Define the directory path
directory_path = "data/output"
output_file_path = os.path.join(directory_path, "contents.txt")

try:
    # Get a list of all files in the specified directory
    filenames = os.listdir(directory_path)

    # Filter out directories, only keep files
    file_list = [
        f for f in filenames if os.path.isfile(os.path.join(directory_path, f))
    ]

    # Write the list of filenames to "contents.txt" without a header
    with open(output_file_path, "w") as output_file:
        for filename in file_list:
            output_file.write(filename + "\n")

    print(f"List of filenames has been written to {output_file_path}")

except FileNotFoundError:
    print(f"The directory '{directory_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
