import streamlit as st
from PIL import Image
import subprocess
import shutil
import os

# Create a directory to save the uploaded images
test_img_folder = 'LR/*'
curr = 'LR/'
old = 'old/'
upload_dir = curr
os.makedirs(upload_dir, exist_ok=True)

# Ensuring that the user cannot press the download archive button without successful execution of the upscaling function
upscale_successful = False

# Directory to archive
directory_to_archive = 'results'
archive_name = 'archive.zip'

# Streamlit app
st.title("Image Super Resolution using GANs")

# File uploader widget
uploaded_files = st.file_uploader("Upload one or more images:", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Function to save uploaded images
def save_uploaded_images(uploaded_files):
    if uploaded_files:
        for image_file in uploaded_files:
            image = Image.open(image_file)
            image.save(os.path.join(upload_dir, image_file.name))
        st.success(f"Saved {len(uploaded_files)} image(s).")

# Save uploaded images
save_uploaded_images(uploaded_files)

option = st.selectbox(
    'Which device would you like to use to upscale the images?:',
    ('cpu', 'cuda'))

st.write('You selected:', option)

# Function to run the external script with arguments
def run_test():
    try:
        # Run the external script with the specified arguments
        subprocess.run(["python", "test.py", "--device", option], check=True, text=True, capture_output=True)
        st.write("Upscaling successful.")
        upscale_successful = True
    except subprocess.CalledProcessError as e:
        st.error(f"Error upscaling: {e.stderr}")

# Button to trigger the script execution
if st.button("Upscale!"):
    run_test()

# Button to create and download the archive with upscaled images
if st.button("Create and Download Archive"):
    try:
        if(not upscale_successful):
            # Create the archive in the working directory (where the Streamlit script is located)
            archive_path = os.path.join(os.getcwd(), archive_name)
            shutil.make_archive(archive_path, 'zip', directory_to_archive)

            # Provide a download button to the user with the correct MIME type
            with open(f"{archive_path}.zip", 'rb') as f:
                st.download_button(label=f"Download {archive_name}.zip", data=f, key=f"{archive_name}.zip", mime="application/zip")

                # List all files in the directory
                files = os.listdir(directory_to_archive)
                for file_name in files:
                    file_path = os.path.join(directory_to_archive, file_name)
                    if os.path.isfile(file_path):  # Check if it's a file (not a directory)
                        os.remove(file_path)

                # Listing all files in the test_img_folder
                files_to_move = os.listdir('./LR/')
                for file_name in files_to_move:
                    source_path = os.path.join(curr, file_name)
                    destination_path = os.path.join(old, file_name)
                    # Move the file from source to destination
                    shutil.move(source_path, destination_path)
                    print(f"Moved {file_name} to {old}")
    except Exception as e:
        st.error(f"Error creating the archive: {e}")