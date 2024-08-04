from PIL import Image
import os


def convert_webp_to_png(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.endswith(".webp"):
            file_path = os.path.join(source_dir, filename)
            img = Image.open(file_path)
            target_filename = os.path.splitext(filename)[0] + ".png"
            img.save(os.path.join(target_dir, target_filename), "PNG")


# Example usage
source_directory = "./images/webps"
target_directory = "./images"
convert_webp_to_png(source_directory, target_directory)
