import argparse
import os

from PIL import Image


def flip_imgs(folder_path):
    # Create a new folder to save the flipped images
    output_folder = os.path.join(folder_path, "flipped_images")
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all the files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            image_path = os.path.join(folder_path, filename)

            # Open the image
            with Image.open(image_path) as img:
                # Flip the image up-down
                flipped_ud = img.transpose(Image.FLIP_TOP_BOTTOM)
                flipped_ud_path = os.path.join(
                    output_folder,
                    f"{os.path.splitext(filename)[0]}_flipped_ud{os.path.splitext(filename)[1]}",
                )
                flipped_ud.save(flipped_ud_path)

                # Flip the image left-right
                flipped_lr = img.transpose(Image.FLIP_LEFT_RIGHT)
                flipped_lr_path = os.path.join(
                    output_folder,
                    f"{os.path.splitext(filename)[0]}_flipped_lr{os.path.splitext(filename)[1]}",
                )
                flipped_lr.save(flipped_lr_path)

                # Flip the image up-down and left-right
                flipped_ud = flipped_ud.transpose(Image.FLIP_LEFT_RIGHT)
                flipped_ud_path = os.path.join(
                    output_folder,
                    f"{os.path.splitext(filename)[0]}_flipped_ud_lf{os.path.splitext(filename)[1]}",
                )
                flipped_ud.save(flipped_ud_path)

    print("Flipping complete! Flipped images saved in:", output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    flip_imgs(args.dataset)
