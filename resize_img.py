import argparse
from pathlib import Path

from alive_progress import alive_bar
from PIL import Image


def resize_images(input_folder, output_folder, new_size):
    # Ensure the output folder exists
    if not Path(output_folder).exists():
        Path(output_folder).mkdir()

    # Loop through all files in the input folder
    with alive_bar(len(os.listdir(input_folder))) as bar:
        for filename in os.listdir(input_folder):
            # Check if the file is an image
            if filename.endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG")
            ):
                img_path = os.path.join(input_folder, filename)
                with Image.open(img_path) as img:
                    # Resize the image
                    resized_img = img.resize(new_size)

                    # Save the resized image in the output folder
                    output_path = os.path.join(output_folder, filename)
                    resized_img.save(output_path)
                    bar.text(f"Resized and saved: {output_path}")
            bar()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="input directory")
    parser.add_argument("output_dir", help="output directory")
    parser.add_argument(
        "--size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="[train, test] image sizes",
    )
    args = parser.parse_args()

    resize_images(args.input_dir, args.output_dir, args.size)
