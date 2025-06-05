import argparse
from pathlib import Path

import cv2

VID_FORMATS = [
    ".mov",
    ".avi",
    ".mp4",
    ".mpg",
    ".mpeg",
    ".m4v",
    ".wmv",
    ".mkv",
]  # acceptable video suffixes


def main(args):
    # Create output directory if it doesn't exist
    input_dir = Path(args.input_dir)
    output_dir = Path(input_dir / "captured_frames/")
    output_dir.mkdir(exist_ok=True)

    # Get all video files in the input directory
    files = sorted(input_dir.glob("**/*"))
    videos = [file for file in files if file.suffix.lower() in VID_FORMATS]
    print(videos)

    if not videos:
        print("No video files found in the input directory.")
        exit()

    # Process Each Video File
    for video in videos:
        cap = cv2.VideoCapture(video.as_posix())

        if not cap.isOpened():
            print(f"Error opening video file: {video}")
            continue

        print(f"Playing: {video}")
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            frame_count += 1
            cv2.imshow(
                "Video Playback - Press 's' to Save, 'd' to Play next video, 'q' to Quit",
                frame,
            )

            key = cv2.waitKey(30) & 0xFF
            if key == ord("s"):
                frame_filename = f"{video.stem}_frame{frame_count:05}.jpg"
                frame_path = Path(output_dir / frame_filename)
                cv2.imwrite(frame_path.as_posix(), frame)
                print(f"Saved frame to {frame_path}")
                saved_count += 1
            elif key == ord("d"):
                print("Quitting current video.")
                break
            elif key == ord("q"):
                exit(0)

        cap.release()

    cv2.destroyAllWindows()
    print("All videos processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    args = parser.parse_args()
    main(args)
