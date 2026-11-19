import os
import sys
import glob
from PIL import Image, ImageChops
from ultralytics import YOLO

# --- CONFIGURATION ---
ROOT_FOLDER = "ADD PATH HERE"  # Folder with your images (subfolders are included)
TARGET_RATIO = 16 / 9
COUCH_CLASS_ID = 57  # COCO dataset ID for 'couch/sofa'
MIN_PADDING = 70  # Minimum pixels of whitespace on any side

if ROOT_FOLDER == "ADD PATH HERE" or not os.path.isdir(ROOT_FOLDER):
    print("Set ROOT_FOLDER at the top of crop_images.py to your image folder.")
    sys.exit(1)

# Load the AI model used for photos with a room background
print("Loading YOLO AI model...")
model = YOLO('yolov8n.pt')


def is_white_background(img):
    """Detects if the image has a white background by checking corner pixels."""
    img_gray = img.convert('L')
    pixels = img_gray.load()
    width, height = img.size

    corners = [
        pixels[0, 0], pixels[width - 1, 0],
        pixels[0, height - 1], pixels[width - 1, height - 1]
    ]
    return sum(corners) / 4 > 240


def trim_and_center_white_bg(img):
    """Removes excess whitespace, adds MIN_PADDING, and centers in a 16:9 canvas."""
    bg = Image.new(img.mode, img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()

    if not bbox:
        return img

    cropped_img = img.crop(bbox)
    content_width, content_height = cropped_img.size

    min_padded_width = content_width + (MIN_PADDING * 2)
    min_padded_height = content_height + (MIN_PADDING * 2)

    if (min_padded_width / min_padded_height) < TARGET_RATIO:
        final_height = min_padded_height
        final_width = int(final_height * TARGET_RATIO)
    else:
        final_width = min_padded_width
        final_height = int(final_width / TARGET_RATIO)

    final_img = Image.new('RGB', (final_width, final_height), (255, 255, 255))

    paste_x = (final_width - content_width) // 2
    paste_y = (final_height - content_height) // 2
    final_img.paste(cropped_img, (paste_x, paste_y))

    return final_img


def process_room_bg(img, img_path):
    """Room background: finds the sofa using AI and crops around it."""
    width, height = img.size

    results = model(img_path, verbose=False)
    couches = [box for box in results[0].boxes if int(box.cls) == COUCH_CLASS_ID]

    if not couches:
        return trim_and_center_white_bg(img)

    best_couch = max(couches, key=lambda x: x.conf)
    x1, y1, x2, y2 = map(int, best_couch.xyxy[0])

    sofa_width = x2 - x1
    sofa_height = y2 - y1
    center_x = x1 + sofa_width // 2
    center_y = y1 + sofa_height // 2

    req_width = max(sofa_width + (MIN_PADDING * 2), int((sofa_height + (MIN_PADDING * 2)) * TARGET_RATIO))
    req_height = int(req_width / TARGET_RATIO)

    crop_x1 = max(0, center_x - req_width // 2)
    crop_y1 = max(0, center_y - req_height // 2)
    crop_x2 = min(width, center_x + req_width // 2)
    crop_y2 = min(height, center_y + req_height // 2)

    return img.crop((crop_x1, crop_y1, crop_x2, crop_y2))


def main():
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(ROOT_FOLDER, '**', ext), recursive=True))

    print(f"Found {len(image_paths)} images. Starting overwrite process...")

    for img_path in image_paths:
        try:
            # --- NEW: Check filename for exclusion before opening ---
            base_name = os.path.basename(img_path)

            if 'skip' in base_name.lower():
                print(f"⏩ Skipped (Found 'skip' in filename): {base_name}")
                continue
            # --------------------------------------------------------

            with Image.open(img_path) as img:
                working_img = img.convert('RGB')
                orig_width, orig_height = working_img.size

                white_bg = is_white_background(working_img)

                if white_bg:
                    final_img = trim_and_center_white_bg(working_img)
                else:
                    final_img = process_room_bg(working_img, img_path)

            final_img.save(img_path, quality=90)
            print(f"✅ Overwritten: {base_name}")

        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")


if __name__ == "__main__":
    main()