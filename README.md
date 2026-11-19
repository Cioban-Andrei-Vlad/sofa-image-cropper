# 🛋️ Sofa Image Cropper

A Python script that batch-crops product photos of sofas to a clean 16:9 format. It works out on its own whether each photo is a studio shot on a white background or a photo in a room, and handles each one differently. For room photos it uses YOLOv8 object detection to find the sofa.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-111F68?style=for-the-badge&logo=yolo&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## ✨ Features

### 🤍 White-Background Photos
- Detected automatically by checking the corner pixels
- Trims the extra white space around the product
- Centers it on a white 16:9 canvas with a minimum padding on every side

### 🏠 Room Photos
- Uses **YOLOv8** (COCO class 57, *couch*) to find the sofa
- Crops a 16:9 frame centered on the sofa, with padding
- Falls back to the white-background method if no sofa is found

### 📂 Batch Processing
- Goes through a folder **and all its subfolders**
- Supports `.jpg`, `.jpeg` and `.png`
- Skips any file with `skip` in its name

---

## 📋 Prerequisites

- **Python 3.10+**

```bash
pip install -r requirements.txt
```

The YOLOv8 model (`yolov8n.pt`) downloads automatically the first time you run the script.

---

## 🚀 Getting Started

### 1. Set your image folder

Open `crop_images.py` and replace the placeholder at the top:

```python
ROOT_FOLDER = "ADD PATH HERE"  # e.g. "C:/Users/you/Pictures/sofas"
```

### 2. (Optional) Change the settings

| Setting | Default | What it does |
|---|---|---|
| `TARGET_RATIO` | `16 / 9` | Aspect ratio of the output images |
| `MIN_PADDING` | `70` | Minimum pixels of space around the sofa |
| `COUCH_CLASS_ID` | `57` | COCO class to look for (57 = couch) |

### 3. Run it

```bash
python crop_images.py
```

> ⚠️ **The script overwrites the original images.** Run it on a copy of your folder if you want to keep the originals.

---

## 👤 Author

**Andrei-Vlad Cioban**
