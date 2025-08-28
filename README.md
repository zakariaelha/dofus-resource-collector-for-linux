# Dofus Resource Clicker (Private Server Testing)

**⚠️ Disclaimer:**  
This tool is designed **only for private/test servers** for research, testing, or automation experiments. **Do not use this on official servers**, as it violates game rules and can lead to bans.

---

## Overview

This Python-based tool automates resource collection in **Dofus private servers**. It uses template matching and OCR to detect resource nodes and optional numbers above the character, and can automatically click them.

Features:

- **Tiny always-on-top GUI overlay**  
  - Status (Running / Paused)  
  - Clicked resources counter  
  - OCR total counter  
  - Elapsed time  
  - Buttons: Pause/Resume, Capture Template, Set OCR Region, Exit

- **Template capture** for resources (click a resource to create a template image)  
- **OCR region selection** to automatically read numbers appearing above the character  
- **Randomized click intervals and optional map panning**  
- **Logging to `logs.txt`** for every click, OCR read, and template capture  
- Fully **configurable** via `config.json`

---

## Requirements

- **Python 3.10+**  
- **Tesseract OCR engine**  
  - Arch Linux: `sudo pacman -S tesseract tesseract-data-eng`  
  - Debian/Ubuntu: `sudo apt install tesseract-ocr`  
  - macOS: `brew install tesseract`  
  - Windows: [Download Tesseract](https://github.com/tesseract-ocr/tesseract/releases) and set path in `config.json`

- Python libraries (install via `requirements.txt`):
```
mss==9.0.1
opencv-python==4.10.0.84
numpy==2.0.1
pyautogui==0.9.54
pynput==1.7.7
pillow==10.4.0
pytesseract==0.3.10
```

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dofus-resource-clicker.git
cd dofus-resource-clicker
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv rc_env
source rc_env/bin/activate  # Linux/macOS
rc_env\Scripts\activate     # Windows
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR engine for your OS (see Requirements above).

---

## Configuration

Create a `config.json` file in the same folder as the script:

```json
{
  "scan_region": null,
  "template_capture_size": 64,
  "match_threshold": 0.88,
  "click_interval_min_s": 0.3,
  "click_interval_max_s": 0.7,
  "recenter_after_clicks": 20,
  "pan_pixels": 250,
  "pan_pause_s": 0.6,
  "dry_run": true,
  "ocr_region": null,
  "tesseract_cmd": "/usr/bin/tesseract"
}
```

- Set `"dry_run": false` when ready to perform real clicks.  
- `"ocr_region"` will be set by the GUI when you drag the OCR box.  
- `"tesseract_cmd"` should point to your Tesseract binary.

---

## Usage

1. Launch the script in **dry-run mode** to test safely:
```bash
python resource_clicker_gui_ocr.py --dry-run
```

2. Use the GUI:
- **Capture Template:** Hover over a resource and click to save a template image.  
- **Set OCR Region:** Click-drag the area above your character where resource numbers appear.  

3. When ready, uncheck dry-run or remove `--dry-run` to perform real clicks.

---

## Logging

- Logs are stored in `logs.txt`.  
- Each entry records clicks, OCR reads, template captures, and runtime events.

---

## Notes

- Optimized for **1280×720 resolution**, but GUI allows custom OCR region selection.  
- Only use on **private/test servers**. Using on official servers can result in bans.

---

## License

This project is intended for **educational and private server testing purposes**. No license is provided for commercial use or public servers.


