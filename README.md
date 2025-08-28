Dofus Resource Clicker
⚠️ Disclaimer:
This tool is designed only for private/test servers for research, testing, or automation experiments. Use this on official servers at your own risk, as it violates game rules and can lead to bans.
Overview
This Python-based tool automates resource collection in Dofus. It uses template matching and OCR to detect resource nodes and optional numbers above the character, and can automatically click them.
The project features:
• Tiny always-on-top GUI overlay
• Status (Running / Paused)
• Clicked resources counter
• OCR total counter
• Elapsed time
• Buttons: Pause/Resume, Capture Template, Set OCR Region, Exit
• Template capture for resources (click a resource to create a template image)
• OCR region selection to automatically read numbers appearing above the character
• Randomized click intervals and optional map panning
• Logging to logs.txt for every click, OCR read, and template capture
• Fully configurable via config.json
Requirements
• Python 3.10+
• Tesseract OCR engine
• Arch Linux: sudo pacman -S tesseract tesseract-data-eng
• Windows: download from Tesseract Releases
• Linux (Debian/Ubuntu): sudo apt install tesseract-ocr
Usage
• Create and edit config.json (template included). Set "dry_run": true to test safely.
• Launch the script:
python resource_clicker_gui_ocr.py --dry-run 
• Use the GUI to:
• Capture templates by hovering over a resource and clicking the button.
• Set OCR region by click-dragging the area above your character.
• When ready, uncheck dry-run or remove --dry-run to start real clicks.
Notes
• Resolution: Optimized for 1280×720, but GUI allows custom OCR region selection.
• Logs are stored in logs.txt.
• Only works on private/test servers. Using this on official servers may result in bans.
