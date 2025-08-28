"""
Resource Clicker — GUI + OCR + Logger (Fixed f-strings)
------------------------------------------------------
This version fixes all unterminated f-strings and quote issues.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyautogui
from mss import mss
from PIL import Image, ImageTk

try:
    import cv2
except Exception:
    raise SystemExit("OpenCV (cv2) is required. Please install requirements.txt")

try:
    import pytesseract
except Exception:
    pytesseract = None  # we'll check later

from pynput import keyboard, mouse
import tkinter as tk

# ----------------------
# Setup
# ----------------------
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
pyautogui.FAILSAFE = True

BASE = Path.cwd()
TEMPLATES_DIR = BASE / "templates"
CONFIG_PATH = BASE / "config.json"
LOG_PATH = BASE / "logs.txt"
TEMPLATES_DIR.mkdir(exist_ok=True)

DEFAULT_CONFIG = {
    "scan_region": None,
    "template_capture_size": 64,
    "match_threshold": 0.88,
    "click_interval_min_s": 0.3,
    "click_interval_max_s": 0.7,
    "recenter_after_clicks": 20,
    "pan_pixels": 250,
    "pan_pause_s": 0.6,
    "dry_run": True,
    "ocr_region": None,
    "tesseract_cmd": None
}

# ----------------------
# Config & utils
# ----------------------

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        logging.info(f"Wrote default config to {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for k, v in DEFAULT_CONFIG.items():
        if k not in cfg:
            cfg[k] = v
    return cfg


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def log_event(entry: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {entry}\n")


def save_image_rgb(array: np.ndarray, path: Path) -> None:
    img = Image.fromarray(array)
    img.save(path)

# ----------------------
# Image / templates
# ----------------------
@dataclass
class Template:
    name: str
    path: Path
    threshold: float
    image: Optional[np.ndarray] = field(default=None, init=False)

    def load(self) -> None:
        if self.image is None:
            img = cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Failed to load template {self.path}")
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.image = img


def discover_templates(threshold: float) -> List[Template]:
    tpls: List[Template] = []
    for p in sorted(TEMPLATES_DIR.glob("*.png")):
        tpls.append(Template(name=p.stem, path=p, threshold=threshold))
    if not tpls:
        logging.warning("No templates found in ./templates/. Use Capture Template to create one.")
    return tpls

# ----------------------
# Screen capture
# ----------------------
@dataclass
class WindowRegion:
    x: int
    y: int
    w: int
    h: int

    @property
    def tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


class Screen:
    def __init__(self, region: Optional[WindowRegion] = None):
        self.sct = mss()
        self.region = region

    def grab_rgb(self) -> np.ndarray:
        if self.region:
            bbox = {"left": self.region.x, "top": self.region.y, "width": self.region.w, "height": self.region.h}
        else:
            monitor = self.sct.monitors[1]
            bbox = {"left": monitor["left"], "top": monitor["top"], "width": monitor["width"], "height": monitor["height"]}
        shot = self.sct.grab(bbox)
        img = np.array(shot)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

# ----------------------
# Matching
# ----------------------

def find_template_positions(haystack_rgb: np.ndarray, template_rgb: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    res = cv2.matchTemplate(haystack_rgb, template_rgb, cv2.TM_CCOEFF_NORMED)
    y_idxs, x_idxs = np.where(res >= threshold)
    w = template_rgb.shape[1]
    h = template_rgb.shape[0]
    coords = list(zip(x_idxs.tolist(), y_idxs.tolist()))
    picks: List[Tuple[int, int]] = []
    taken = [False] * len(coords)
    for i, (x, y) in enumerate(coords):
        if taken[i]:
            continue
        cx, cy = x + w // 2, y + h // 2
        picks.append((cx, cy))
        for j, (x2, y2) in enumerate(coords[i + 1 :], start=i + 1):
            if taken[j]:
                continue
            cx2, cy2 = x2 + w // 2, y2 + h // 2
            if abs(cx - cx2) < w and abs(cy - cy2) < h:
                taken[j] = True
    return picks

# ----------------------
# Resource Clicker
# ----------------------
@dataclass
class ResourceConfig:
    scan_region: Optional[WindowRegion]
    templates: List[Template]
    click_interval_min_s: float
    click_interval_max_s: float
    recenter_after_clicks: int
    pan_pixels: int
    pan_pause_s: float
    dry_run: bool
    match_threshold: float


class ResourceClicker:
    def __init__(self, cfg: ResourceConfig, gui_handle: "OverlayGUI"):
        self.cfg = cfg
        self.screen = Screen(cfg.scan_region)
        for t in self.cfg.templates:
            t.load()
        self.clicks_since_recenter = 0
        self.paused = False
        self.running = True
        self.clicked_count = 0
        self.ocr_count = 0
        self.gui = gui_handle

    def _click(self, x: int, y: int) -> None:
        if self.cfg.dry_run:
            logging.info(f"[DRY] Would click at ({x},{y})")
            log_event(f"DRY_CLICK at ({x},{y})")
            return
        pyautogui.moveTo(x, y, duration=float(np.random.uniform(0.04, 0.12)))
        pyautogui.click()
        log_event(f"CLICK at ({x},{y})")

    def _random_pause(self) -> None:
        time.sleep(float(np.random.uniform(self.cfg.click_interval_min_s, self.cfg.click_interval_max_s)))

    def _pan_map(self, direction: str) -> None:
        if self.cfg.dry_run:
            logging.info(f"[DRY] Would pan map {direction} by {self.cfg.pan_pixels} px")
            log_event(f"DRY_PAN {direction}")
            return
        if direction == "right":
            pyautogui.dragRel(self.cfg.pan_pixels, 0, duration=0.3)
        elif direction == "left":
            pyautogui.dragRel(-self.cfg.pan_pixels, 0, duration=0.3)
        elif direction == "up":
            pyautogui.dragRel(0, -self.cfg.pan_pixels, duration=0.3)
        elif direction == "down":
            pyautogui.dragRel(0, self.cfg.pan_pixels, duration=0.3)
        time.sleep(self.cfg.pan_pause_s)

    def _do_ocr(self) -> int:
        cfg = load_config()
        ocr_reg = cfg.get("ocr_region")
        if not ocr_reg:
            return 0
        left, top, w, h = ocr_reg
        shot = mss().grab({"left": left, "top": top, "width": w, "height": h})
        arr = np.array(shot)[:, :, :3]
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if pytesseract is None:
            logging.warning("pytesseract not installed; OCR skipped")
            return 0
        tcmd = cfg.get("tesseract_cmd")
        if tcmd:
            pytesseract.pytesseract.tesseract_cmd = tcmd
        txt = pytesseract.image_to_string(th, config="--psm 7 digits")
        nums = [int(s) for s in __extract_ints(txt)]
        return sum(nums) if nums else 0

    def run(self) -> None:
        logging.info("ResourceClicker started.")
        start = time.time()
        try:
            while self.running:
                if self.paused:
                    time.sleep(0.2)
                    continue
                frame = self.screen.grab_rgb()
                any_click = False
                for t in self.cfg.templates:
                    locs = find_template_positions(frame, t.image, t.threshold)
                    for (cx, cy) in locs:
                        if self.cfg.scan_region:
                            cx_abs = self.cfg.scan_region.x + cx
                            cy_abs = self.cfg.scan_region.y + cy
                        else:
                            cx_abs, cy_abs = cx, cy
                        self._click(cx_abs, cy_abs)
                        self.clicked_count += 1
                        log_event(f"Clicked #{self.clicked_count} at ({cx_abs},{cy_abs})")
                        ocr_val = self._do_ocr()
                        if ocr_val:
                            self.ocr_count += ocr_val
                            log_event(f"OCR read +{ocr_val}; ocr_total={self.ocr_count}")
                        if self.gui:
                            self.gui.update_counts(self.clicked_count, self.ocr_count)
                        any_click = True
                        self._random_pause()
                        self.clicks_since_recenter += 1
                        if self.clicks_since_recenter >= self.cfg.recenter_after_clicks:
                            for d in ("right", "down", "left", "up"):
                                self._pan_map(d)
                            self.clicks_since_recenter = 0
                if not any_click:
                    time.sleep(0.12)
        except Exception as e:
            logging.exception(f"ResourceClicker error: {e}")
        finally:
            elapsed = int(time.time() - start)
            log_event(f"Stopped. Elapsed {elapsed} s. Clicks: {self.clicked_count}, OCR: {self.ocr_count}")
            logging.info("ResourceClicker stopped.")

# ----------------------
# Helper functions
# ----------------------

def __extract_ints(s: str) -> List[str]:
    import re
    return re.findall(r"[0-9]+", s)
