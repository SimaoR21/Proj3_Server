# color.py
import cv2
import numpy as np

# HSV color ranges (OpenCV format)
HSV_RANGES = {
    'red':      [((0,120,70),(10,255,255)), ((170,120,70),(179,255,255))],
    'orange':   [((10,100,20),(25,255,255))],
    'yellow':   [((25,100,50),(35,255,255))],
    'green':    [((36,50,50),(85,255,255))],
    'cyan':     [((78,50,50),(100,255,255))],
    'blue':     [((100,50,50),(135,255,255))],
    'purple':   [((135,50,50),(160,255,255))],
    'pink':     [((160,50,50),(172,255,255))],
    'brown':    [((10,50,20),(30,200,150))],
    'white':    [((0,0,200),(179,40,255))],
    'black':    [((0,0,0),(179,255,50))],
    'gray':     [((0,0,50),(179,40,200))]
}


def crop_center(roi):
    """Crop central trunk region to avoid edges, legs, background."""
    h, w = roi.shape[:2]
    if h < 60 or w < 60:
        return roi
    return roi[int(0.25*h):int(0.85*h), int(0.1*w):int(0.9*w)]


def get_color_and_shade(roi):
    """
    Returns (color_shade_label, confidence)
    - Detects base color from HSV masks
    - Detects shade (bright, light, deep, dark)
    - Black/white/gray NEVER get shades
    - Applies strict confidence threshold
    """
    if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
        return "unknown", 0.0

    roi = crop_center(roi)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    total_pixels = hsv.size // 3
    best_color = "unknown"
    best_count = 0

    # Detect base color using masks
    for color, ranges in HSV_RANGES.items():
        combined = None
        for low, high in ranges:
            mask = cv2.inRange(hsv, np.array(low), np.array(high))
            combined = mask if combined is None else cv2.bitwise_or(combined, mask)
        count = int(combined.sum() / 255)

        if count > best_count:
            best_count = count
            best_color = color

    confidence = best_count / total_pixels

    # Strict threshold: only accept strong detections
    MIN_CONF = 0.18
    if confidence < MIN_CONF:
        return "unknown", confidence

    # Shades ONLY for real chromatic colors
    if best_color in ["black", "white", "gray"]:
        return best_color, confidence   # no shading possible

    # Shade detection
    avg_s = np.mean(s)
    avg_v = np.mean(v)
    shade = ""

    if avg_v > 190:
        shade = "bright"
    elif avg_v > 150 and avg_s < 90:
        shade = "light"
    elif avg_v < 80:
        shade = "dark"
    elif avg_s > 150 and avg_v < 140:
        shade = "deep"

    final_label = f"{shade} {best_color}".strip()

    return final_label, confidence


def get_coloradd_symbol(color_label):
    """
    Retorna o símbolo ColorADD em Unicode baseado na cor.
    Combina para cores secundárias e tons.
    """
    if not color_label or color_label == "unknown":
        return ''

    color_lower = color_label.lower().strip()

    # Símbolos base (conforme disseste)
    blue_sym = '▼'    # \u25BD
    yellow_sym = '/'   # traço
    red_sym = '▲'      # \u25B3

    symbol = ''

    if 'blue' in color_lower or 'cyan' in color_lower:
        symbol = blue_sym

    elif 'yellow' in color_lower:
        symbol = yellow_sym

    elif 'red' in color_lower or 'pink' in color_lower:
        symbol = red_sym

    elif 'green' in color_lower:
        # Green = blue + yellow
        symbol = blue_sym + yellow_sym

    elif 'orange' in color_lower:
        # Orange = red + yellow
        symbol = red_sym + yellow_sym

    elif 'purple' in color_lower or 'violet' in color_lower:
        # Purple = red + blue
        symbol = red_sym + blue_sym

    elif 'brown' in color_lower:
        # Brown = red + green = red + blue + yellow
        symbol = red_sym + blue_sym + yellow_sym

    elif 'black' in color_lower:
        symbol = '■'  # \u25A0 quadrado cheio

    elif 'white' in color_lower:
        symbol = '□'  # \u25A1 quadrado vazio

    elif 'gray' in color_lower:
        symbol = '■□'  # \u2592 padrão cinza

    # Tons: adiciona símbolo extra
    if 'light' in color_lower or 'bright' in color_lower:
        symbol += '□'  # claro

    if 'dark' in color_lower or 'deep' in color_lower:
        symbol += '■'  # escuro

    return symbol
