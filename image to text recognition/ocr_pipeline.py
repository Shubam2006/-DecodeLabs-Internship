"""
================================================================================
 Project 4 : Image or Text Recognition (Basic)
 Track     : Path 1 - Optical Character Recognition (OCR)
 Engineer  : DecodeLabs AI Engineer Track - Batch 2026
================================================================================

GOAL
----
Ingest a raw image, run it through a pre-processing pipeline, extract
machine-readable text using a pre-trained OCR engine (pytesseract /
Google Tesseract), apply a confidence filter, and display the result
clearly.

This script follows the "Mission Parameters" from the project brief:

    Objective   : Engineer a Python script capable of ingesting raw visual
                  data and extracting accurate, machine-readable
                  intelligence.
    Toolkit     : pytesseract (OCR engine) + OpenCV (image pre-processing)
    Deliverable : A fully functioning recognition pipeline that proves the
                  machine can read text with validated confidence.

GATEKEEPER CHECKLIST (all 4 satisfied below)
----------------------------------------------
 [x] 1. Library Integration   -> pytesseract used cleanly, error-handled
 [x] 2. Pre-Processing        -> grayscale + blur + adaptive threshold
 [x] 3. Accuracy Benchmarking -> 80% confidence minimum enforced
 [x] 4. Visual Confirmation   -> annotated image + clean text output
================================================================================
"""

import sys
import os
import cv2
import pytesseract
from pytesseract import Output

# --------------------------------------------------------------------------
# WINDOWS FIX: pytesseract is just a wrapper - it needs to know where the
# actual Tesseract-OCR program is installed. On Windows it usually isn't
# added to PATH automatically, so we point to it directly here.
# Comment this block out on Mac/Linux where tesseract is normally on PATH.
# --------------------------------------------------------------------------
_default_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.name == "nt" and os.path.exists(_default_win_path):
    pytesseract.pytesseract.tesseract_cmd = _default_win_path


# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 80          # The Gatekeeper Rule: 80% is the minimum
THRESH_BLOCK_SIZE = 31             # adaptive threshold neighborhood size
THRESH_C = 10                      # constant subtracted from the mean
TESSERACT_PSM = 6                  # 6 = "Assume a single uniform block of text"
                                    # (good default for invoices/documents;
                                    #  see PSM table in the brief for other modes)


# --------------------------------------------------------------------------
# STEP 1 - PRE-PROCESSING  ("The Logic Skeleton")
# --------------------------------------------------------------------------
def preprocess_image(image_path: str):
    """
    Cleans raw visual data before it reaches the OCR engine.

    Pipeline (matches the brief's 3-step pre-processing diagram):
        1. Grayscale conversion -> collapses the 3D RGB matrix into a
           1D intensity matrix (removes distracting color data).
        2. Gaussian blur        -> smooths micro-imperfections / noise.
        3. Adaptive thresholding-> forces every pixel to a binary
           black/white decision so character contours are crisp,
           even under uneven lighting.
    """
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")

    # Step 1: Grayscale Conversion
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Step 2: Gaussian Blur (kills salt-and-pepper / shadow noise)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Adaptive Thresholding (local Otsu-style binary decision,
    # robust to uneven lighting across the page)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        THRESH_BLOCK_SIZE,
        THRESH_C,
    )

    return original, binary


# --------------------------------------------------------------------------
# STEP 2 - OCR EXTRACTION  ("Path 1: Optical Character Recognition")
# --------------------------------------------------------------------------
def run_ocr(processed_image):
    """
    Runs the cleaned image through pytesseract and returns Tesseract's
    raw structured output (text + per-word confidence scores).

    --psm controls Page Segmentation Mode (see brief's PSM tuning table):
        3  -> fully automatic (mixed layouts)
        6  -> single uniform block of text (book pages / invoices)
        7  -> single text line (number plates, headers)
        11 -> sparse, scattered text
    """
    config = f"--psm {TESSERACT_PSM}"
    data = pytesseract.image_to_data(
        processed_image, config=config, output_type=Output.DICT
    )
    return data


# --------------------------------------------------------------------------
# STEP 3 - CONFIDENCE FILTER  ("The 80% Threshold Gate")
# --------------------------------------------------------------------------
def filter_by_confidence(ocr_data, threshold=CONFIDENCE_THRESHOLD):
    """
    Implements the exact rule from the brief:

        if confidence >= 0.80:
            draw_box_and_label()
        else:
            drop_detection()

    Returns a list of dicts: {text, confidence, x, y, w, h}
    """
    results = []
    n_boxes = len(ocr_data["text"])

    for i in range(n_boxes):
        word = ocr_data["text"][i].strip()
        conf = int(float(ocr_data["conf"][i]))

        if word == "" or conf < 0:          # Tesseract uses -1 for "no text"
            continue

        if conf >= threshold:                # PASS the gate
            results.append({
                "text": word,
                "confidence": conf,
                "x": ocr_data["left"][i],
                "y": ocr_data["top"][i],
                "w": ocr_data["width"][i],
                "h": ocr_data["height"][i],
            })
        # else: silently dropped (drop_detection equivalent)

    return results


# --------------------------------------------------------------------------
# STEP 4 - VISUAL CONFIRMATION  ("Display the output clearly")
# --------------------------------------------------------------------------
def draw_annotations(original_image, results, output_path="output_annotated.png"):
    """
    Draws a bounding box + confidence label over every word that
    passed the confidence gate, then saves the annotated image.
    """
    annotated = original_image.copy()

    for r in results:
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        label = f'{r["text"]} ({r["confidence"]}%)'

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(
            annotated, label, (x, max(y - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1, cv2.LINE_AA
        )

    cv2.imwrite(output_path, annotated)
    return output_path


def print_report(results, threshold=CONFIDENCE_THRESHOLD):
    """Clean, human-readable console report of what the machine 'read'."""
    print("=" * 60)
    print(" PROJECT 4 - OCR RECOGNITION REPORT")
    print("=" * 60)
    print(f" Confidence gate     : >= {threshold}%")
    print(f" Words accepted      : {len(results)}")
    print("-" * 60)

    if not results:
        print(" No text passed the confidence threshold.")
    else:
        full_text = " ".join(r["text"] for r in results)
        print(f" Reconstructed text  : {full_text}")
        print("-" * 60)
        print(f" {'Word':<20}{'Confidence':<12}{'Position (x,y)'}")
        for r in results:
            pos = f'({r["x"]}, {r["y"]})'
            print(f' {r["text"]:<20}{r["confidence"]}%{"":<8}{pos}')

    print("=" * 60)


# --------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------
def main(image_path: str):
    print(f"\n[1/4] Ingesting raw image: {image_path}")
    original, processed = preprocess_image(image_path)

    print("[2/4] Running pytesseract OCR engine...")
    ocr_data = run_ocr(processed)

    print(f"[3/4] Applying {CONFIDENCE_THRESHOLD}% confidence gate...")
    results = filter_by_confidence(ocr_data, CONFIDENCE_THRESHOLD)

    print("[4/4] Generating visual confirmation...\n")
    out_path = draw_annotations(original, results)

    print_report(results)
    print(f"\nAnnotated output saved to: {out_path}")

    # Bonus: also save the intermediate pre-processed (binary) image so
    # you can SEE the pre-processing step working, like the brief shows.
    cv2.imwrite("preprocessed_binary.png", processed)
    print("Pre-processed (binary) image saved to: preprocessed_binary.png")


if __name__ == "__main__":
    # Usage: python3 ocr_pipeline.py <image_path>
    image_arg = sys.argv[1] if len(sys.argv) > 1 else "sample_invoice.png"
    main(image_arg)
