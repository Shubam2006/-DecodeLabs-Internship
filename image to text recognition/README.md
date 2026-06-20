# Project 4 — Image to Text Recognition (OCR)

**Track:** DecodeLabs AI Engineer — Industrial Training Kit (Batch 2026)
**Path:** Path 1 — Optical Character Recognition (Basic)

A Python pipeline that takes a raw image, cleans it up, extracts readable
text using a pre-trained OCR engine, filters out low-confidence guesses,
and shows the result clearly — both as a report and as an annotated image.

---

## What this does

1. **Ingests** a raw image (photo, scan, invoice, etc.)
2. **Pre-processes** it — grayscale → Gaussian blur → adaptive thresholding
3. **Runs OCR** using `pytesseract` (Google's Tesseract engine)
4. **Filters results** — keeps only text with **≥ 80% confidence**
5. **Outputs**:
   - A console report (recognized text + confidence + position)
   - An annotated image with bounding boxes and confidence labels
   - The pre-processed (binary) image, so you can see the cleanup step

---

## Files

| File | Description |
|---|---|
| `ocr_pipeline.py` | Main script — run this |
| `sample_invoice.png` | Sample test image (synthetic, noisy invoice) |
| `output_annotated.png` | Example output — text boxes + confidence labels |
| `preprocessed_binary.png` | Example output — image after grayscale/blur/threshold |

---

## Requirements

- Python 3
- [Tesseract OCR engine](https://github.com/tesseract-ocr/tesseract) installed on the system
- Python packages:
  ```bash
  pip install opencv-python pytesseract --break-system-packages
  ```

> Tesseract itself is a system binary, not a pip package. On Ubuntu/Debian:
> ```bash
> sudo apt-get install tesseract-ocr
> ```

---

## How to run

```bash
python3 ocr_pipeline.py path/to/your_image.png
```

If no image path is given, it defaults to `sample_invoice.png`.

**Example:**
```bash
python3 ocr_pipeline.py sample_invoice.png
```

**Output:**
```
============================================================
 PROJECT 4 - OCR RECOGNITION REPORT
============================================================
 Confidence gate     : >= 80%
 Words accepted      : 11
------------------------------------------------------------
 Reconstructed text  : Store: DecodeLabs Mart Item: Wireless ...
------------------------------------------------------------
 Word                Confidence  Position (x,y)
 Store:              93%        (41, 144)
 DecodeLabs          91%        (124, 138)
 ...
============================================================

Annotated output saved to: output_annotated.png
Pre-processed (binary) image saved to: preprocessed_binary.png
```

---

## How it works (pipeline breakdown)

### 1. Pre-processing (`preprocess_image`)
| Step | Purpose |
|---|---|
| Grayscale conversion | Drops color channels, keeps just pixel intensity |
| Gaussian blur | Smooths noise, shadows, small artifacts |
| Adaptive thresholding | Forces every pixel to pure black/white for crisp character edges, even under uneven lighting |

### 2. OCR extraction (`run_ocr`)
Uses `pytesseract.image_to_data()` with `--psm 6` (assumes a single uniform
block of text — good for documents/invoices). Returns each detected word
along with its confidence score and bounding box position.

Other PSM modes you can try by changing `TESSERACT_PSM` in the config:
| Mode | Use case |
|---|---|
| `3` | Fully automatic (mixed layouts) |
| `6` | Single uniform block (documents, invoices) — **default here** |
| `7` | Single line (headers, number plates) |
| `11` | Sparse, scattered text |

### 3. Confidence filtering (`filter_by_confidence`)
The project's "Gatekeeper Rule" — anything below 80% confidence is dropped:
```python
if confidence >= 80:
    keep_it()
else:
    drop_it()
```

### 4. Visual confirmation (`draw_annotations`, `print_report`)
Draws a green bounding box + confidence label over every accepted word and
saves it as an image, plus prints a clean text report to the console.

---

## Configuration

All tunable values live at the top of `ocr_pipeline.py`:

```python
CONFIDENCE_THRESHOLD = 80     # minimum % confidence to accept a word
THRESH_BLOCK_SIZE = 31        # adaptive threshold neighborhood size
THRESH_C = 10                 # threshold constant
TESSERACT_PSM = 6             # page segmentation mode
```

---

## Project checklist (Gatekeeper Rule)

- [x] **Library Integration** — `pytesseract` used cleanly with error handling
- [x] **Pre-Processing Integrity** — grayscale + blur + adaptive threshold implemented
- [x] **Accuracy Benchmarking** — 80% minimum confidence enforced
- [x] **Visual Confirmation** — annotated bounding-box image + readable text report

---

## Notes

- `sample_invoice.png` is a synthetically generated test image (not a real
  invoice) used only to demonstrate the pipeline end-to-end.
- Low-confidence words are *intentionally* dropped, not corrected — that's
  the "drop_detection()" behavior described in the project brief.
