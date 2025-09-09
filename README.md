# DJI-Thermal-Image-Analysis

A Python app for analyzing DJI R-JPEG thermal images. It calls DJI’s CLI (`dji_irp.exe`) to extract true temperature maps (float32 °C) from each image, then provides an interactive viewer to:

* Colorize and inspect temperatures
* Measure spots, boxes (with min / mean / max + min/max locations), and lines (with min / mean / max along the line)
* Zoom/pan, adjust contrast (`vmin`/`vmax`), copy measurement summaries
* Batch-export float32 TIFF and CSV per image

---

## Requirements

### Python packages

* numpy
* opencv-python
* Pillow
* tifffile

### External tools

* **DJI Thermal SDK CLI** (Not included in this repository.)
* **ExifTool** (Already included in this repository for convenience.)

---

## Installation

### Clone the repository

```bash
git clone https://github.com/emireroglu2k/DJI-Thermal-Image-Analysis.git
cd DJI-Thermal-Image-Analysis
```

### Set up a Python environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Provide the DJI Thermal SDK CLI (`dji_irp.exe`)

1. Download the DJI Thermal SDK from [DJI Downloads](https://www.dji.com/global/downloads/softwares/dji-thermal-sdk).
2. Copy the `dji_thermal_sdk` folder to the root directory **or** add the containing folder to your system `PATH`.

### ExifTool

`exiftool.exe` is already included in this repository, no installation required. Just make sure that `exiftool.exe` is in the root directory.
