import os
import json
import threading
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
from tifffile import imwrite
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ---------------- Config ----------------
DJI_IRP_REL = r"dji_thermal_sdk\utility\bin\windows\release_x64\dji_irp.exe"  # relative to this script
EXIFTOOL_CMD = "exiftool"  # will also look for exiftool.exe next to script

# ---------------- Tool helpers ----------------
def script_dir() -> Path:
    return Path(__file__).resolve().parent

def resolve_dji_irp() -> str:
    cand = script_dir() / DJI_IRP_REL
    if cand.exists():
        return str(cand)
    # fall back to PATH
    try:
        subprocess.run(["dji_irp.exe", "-h"], capture_output=True, text=True, check=False)
        return "dji_irp.exe"
    except Exception:
        pass
    raise FileNotFoundError(
        f"dji_irp.exe not found at '{cand}'. "
        "Place the DJI Thermal SDK CLI binaries there or add to PATH."
    )

def resolve_exiftool() -> str | None:
    try:
        subprocess.run([EXIFTOOL_CMD, "-ver"], capture_output=True, text=True, check=True)
        return EXIFTOOL_CMD
    except Exception:
        pass
    local = script_dir() / "exiftool.exe"
    if local.exists():
        return str(local)
    # Allow app to run without EXIF copy
    return None

def exiftool_json(exiftool_cmd: str, img_path: str, keys: list[str]) -> dict:
    args = [exiftool_cmd, "-j"] + [f"-{k}" for k in keys] + [img_path]
    res = subprocess.run(args, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)
    return data[0] if data else {}

def get_image_dims(exiftool_cmd: str | None, img_path: str) -> tuple[int, int]:
    # If exiftool not available, fall back to JPEG decoding for dims
    if exiftool_cmd:
        try:
            d = exiftool_json(exiftool_cmd, img_path, ["ImageWidth", "ImageHeight"])
            return int(d.get("ImageWidth")), int(d.get("ImageHeight"))
        except Exception:
            pass
    # Fallback: use cv2 (reads preview size)
    im = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError("Failed to read image for dimensions.")
    h, w = im.shape[:2]
    return w, h

def read_measurement_params(exiftool_cmd: str | None, img_path: str) -> dict:
    if not exiftool_cmd:
        return {}
    try:
        d = exiftool_json(exiftool_cmd, img_path, ["UserComment"])
        uc = d.get("UserComment")
        if isinstance(uc, str) and uc.strip().startswith("{"):
            j = json.loads(uc)
            return j.get("measurement_params", {}) or {}
    except Exception:
        pass
    return {}

def measure_float32_celsius(dji_irp_cmd: str, exiftool_cmd: str | None, in_jpg: str) -> np.ndarray:
    """
    Call DJI CLI to output float32 °C raster and load it as (H,W).
    Handles common case: preview 1280x1024 while thermal is 640x512.
    """
    w_exif, h_exif = get_image_dims(exiftool_cmd, in_jpg)

    with tempfile.TemporaryDirectory() as td:
        out_bin = Path(td, "measure_float32.raw")
        cmd = [dji_irp_cmd, "-s", in_jpg, "-a", "measure", "-o", str(out_bin), "--measurefmt", "float32"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out_bin.exists():
            raise RuntimeError(f"DJI measure failed.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        buf = out_bin.read_bytes()

    arr = np.frombuffer(buf, dtype=np.float32)
    n = arr.size
    expected = w_exif * h_exif
    if n == expected:
        return arr.reshape((h_exif, w_exif))

    # Common: thermal is half each dimension
    if expected % n == 0:
        factor = expected // n
        if factor == 4 and (w_exif % 2 == 0) and (h_exif % 2 == 0):
            w_th, h_th = w_exif // 2, h_exif // 2
            if w_th * h_th == n:
                return arr.reshape((h_th, w_th))

    # Try known sizes
    for (w_th, h_th) in [(640, 512), (640, 480), (400, 300)]:
        if w_th * h_th == n:
            return arr.reshape((h_th, w_th))

    raise ValueError(f"Size mismatch: got {n}; preview {w_exif}x{h_exif}={expected}. Could not infer thermal shape.")

def colorize(temp_img: np.ndarray, vmin=None, vmax=None):
    if vmin is None: vmin = float(np.nanpercentile(temp_img, 1))
    if vmax is None: vmax = float(np.nanpercentile(temp_img, 99))
    if vmax <= vmin: vmax = vmin + 1e-3
    norm = np.clip((temp_img - vmin) / (vmax - vmin), 0, 1)
    img8 = (norm * 255).astype(np.uint8)
    vis = cv2.applyColorMap(img8, cv2.COLORMAP_INFERNO)
    return vis, (vmin, vmax)

def make_preview_image(vis_bgr: np.ndarray, max_w: int = 520, max_h: int = 520) -> ImageTk.PhotoImage:
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(vis_rgb)
    im.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(im)

def save_outputs(temp_img: np.ndarray, src_path: str, out_root: str, exiftool_cmd: str | None):
    """
    Save a float32 TIFF and a CSV under:
        <out_root>/TIFF/...
        <out_root>/CSV/...
    """
    out_root = Path(out_root)
    tiff_dir = out_root / "TIFF"
    csv_dir  = out_root / "CSV"
    tiff_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    base = Path(src_path).stem
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    tiff_path = tiff_dir / f"{base}_tempC_{stamp}.tif"
    csv_path  = csv_dir  / f"{base}_tempC_{stamp}.csv"

    # TIFF
    imwrite(str(tiff_path), temp_img.astype(np.float32))

    # CSV
    h, w = temp_img.shape
    ys, xs = np.indices((h, w))
    flat = np.column_stack([xs.ravel(), ys.ravel(), temp_img.ravel()])
    header = "x_pixel,y_pixel,temperature_c"
    np.savetxt(str(csv_path), flat, delimiter=",", header=header, comments="", fmt="%.0f,%.0f,%.6f")

    # Copy EXIF if available
    if exiftool_cmd:
        try:
            subprocess.run(
                [exiftool_cmd, "-overwrite_original", "-tagsfromfile", src_path, str(tiff_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            pass

    return str(tiff_path), str(csv_path)


class ThermalViewerPopup(tk.Toplevel):
    """
    Tkinter viewer supporting:
      - Tools: Spot, Box, Line, Pan
      - Clear Last / Clear All
      - Hover readout, pinned labels
      - vmin/vmax sliders + autorange/reset
      - Zoom (mouse wheel), Pan (Space drag or Pan tool)
      - Measurement log with stats
      - NEW: true image zoom (resamples image), and Box overlays (min/max points + temps + avg)
    """
    def __init__(self, parent, temp: np.ndarray, title="Thermal Viewer"):
        super().__init__(parent)
        self.title(title)
        self.geometry("1200x800")
        self.minsize(900, 600)

        self.temp = temp
        self.h, self.w = temp.shape
        vis_bgr, (vmin, vmax) = colorize(temp)
        self.vmin = tk.DoubleVar(value=vmin)
        self.vmax = tk.DoubleVar(value=vmax)

        # Render state
        self._img_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)  # colorized base at native resolution
        self._photo = None
        self._photo_scale = None   # cache: at what scale the current PhotoImage was rendered
        self._photo_w = None
        self._photo_h = None

        # Canvas transform
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self._drag_start = None
        self._space_pan = False

        # Tool + shapes state
        self.tool = tk.StringVar(value="spot")  # spot|box|line|pan
        self._drawing = False
        self._start_pt = None
        self._temp_shape_id = None
        self._hover_label_id = None
        self.measurements = []  # list of dicts

        # UI
        self._build_ui()
        self._bind_events()

        # Initial fit
        self.after(50, self.fit_to_window)

    # ---------- UI ----------
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Toolbar
        bar = ttk.Frame(self, padding=(8, 6))
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(99, weight=1)

        ttk.Label(bar, text="Tool:").grid(row=0, column=0, padx=(0,6))
        for i, (name, key) in enumerate([("Spot","spot"),("Box","box"),("Line","line"),("Pan","pan")], start=1):
            ttk.Radiobutton(bar, text=name, variable=self.tool, value=key).grid(row=0, column=i, padx=2)

        ttk.Separator(bar, orient="vertical").grid(row=0, column=6, sticky="ns", padx=8)

        ttk.Button(bar, text="Clear Last", command=self.clear_last).grid(row=0, column=7, padx=2)
        ttk.Button(bar, text="Clear All", command=self.clear_all).grid(row=0, column=8, padx=2)

        ttk.Separator(bar, orient="vertical").grid(row=0, column=9, sticky="ns", padx=8)

        ttk.Button(bar, text="Autorange", command=self.autorange).grid(row=0, column=10, padx=2)
        ttk.Button(bar, text="Reset", command=self.reset_contrast).grid(row=0, column=11, padx=2)

        ttk.Label(bar, text="vmin").grid(row=0, column=12, padx=(12, 2))
        self.s_min = ttk.Scale(bar, from_=-50, to=200, variable=self.vmin, command=self._on_contrast_change, length=140)
        self.s_min.grid(row=0, column=13, padx=2)

        ttk.Label(bar, text="vmax").grid(row=0, column=14, padx=(12, 2))
        self.s_max = ttk.Scale(bar, from_=-50, to=200, variable=self.vmax, command=self._on_contrast_change, length=140)
        self.s_max.grid(row=0, column=15, padx=2)

        help_txt = "Wheel: Zoom   Space+Drag/Pan tool: Pan   1/2/3/4: Spot/Box/Line/Pan   A: Autorange   R: Reset   C: Clear All"
        self.help_var = tk.StringVar(value=help_txt)
        ttk.Label(bar, textvariable=self.help_var, foreground="#666").grid(row=0, column=99, sticky="e")

        # Body split (canvas | side panel)
        body = ttk.Panedwindow(self, orient="horizontal")
        body.grid(row=1, column=0, sticky="nsew")

        # Canvas side
        left = ttk.Frame(body, padding=(6,6))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        body.add(left, weight=4)

        self.hover_var = tk.StringVar(value="(x,y)=–   T=– °C")
        ttk.Label(left, textvariable=self.hover_var).grid(row=0, column=0, sticky="w", pady=(0,4))

        self.canvas = tk.Canvas(left, bg="#111", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=1, column=0, sticky="nsew")

        # Sidebar
        right = ttk.Frame(body, padding=(6,6))
        right.rowconfigure(1, weight=1)
        body.add(right, weight=2)

        ttk.Label(right, text="Measurements", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        cols = ("type","details","stats")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=18)
        for c, w in zip(cols, (80, 200, 220)):
            self.tree.heading(c, text=c.capitalize())
            self.tree.column(c, width=w, anchor="w")
        self.tree.grid(row=1, column=0, sticky="nsew", pady=(6,6))

        btns = ttk.Frame(right)
        btns.grid(row=2, column=0, sticky="ew")
        ttk.Button(btns, text="Copy Selected", command=self.copy_selected).pack(side="left", padx=2)
        ttk.Button(btns, text="Copy All", command=self.copy_all).pack(side="left", padx=2)

        # Initial render
        self._ensure_photo()  # render at current scale (1.0)

    # ---------- Events ----------
    def _bind_events(self):
        self.canvas.bind("<Configure>", lambda e: self._on_canvas_resize())
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda e: self.hover_var.set("(x,y)=–   T=– °C"))

        self.canvas.bind("<ButtonPress-1>", self._on_btn1_press)
        self.canvas.bind("<B1-Motion>", self._on_btn1_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_btn1_release)

        # Space for temporary pan
        self.bind("<KeyPress-space>", lambda e: self._set_space_pan(True))
        self.bind("<KeyRelease-space>", lambda e: self._set_space_pan(False))

        # Mouse wheel zoom
        if self._is_windows():
            self.canvas.bind("<MouseWheel>", self._on_wheel)
        else:
            self.canvas.bind("<Button-4>", lambda e: self._zoom_at(1.1, e.x, e.y))
            self.canvas.bind("<Button-5>", lambda e: self._zoom_at(1/1.1, e.x, e.y))

        # Keyboard shortcuts
        self.bind("1", lambda e: self.tool.set("spot"))
        self.bind("2", lambda e: self.tool.set("box"))
        self.bind("3", lambda e: self.tool.set("line"))
        self.bind("4", lambda e: self.tool.set("pan"))
        self.bind("a", lambda e: self.autorange())
        self.bind("A", lambda e: self.autorange())
        self.bind("r", lambda e: self.reset_contrast())
        self.bind("R", lambda e: self.reset_contrast())
        self.bind("c", lambda e: self.clear_all())
        self.bind("C", lambda e: self.clear_all())

    # ---------- Utility ----------
    def _is_windows(self):
        return os.name == "nt"

    def _on_canvas_resize(self):
        # Keep image centered after window resize by refitting offsets
        self.fit_to_window()

    def fit_to_window(self):
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        scale_x = cw / self.w
        scale_y = ch / self.h
        self.scale = min(scale_x, scale_y)
        self.offset_x = (cw - self.w * self.scale) / 2
        self.offset_y = (ch - self.h * self.scale) / 2
        self._ensure_photo()
        self._redraw()

    def _ensure_photo(self):
        """Render a PhotoImage matching current contrast and scale."""
        # Rebuild the base colorized image according to vmin/vmax already set in _img_rgb
        # Now resample to (w*scale, h*scale)
        target_w = max(1, int(round(self.w * self.scale)))
        target_h = max(1, int(round(self.h * self.scale)))
        need_new = (
            self._photo is None or
            self._photo_scale is None or
            abs(self._photo_scale - self.scale) > 1e-6 or
            self._photo_w != target_w or
            self._photo_h != target_h
        )
        if need_new:
            img = Image.fromarray(self._img_rgb)
            # Bilinear keeps it smooth; use NEAREST if you prefer sharp pixels when zooming in
            img = img.resize((target_w, target_h), Image.BILINEAR)
            self._photo = ImageTk.PhotoImage(img)
            self._photo_scale = self.scale
            self._photo_w = target_w
            self._photo_h = target_h

    def _redraw(self):
        self.canvas.delete("all")
        self._ensure_photo()

        # Draw image at current offset
        self.canvas.create_image(self.offset_x, self.offset_y, image=self._photo, anchor="nw", tags=("image",))

        # Re-draw existing measurements overlays
        for m in self.measurements:
            self._draw_measurement(m)

        if self._hover_label_id is not None:
            self.canvas.tag_raise(self._hover_label_id)

    def _to_canvas(self, x, y):
        # image coords -> canvas coords
        return x * self.scale + self.offset_x, y * self.scale + self.offset_y

    def _to_image(self, cx, cy):
        # canvas coords -> image coords (int pixel)
        x = (cx - self.offset_x) / self.scale
        y = (cy - self.offset_y) / self.scale
        return int(np.clip(round(x), 0, self.w - 1)), int(np.clip(round(y), 0, self.h - 1))

    def _refresh_contrast_and_photo(self):
        """Re-colorize according to vmin/vmax, then rebuild PhotoImage at current scale."""
        vmin = float(self.vmin.get())
        vmax = float(self.vmax.get())
        if vmax <= vmin:
            vmax = vmin + 1e-3
            self.vmax.set(vmax)
        vis, _ = colorize(self.temp, vmin=vmin, vmax=vmax)
        self._img_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        self._photo = None
        self._photo_scale = None
        self._ensure_photo()
        self._redraw()

    def _on_contrast_change(self, _evt=None):
        self._refresh_contrast_and_photo()

    def autorange(self):
        vis, (vmin, vmax) = colorize(self.temp, None, None)
        self.vmin.set(vmin)
        self.vmax.set(vmax)
        self._img_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        self._photo = None
        self._photo_scale = None
        self._ensure_photo()
        self._redraw()

    def reset_contrast(self):
        vmin = float(np.nanmin(self.temp))
        vmax = float(np.nanmax(self.temp))
        self.vmin.set(vmin)
        self.vmax.set(vmax)
        self._refresh_contrast_and_photo()

    # ---------- Measurements ----------
    def clear_last(self):
        if not self.measurements:
            return
        self.measurements.pop()
        self._redraw()
        self._rebuild_tree()

    def clear_all(self):
        self.measurements.clear()
        self._redraw()
        self._rebuild_tree()

    def copy_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        rows = []
        for iid in sel:
            vals = self.tree.item(iid, "values")
            rows.append("\t".join(str(v) for v in vals))
        self.clipboard_clear()
        self.clipboard_append("\n".join(rows))
        self.update()

    def copy_all(self):
        rows = []
        for iid in self.tree.get_children():
            vals = self.tree.item(iid, "values")
            rows.append("\t".join(str(v) for v in vals))
        self.clipboard_clear()
        self.clipboard_append("\n".join(rows))
        self.update()

    def _add_measurement(self, m: dict):
        self.measurements.append(m)
        self._draw_measurement(m)
        self._append_tree(m)

    def _append_tree(self, m: dict):
        mtype = m["type"]
        if mtype == "spot":
            t = self.temp[m["y"], m["x"]]
            details = f"x={m['x']}, y={m['y']}"
            stats = f"T={t:.2f}°C"
        elif mtype == "box":
            x0, y0, x1, y1 = m["x0"], m["y0"], m["x1"], m["y1"]
            ylo, yhi = sorted([y0, y1])
            xlo, xhi = sorted([x0, x1])
            roi = self.temp[ylo:yhi+1, xlo:xhi+1]
            details = f"({x0},{y0})→({x1},{y1})  {roi.shape[1]}×{roi.shape[0]} px"
            stats = f"min={np.nanmin(roi):.2f}  mean={np.nanmean(roi):.2f}  max={np.nanmax(roi):.2f} °C"
        elif mtype == "line":
            x0, y0, x1, y1 = m["x0"], m["y0"], m["x1"], m["y1"]
            vals = self._sample_line_values(x0,y0,x1,y1)
            details = f"({x0},{y0})→({x1},{y1})  n={len(vals)}"
            stats = f"min={np.nanmin(vals):.2f}  mean={np.nanmean(vals):.2f}  max={np.nanmax(vals):.2f} °C"
        else:
            details = "-"
            stats = "-"
        self.tree.insert("", "end", values=(mtype, details, stats))

    def _rebuild_tree(self):
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        for m in self.measurements:
            self._append_tree(m)

    def _draw_cross(self, cx, cy, color="#00ffff", r=8, w=2):
        self.canvas.create_line(cx-r, cy, cx+r, cy, fill=color, width=w)
        self.canvas.create_line(cx, cy-r, cx, cy+r, fill=color, width=w)

    def _label_at(self, cx, cy, text, fill="#ffffff"):
        self.canvas.create_text(cx+1, cy+1, text=text, fill="#000000", anchor="nw", font=("Segoe UI", 9, "bold"))
        self.canvas.create_text(cx, cy, text=text, fill=fill, anchor="nw", font=("Segoe UI", 9, "bold"))

    def _draw_measurement(self, m: dict):
        if m["type"] == "spot":
            cx, cy = self._to_canvas(m["x"], m["y"])
            self._draw_cross(cx, cy, color="#00ffff")
            t = float(self.temp[m["y"], m["x"]])
            self._label_at(cx+10, cy-12, f"{t:.2f}°C", fill="#00ffff")

        elif m["type"] == "box":
            # rectangle
            x0, y0, x1, y1 = m["x0"], m["y0"], m["x1"], m["y1"]
            xlo, xhi = sorted([x0, x1])
            ylo, yhi = sorted([y0, y1])
            cx0, cy0 = self._to_canvas(xlo, ylo)
            cx1, cy1 = self._to_canvas(xhi, yhi)
            self.canvas.create_rectangle(cx0, cy0, cx1, cy1, outline="#ffcc00", width=2)

            # Stats within box
            roi = self.temp[ylo:yhi+1, xlo:xhi+1]
            if roi.size > 0:
                rmin = float(np.nanmin(roi))
                rmax = float(np.nanmax(roi))
                rmean = float(np.nanmean(roi))

                # locate min/max indices in ROI
                flat_min = int(np.nanargmin(roi))
                flat_max = int(np.nanargmax(roi))
                ry, rx = roi.shape
                min_iy, min_ix = divmod(flat_min, rx)
                max_iy, max_ix = divmod(flat_max, rx)
                min_x = xlo + max(0, min_ix)
                min_y = ylo + max(0, min_iy)
                max_x = xlo + max(0, max_ix)
                max_y = ylo + max(0, max_iy)

                # draw min/max cross + labels
                mcx, mcy = self._to_canvas(min_x, min_y)
                self._draw_cross(mcx, mcy, color="#00b7ff")
                self._label_at(mcx+10, mcy-12, f"min {rmin:.2f}°C", fill="#00b7ff")

                Mcx, Mcy = self._to_canvas(max_x, max_y)
                self._draw_cross(Mcx, Mcy, color="#ff5577")
                self._label_at(Mcx+10, Mcy-12, f"max {rmax:.2f}°C", fill="#ff5577")

                # avg label near box center
                ccx, ccy = self._to_canvas((xlo+xhi)//2, (ylo+yhi)//2)
                self._label_at(ccx+8, ccy+8, f"avg {rmean:.2f}°C", fill="#ffcc00")

            elif m["type"] == "line":
                x0, y0, x1, y1 = m["x0"], m["y0"], m["x1"], m["y1"]
                cx0, cy0 = self._to_canvas(x0, y0)
                cx1, cy1 = self._to_canvas(x1, y1)
                # draw the line
                self.canvas.create_line(cx0, cy0, cx1, cy1, fill="#00ff66", width=2)

                # sample values along line
                n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
                xs = np.linspace(x0, x1, n)
                ys = np.linspace(y0, y1, n)
                xs_i = np.clip(np.round(xs).astype(int), 0, self.w - 1)
                ys_i = np.clip(np.round(ys).astype(int), 0, self.h - 1)
                vals = self.temp[ys_i, xs_i]

                if vals.size > 0:
                    vmin = float(np.nanmin(vals))
                    vmax = float(np.nanmax(vals))
                    vmean = float(np.nanmean(vals))
                    i_min = int(np.nanargmin(vals))
                    i_max = int(np.nanargmax(vals))

                    # canvas positions for min/max
                    min_cx, min_cy = self._to_canvas(xs_i[i_min], ys_i[i_min])
                    max_cx, max_cy = self._to_canvas(xs_i[i_max], ys_i[i_max])

                    # draw crosses + labels
                    self._draw_cross(min_cx, min_cy, color="#00b7ff")
                    self._label_at(min_cx + 10, min_cy - 12, f"min {vmin:.2f}°C", fill="#00b7ff")

                    self._draw_cross(max_cx, max_cy, color="#ff5577")
                    self._label_at(max_cx + 10, max_cy - 12, f"max {vmax:.2f}°C", fill="#ff5577")

                    # avg near the geometric mid point of the line
                    midx = int(round((x0 + x1) / 2))
                    midy = int(round((y0 + y1) / 2))
                    ccx, ccy = self._to_canvas(midx, midy)
                    self._label_at(ccx + 8, ccy + 8, f"avg {vmean:.2f}°C", fill="#00ff66")

    def _sample_line_values(self, x0, y0, x1, y1):
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        xs = np.clip(np.round(xs).astype(int), 0, self.w-1)
        ys = np.clip(np.round(ys).astype(int), 0, self.h-1)
        return self.temp[ys, xs]

    # ---------- Mouse / keyboard handlers ----------
    def _on_motion(self, e):
        ix, iy = self._to_image(e.x, e.y)
        t = float(self.temp[iy, ix])
        self.hover_var.set(f"(x,y)=({ix},{iy})   T={t:.2f} °C   view[{self.vmin.get():.2f}, {self.vmax.get():.2f}] °C")

        if self._hover_label_id is not None:
            self.canvas.delete(self._hover_label_id)
            self._hover_label_id = None
        r = 4
        self._hover_label_id = self.canvas.create_oval(e.x-r, e.y-r, e.x+r, e.y+r, outline="#ffffff")

        if self._drawing and self._start_pt is not None:
            sx, sy = self._start_pt
            self._draw_temp_shape(sx, sy, e.x, e.y)

    def _on_btn1_press(self, e):
        if self.tool.get() == "pan" or self._space_pan:
            self._drag_start = (e.x, e.y, self.offset_x, self.offset_y)
            return
        self._drawing = True
        self._start_pt = (e.x, e.y)

    def _on_btn1_drag(self, e):
        if self.tool.get() == "pan" or self._space_pan:
            if self._drag_start:
                sx, sy, ox, oy = self._drag_start
                dx = e.x - sx
                dy = e.y - sy
                self.offset_x = ox + dx
                self.offset_y = oy + dy
                self._redraw()
            return

    def _on_btn1_release(self, e):
        if self.tool.get() == "pan" or self._space_pan:
            self._drag_start = None
            return

        if not self._drawing or self._start_pt is None:
            return
        sx, sy = self._start_pt
        ex, ey = e.x, e.y

        tool = self.tool.get()
        if tool == "spot":
            ix, iy = self._to_image(ex, ey)
            self._add_measurement({"type":"spot","x":ix,"y":iy})
        elif tool == "box":
            x0, y0 = self._to_image(sx, sy)
            x1, y1 = self._to_image(ex, ey)
            self._add_measurement({"type":"box","x0":x0,"y0":y0,"x1":x1,"y1":y1})
        elif tool == "line":
            x0, y0 = self._to_image(sx, sy)
            x1, y1 = self._to_image(ex, ey)
            self._add_measurement({"type":"line","x0":x0,"y0":y0,"x1":x1,"y1":y1})

        if self._temp_shape_id is not None:
            self.canvas.delete(self._temp_shape_id)
            self._temp_shape_id = None
        self._drawing = False
        self._start_pt = None

    def _draw_temp_shape(self, sx, sy, ex, ey):
        if self._temp_shape_id is not None:
            self.canvas.delete(self._temp_shape_id)
            self._temp_shape_id = None
        tool = self.tool.get()
        if tool == "box":
            self._temp_shape_id = self.canvas.create_rectangle(sx, sy, ex, ey, outline="#ffcc00", dash=(4,2), width=2)
        elif tool == "line":
            self._temp_shape_id = self.canvas.create_line(sx, sy, ex, ey, fill="#00ff66", dash=(4,2), width=2)
        elif tool == "spot":
            r = 8
            self._temp_shape_id = self.canvas.create_oval(ex-r, ey-r, ex+r, ey+r, outline="#00ffff", width=2)

    def _set_space_pan(self, flag: bool):
        self._space_pan = flag
        self.canvas.config(cursor="fleur" if flag or self.tool.get()=="pan" else "crosshair")

    def _on_wheel(self, e):
        if e.delta > 0:
            self._zoom_at(1.1, e.x, e.y)
        elif e.delta < 0:
            self._zoom_at(1/1.1, e.x, e.y)

    def _zoom_at(self, factor, cx, cy):
        old_scale = self.scale
        new_scale = np.clip(old_scale * factor, 0.2, 20.0)
        if new_scale == old_scale:
            return
        # Keep cursor pointing to same image pixel after zoom
        ix, iy = self._to_image(cx, cy)
        nx, ny = ix * new_scale + self.offset_x, iy * new_scale + self.offset_y
        self.offset_x += (cx - nx)
        self.offset_y += (cy - ny)
        self.scale = new_scale
        # Re-render PhotoImage at new scale and redraw
        self._ensure_photo()
        self._redraw()


# ---------------- Tk UI ----------------
class ThermalUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DJI R-JPEG Thermal Analyzer")
        self.geometry("980x640")
        self.minsize(900, 580)

        # ttk styling
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Toolbar.TFrame", padding=6)
        style.configure("Info.TLabel", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 11))

        # tools
        try:
            self.dji_irp = resolve_dji_irp()
        except Exception as e:
            messagebox.showerror("DJI CLI missing", str(e))
            self.dji_irp = None

        self.exiftool = resolve_exiftool()

        # state
        self.current_folder: Path | None = None
        self.files: list[Path] = []
        self.index: int = -1
        self.cache: dict[str, np.ndarray] = {}  # full path -> temp map
        self.preview_cache: dict[str, ImageTk.PhotoImage] = {}

        # build UI
        self._build_toolbar()
        self._build_body()
        self._bind_keys()

    # ---------- UI construction ----------
    def _build_toolbar(self):
        bar = ttk.Frame(self, style="Toolbar.TFrame")
        bar.pack(side="top", fill="x")

        ttk.Button(bar, text="Open Folder…", command=self.on_open_folder).pack(side="left", padx=4)
        ttk.Button(bar, text="View", command=self.on_view).pack(side="left", padx=4)
        ttk.Button(bar, text="Save Current…", command=self.on_save_current).pack(side="left", padx=4)
        ttk.Button(bar, text="Save All…", command=self.on_save_all).pack(side="left", padx=4)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(bar, textvariable=self.status_var).pack(side="right")

    def _build_body(self):
        # Paned layout
        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # Left: file list
        left = ttk.Frame(paned, padding=(8, 8))
        ttk.Label(left, text="Files", style="Title.TLabel").pack(anchor="w")

        self.listbox = tk.Listbox(left, activestyle="dotbox", exportselection=False)
        self.listbox.pack(fill="both", expand=True, pady=(6, 8))
        self.listbox.bind("<<ListboxSelect>>", self.on_select_file)
        self.listbox.bind("<Double-1>", lambda e: self.on_view())

        nav = ttk.Frame(left)
        nav.pack(fill="x")
        ttk.Button(nav, text="◀ Prev", command=self.on_prev).pack(side="left", padx=2)
        ttk.Button(nav, text="Next ▶", command=self.on_next).pack(side="left", padx=2)

        paned.add(left, weight=1)

        # Right: preview + info
        right = ttk.Frame(paned, padding=(8, 8))
        paned.add(right, weight=3)

        ttk.Label(right, text="Preview", style="Title.TLabel").pack(anchor="w")
        self.preview_label = ttk.Label(right)
        self.preview_label.pack(fill="both", expand=True, pady=(6, 8))

        info = ttk.Frame(right)
        info.pack(fill="x")

        self.stats_var = tk.StringVar(value="Stats: –")
        self.meta_var = tk.StringVar(value="Capture params: –")

        ttk.Label(info, textvariable=self.stats_var, style="Info.TLabel").pack(anchor="w")
        ttk.Label(info, textvariable=self.meta_var, style="Info.TLabel").pack(anchor="w", pady=(4, 0))

    def _bind_keys(self):
        self.bind("<Up>",   lambda e: self.on_prev())
        self.bind("<Down>", lambda e: self.on_next())
        self.bind("<Return>", lambda e: self.on_view())

    # ---------- Actions ----------
    def on_open_folder(self):
        path = filedialog.askdirectory(title="Select folder containing DJI RJPEGs")
        if not path:
            return
        self.current_folder = Path(path)
        self._load_folder()

    def _load_folder(self):
        self.files = []
        self.index = -1
        self.cache.clear()
        self.preview_cache.clear()
        self.listbox.delete(0, "end")

        # Collect *_T.JPG / *_T.jpg
        patt = ["*_T.JPG", "*_T.jpg"]
        for pat in patt:
            self.files.extend(sorted(self.current_folder.glob(pat)))
        if not self.files:
            self.status("No *_T.JPG files found.")
            return

        for p in self.files:
            self.listbox.insert("end", p.name)
        self.index = 0
        self.listbox.select_set(0)
        self.listbox.activate(0)
        self.listbox.see(0)
        self._prepare_preview_async(self.files[0])

    def on_select_file(self, _evt=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        self.index = int(sel[0])
        p = self.files[self.index]
        self._prepare_preview_async(p)

    def on_prev(self):
        if not self.files:
            return
        self.index = (self.index - 1) % len(self.files)
        self.listbox.selection_clear(0, "end")
        self.listbox.select_set(self.index)
        self.listbox.activate(self.index)
        self.listbox.see(self.index)
        self._prepare_preview_async(self.files[self.index])

    def on_next(self):
        if not self.files:
            return
        self.index = (self.index + 1) % len(self.files)
        self.listbox.selection_clear(0, "end")
        self.listbox.select_set(self.index)
        self.listbox.activate(self.index)
        self.listbox.see(self.index)
        self._prepare_preview_async(self.files[self.index])

    def on_view(self):
        if self.index < 0 or not self.files:
            messagebox.showinfo("No file", "Open a folder and select a file.")
            return
        if not self.dji_irp:
            messagebox.showerror("DJI CLI missing", "dji_irp.exe not resolved.")
            return
        p = self.files[self.index]

        def run():
            try:
                temp = self.cache.get(str(p))
                if temp is None:
                    temp = measure_float32_celsius(self.dji_irp, self.exiftool, str(p))
                    self.cache[str(p)] = temp
                title = f"Thermal Viewer — {p.name}"
                # New: open the Tk tools viewer
                self.after(0, lambda: ThermalViewerPopup(self, temp, title=title))
            except Exception as e:
                messagebox.showerror("View error", str(e))
        threading.Thread(target=run, daemon=True).start()

    def on_save_current(self):
        if self.index < 0 or not self.files:
            messagebox.showinfo("No file", "Open a folder and select a file.")
            return
        if not self.dji_irp:
            messagebox.showerror("DJI CLI missing", "dji_irp.exe not resolved.")
            return

        p = self.files[self.index]
        out_root = filedialog.askdirectory(title="Choose output folder (CSV & TIFF subfolders will be created)")
        if not out_root:
            return

        def run():
            try:
                temp = self.cache.get(str(p))
                if temp is None:
                    temp = measure_float32_celsius(self.dji_irp, self.exiftool, str(p))
                    self.cache[str(p)] = temp
                tiff_path, csv_path = save_outputs(temp, str(p), out_root, self.exiftool)
                self.status(f"Saved: {Path(tiff_path).name}, {Path(csv_path).name}")
                messagebox.showinfo(
                    "Saved",
                    f"TIFF → {Path(out_root) / 'TIFF'}\nCSV  → {Path(out_root) / 'CSV'}"
                )
            except Exception as e:
                messagebox.showerror("Save error", str(e))
        threading.Thread(target=run, daemon=True).start()


    def on_save_all(self):
        if not self.files:
            messagebox.showinfo("No files", "Open a folder with *_T.JPG files first.")
            return
        if not self.dji_irp:
            messagebox.showerror("DJI CLI missing", "dji_irp.exe not resolved.")
            return

        out_root = filedialog.askdirectory(title="Choose output folder for all conversions (CSV & TIFF subfolders will be created)")
        if not out_root:
            return

        def run():
            ok, fail = 0, 0
            for p in self.files:
                try:
                    temp = self.cache.get(str(p))
                    if temp is None:
                        temp = measure_float32_celsius(self.dji_irp, self.exiftool, str(p))
                        self.cache[str(p)] = temp
                    save_outputs(temp, str(p), out_root, self.exiftool)
                    ok += 1
                    self.status(f"Saved {p.name}")
                except Exception as e:
                    self.status(f"[ERROR] {p.name}: {e}")
                    fail += 1
            messagebox.showinfo(
                "Done",
                f"Converted {ok} file(s). Failures: {fail}\n\n"
                f"TIFFs → {Path(out_root) / 'TIFF'}\n"
                f"CSVs  → {Path(out_root) / 'CSV'}"
            )
        threading.Thread(target=run, daemon=True).start()


    # ---------- Preview building ----------
    def _prepare_preview_async(self, path: Path):
        # if cached preview, show immediately
        key = str(path)
        if key in self.preview_cache:
            self.preview_label.config(image=self.preview_cache[key])
        else:
            self.preview_label.config(image="", text="Loading preview…")
        self.stats_var.set("Stats: computing…")
        self.meta_var.set("Capture params: reading…")

        def worker():
            try:
                # decode temperatures (cached if already done)
                temp = self.cache.get(key)
                if temp is None:
                    if not self.dji_irp:
                        raise RuntimeError("dji_irp.exe not resolved.")
                    temp = measure_float32_celsius(self.dji_irp, self.exiftool, str(path))
                    self.cache[key] = temp

                # stats
                tmin, tmax, tmean = float(np.nanmin(temp)), float(np.nanmax(temp)), float(np.nanmean(temp))
                self._set_stats(f"Stats: min={tmin:.2f}°C  max={tmax:.2f}°C  avg={tmean:.2f}°C  size={temp.shape[1]}×{temp.shape[0]}")

                # preview image
                vis, _ = colorize(temp)
                photo = make_preview_image(vis)
                self.preview_cache[key] = photo
                self._set_preview(photo)

                # measurement params
                mp = read_measurement_params(self.exiftool, str(path))
                if mp:
                    def _fmt(v):
                        try:
                            f = float(v)
                            return f"{f:.2f}"
                        except Exception:
                            return str(v)
                    short = ", ".join(f"{k}={_fmt(v)}" for k, v in mp.items())
                    self._set_meta("Capture params: " + short)
                else:
                    self._set_meta("Capture params: –")


            except Exception as e:
                self._set_preview(None, text=f"Preview error:\n{e}")
                self._set_stats("Stats: –")
                self._set_meta("Capture params: –")

        threading.Thread(target=worker, daemon=True).start()

    # ---------- UI helpers ----------
    def _set_preview(self, photo: ImageTk.PhotoImage | None, text: str = ""):
        if photo:
            self.preview_label.config(image=photo, text="")
            self.preview_label.image = photo  # prevent GC
        else:
            self.preview_label.config(image="", text=text)

    def _set_stats(self, text: str):
        self.stats_var.set(text)

    def _set_meta(self, text: str):
        self.meta_var.set(text)

    def status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

# ---------------- main ----------------
if __name__ == "__main__":
    app = ThermalUI()
    app.mainloop()
