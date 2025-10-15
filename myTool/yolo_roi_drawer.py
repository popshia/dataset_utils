#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI Picker — Draw a single ROI on one image and output YOLO-normalized coords.

Usage
-----
python roi_picker_single.py <image> [--out OUT.{txt|json}] [--precision 4] [--copy]

Controls
--------
Mouse:
  - Left-drag on image: draw a new ROI
  - Left-drag inside ROI: move ROI (kept in-bounds)
  - Left-drag on handles: resize ROI
  - Middle-drag: pan view
  - Wheel: zoom (cursor-anchored)

Keyboard:
  - r : reset (clear ROI)
  - s : save to --out if provided (or stdout)
  - Enter: same as 's'
  - q / Esc : exit (if ROI exists, auto-save before exit when --out is given)

Output
------
- TXT: "x1,y1,x2,y2" in YOLO normalized [0..1]
- JSON: {"image": <abs path>, "roi": {x1,y1,x2,y2}, "yolo": {x1,y1,x2,y2}}

Notes
-----
- All editing happens in original image coordinates; the view is a 1920x1080 canvas.
- Minimum ROI size is enforced (MIN_W x MIN_H).
- Clipboard copy with --copy uses pyperclip (if installed) or Tkinter fallback.

(C) 2025
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

WIN = "ROI Picker"
DISP_W, DISP_H = 1920, 1080
COLOR_BOX = (0, 255, 0)
COLOR_SEL = (0, 0, 255)
COLOR_UI = (255, 255, 255)
TH = 2
HANDLE_SIZE = 10
HANDLE_HALF = HANDLE_SIZE // 2
MIN_W, MIN_H = 2, 2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _try_copy(text: str):
    # Best-effort clipboard copy
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        pass
    try:
        import tkinter as tk
        r = tk.Tk(); r.withdraw()
        r.clipboard_clear(); r.clipboard_append(text); r.update()
        r.destroy()
        return True
    except Exception:
        return False


class SingleROIPicker:
    def __init__(self, image_path: Path, out: Path | None, precision: int = 4, copy_to_cb: bool = False):
        self.img_path = image_path.resolve()
        self.out = out.resolve() if out else None
        self.prec = precision
        self.copy_to_cb = copy_to_cb

        self.orig = self._imread(str(self.img_path))
        if self.orig is None:
            raise SystemExit(f"[ERROR] 讀取失敗: {self.img_path}")
        h, w = self.orig.shape[:2]
        self.orig_hw = (h, w)

        # View state
        self.base_scale = min(DISP_W / w, DISP_H / h)
        self.zoom = 1.0
        self.pan = [0, 0]  # canvas px
        self.scale = self.base_scale * self.zoom
        self.offset = (0, 0)
        self.view_rect = (0, 0, 0, 0)
        self.canvas = None
        self.disp_img = None
        self.panning = False
        self.pan_start = (0, 0)
        self.pan_origin = (0, 0)
        self.ZOOM_MIN, self.ZOOM_MAX, self.ZOOM_STEP = 0.1, 10.0, 1.2

        # ROI state (single)
        self.roi = None  # dict{x1,y1,x2,y2} in original coords
        self.mode = "idle"  # idle/drawing/moving/resizing
        self.start_canvas = (0, 0)
        self.drag_start_orig = (0, 0)
        self.roi_start = None
        self.resize_handle = None  # tl,tr,br,bl,l,r,t,b

        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self._mouse_cb)
        self._recompose_view()
        self._render()

    # -------- I/O helpers --------
    def _imread(self, path):
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"[WARN] _imread 失敗: {path} ({e})")
            return None

    def _imwrite(self, path, img):
        ext = Path(path).suffix or ".png"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        try:
            buf.tofile(path)
            return True
        except Exception:
            return False

    # -------- coord transforms --------
    def _recompose_view(self):
        h, w = self.orig_hw
        self.scale = self.base_scale * self.zoom
        cur_w = max(1, int(w * self.scale))
        cur_h = max(1, int(h * self.scale))

        base_ox = (DISP_W - cur_w) // 2
        base_oy = (DISP_H - cur_h) // 2

        def clamp_tl(cur_size, disp_size, base_o, pan_val):
            tl_min = min(0, disp_size - cur_size)
            tl_max = max(0, disp_size - cur_size)
            tl = base_o + int(pan_val)
            tl = clamp(tl, tl_min, tl_max)
            return tl, tl - base_o

        tlx, nx = clamp_tl(cur_w, DISP_W, base_ox, self.pan[0])
        tly, ny = clamp_tl(cur_h, DISP_H, base_oy, self.pan[1])
        self.pan = [nx, ny]
        self.offset = (tlx, tly)
        self.view_rect = (tlx, tly, tlx + cur_w, tly + cur_h)

        canvas = np.zeros((DISP_H, DISP_W, 3), dtype=np.uint8)
        zimg = cv2.resize(self.orig, (cur_w, cur_h), interpolation=cv2.INTER_LINEAR)
        x1 = max(0, tlx); y1 = max(0, tly)
        x2 = min(DISP_W, tlx + cur_w); y2 = min(DISP_H, tly + cur_h)
        if x2 > x1 and y2 > y1:
            sx1, sy1 = x1 - tlx, y1 - tly
            sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
            canvas[y1:y2, x1:x2] = zimg[sy1:sy2, sx1:sx2]
        self.canvas = canvas
        self.disp_img = canvas.copy()

    def _orig2canvas(self, x, y):
        ox, oy = self.offset
        return int(x * self.scale) + ox, int(y * self.scale) + oy

    def _canvas2orig(self, x, y):
        ox, oy = self.offset
        return int((x - ox) / self.scale), int((y - oy) / self.scale)

    def _in_view(self, x, y):
        x1, y1, x2, y2 = self.view_rect
        return x1 <= x < x2 and y1 <= y < y2

    # -------- ROI helpers --------
    def _normalize_roi(self, r):
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        h, w = self.orig_hw
        x1 = clamp(x1, 0, w - 1); x2 = clamp(x2, 0, w - 1)
        y1 = clamp(y1, 0, h - 1); y2 = clamp(y2, 0, h - 1)
        if x2 - x1 < MIN_W: x2 = min(w - 1, x1 + MIN_W)
        if y2 - y1 < MIN_H: y2 = min(h - 1, y1 + MIN_H)
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    def _handle_boxes(self, r):
        p1 = self._orig2canvas(r["x1"], r["y1"])
        p2 = self._orig2canvas(r["x2"], r["y2"])
        x1c, y1c = p1; x2c, y2c = p2
        xm = (x1c + x2c) // 2; ym = (y1c + y2c) // 2

        def box(cx, cy):
            return (cx - HANDLE_HALF, cy - HANDLE_HALF, cx + HANDLE_HALF, cy + HANDLE_HALF)

        return {
            "tl": box(x1c, y1c), "tr": box(x2c, y1c), "br": box(x2c, y2c), "bl": box(x1c, y2c),
            "t": box(xm, y1c), "r": box(x2c, ym), "b": box(xm, y2c), "l": box(x1c, ym),
        }

    def _hit_handle(self, x, y):
        if not self.roi: return None
        for k, (x1, y1, x2, y2) in self._handle_boxes(self.roi).items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return k
        return None

    def _inside_roi(self, x, y):
        if not self.roi: return False
        x1c, y1c = self._orig2canvas(self.roi["x1"], self.roi["y1"])
        x2c, y2c = self._orig2canvas(self.roi["x2"], self.roi["y2"])
        return x1c <= x <= x2c and y1c <= y <= y2c

    # -------- mouse callback --------
    def _mouse_cb(self, event, x, y, flags, param=None):
        # Zoom (wheel)
        if event == cv2.EVENT_MOUSEWHEEL:
            if self.scale > 0:
                oxf = (x - self.offset[0]) / self.scale
                oyf = (y - self.offset[1]) / self.scale
            else:
                oxf = oyf = 0.0
            step = self.ZOOM_STEP if flags > 0 else (1.0 / self.ZOOM_STEP)
            new_zoom = clamp(self.zoom * step, self.ZOOM_MIN, self.ZOOM_MAX)
            if abs(new_zoom - self.zoom) > 1e-6:
                self.zoom = new_zoom
                new_scale = self.base_scale * self.zoom
                new_w = max(1, int(self.orig_hw[1] * new_scale))
                new_h = max(1, int(self.orig_hw[0] * new_scale))
                base_ox = (DISP_W - new_w) // 2
                base_oy = (DISP_H - new_h) // 2
                off_x_prime = x - oxf * new_scale
                off_y_prime = y - oyf * new_scale
                self.pan = [int(round(off_x_prime - base_ox)), int(round(off_y_prime - base_oy))]
                self._recompose_view(); self._render()
            return

        # Pan (middle)
        if event == cv2.EVENT_MBUTTONDOWN:
            self.panning = True
            self.pan_start = (x, y)
            self.pan_origin = (self.pan[0], self.pan[1])
            return
        elif event == cv2.EVENT_MOUSEMOVE and self.panning:
            dx = x - self.pan_start[0]; dy = y - self.pan_start[1]
            self.pan = [self.pan_origin[0] + dx, self.pan_origin[1] + dy]
            self._recompose_view(); self._render(); return
        elif event == cv2.EVENT_MBUTTONUP:
            self.panning = False; return

        # Left button interactions
        if event == cv2.EVENT_LBUTTONDOWN:
            # prefer resize if handle hit
            hkey = self._hit_handle(x, y)
            if hkey is not None:
                self.mode = "resizing"; self.resize_handle = hkey
                self.roi_start = dict(self.roi)
                return
            # move if inside current roi
            if self._inside_roi(x, y):
                self.mode = "moving"; self.drag_start_orig = self._canvas2orig(x, y)
                self.roi_start = dict(self.roi)
                return
            # draw new if in view
            if self._in_view(x, y):
                self.mode = "drawing"; self.start_canvas = (x, y)
                self._render(temp_rect=(self.start_canvas, (x, y)))
                return

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mode == "drawing":
                if self._in_view(x, y):
                    vx1, vy1, vx2, vy2 = self.view_rect
                    cx = clamp(x, vx1, vx2 - 1); cy = clamp(y, vy1, vy2 - 1)
                    self._render(temp_rect=(self.start_canvas, (cx, cy)))
            elif self.mode == "moving" and self.roi_start is not None:
                sx, sy = self.drag_start_orig
                cx, cy = self._canvas2orig(x, y)
                dx, dy = cx - sx, cy - sy
                r0 = self.roi_start
                h, w = self.orig_hw
                # clamp motion to keep ROI in bounds
                dx = clamp(dx, -r0["x1"], (w - 1) - r0["x2"]) 
                dy = clamp(dy, -r0["y1"], (h - 1) - r0["y2"]) 
                self.roi = {"x1": r0["x1"] + dx, "y1": r0["y1"] + dy,
                            "x2": r0["x2"] + dx, "y2": r0["y2"] + dy}
                self._render()
            elif self.mode == "resizing" and self.roi_start is not None:
                cx, cy = self._canvas2orig(x, y)
                r0 = self.roi_start
                x1, y1, x2, y2 = r0["x1"], r0["y1"], r0["x2"], r0["y2"]
                hdl = self.resize_handle
                if hdl == "tl":
                    nr = {"x1": cx, "y1": cy, "x2": x2, "y2": y2}
                elif hdl == "tr":
                    nr = {"x1": x1, "y1": cy, "x2": cx, "y2": y2}
                elif hdl == "br":
                    nr = {"x1": x1, "y1": y1, "x2": cx, "y2": cy}
                elif hdl == "bl":
                    nr = {"x1": cx, "y1": y1, "x2": x2, "y2": cy}
                elif hdl == "t":
                    nr = {"x1": x1, "y1": cy, "x2": x2, "y2": y2}
                elif hdl == "r":
                    nr = {"x1": x1, "y1": y1, "x2": cx, "y2": y2}
                elif hdl == "b":
                    nr = {"x1": x1, "y1": y1, "x2": x2, "y2": cy}
                elif hdl == "l":
                    nr = {"x1": cx, "y1": y1, "x2": x2, "y2": y2}
                else:
                    nr = r0
                self.roi = self._normalize_roi(nr)
                self._render()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.mode == "drawing":
                (sx, sy) = self.start_canvas
                ex, ey = x, y
                vx1, vy1, vx2, vy2 = self.view_rect
                ex = clamp(ex, vx1, vx2 - 1); ey = clamp(ey, vy1, vy2 - 1)
                x1o, y1o = self._canvas2orig(sx, sy)
                x2o, y2o = self._canvas2orig(ex, ey)
                nr = self._normalize_roi({"x1": x1o, "y1": y1o, "x2": x2o, "y2": y2o})
                if nr["x2"] > nr["x1"] and nr["y2"] > nr["y1"]:
                    self.roi = nr
                self.mode = "idle"; self._render()
            elif self.mode in ("moving", "resizing"):
                self.mode = "idle"; self.roi_start = None; self.resize_handle = None; self._render()

    # -------- render --------
    def _render(self, temp_rect=None):
        disp = self.disp_img.copy()
        # Draw ROI if exists
        if self.roi:
            p1 = self._orig2canvas(self.roi["x1"], self.roi["y1"])
            p2 = self._orig2canvas(self.roi["x2"], self.roi["y2"])
            cv2.rectangle(disp, p1, p2, COLOR_SEL, TH)
            # handles
            for (x1, y1, x2, y2) in self._handle_boxes(self.roi).values():
                cv2.rectangle(disp, (x1, y1), (x2, y2), COLOR_SEL, -1)

        # temp drawing
        if temp_rect is not None and self.mode == "drawing":
            (sx, sy), (ex, ey) = temp_rect
            cv2.rectangle(disp, (sx, sy), (ex, ey), COLOR_BOX, TH)

        # status bar
        h, w = self.orig_hw
        cv2.rectangle(disp, (0, 0), (DISP_W, 52), (40, 40, 40), -1)
        zoom_txt = f"ZOOM:{int(self.zoom*100)}%"
        roi_txt = "ROI: None"
        if self.roi:
            x1, y1, x2, y2 = self.roi["x1"], self.roi["y1"], self.roi["x2"], self.roi["y2"]
            nx1 = round(x1 / (w - 1), self.prec)
            ny1 = round(y1 / (h - 1), self.prec)
            nx2 = round(x2 / (w - 1), self.prec)
            ny2 = round(y2 / (h - 1), self.prec)
            roi_txt = f"ROI: {nx1},{ny1},{nx2},{ny2}"
        cv2.putText(disp, f"{self.img_path.name}  |  {zoom_txt}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_UI, 2)
        cv2.putText(disp, roi_txt, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_UI, 2)

        cv2.imshow(WIN, disp)

    # -------- export --------
    def _export(self, verbose=True):
        if not self.roi:
            if verbose:
                print("[WARN] 尚未建立 ROI，無可輸出。")
            return None
        h, w = self.orig_hw
        x1, y1, x2, y2 = self.roi["x1"], self.roi["y1"], self.roi["x2"], self.roi["y2"]
        nx1 = round(x1 / (w - 1), self.prec)
        ny1 = round(y1 / (h - 1), self.prec)
        nx2 = round(x2 / (w - 1), self.prec)
        ny2 = round(y2 / (h - 1), self.prec)
        txt = f"{nx1},{ny1},{nx2},{ny2}"

        if self.out:
            if self.out.suffix.lower() == ".json":
                payload = {
                    "image": str(self.img_path),
                    "roi": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                    "yolo": {"x1": nx1, "y1": ny1, "x2": nx2, "y2": ny2},
                }
                self.out.parent.mkdir(parents=True, exist_ok=True)
                self.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                if verbose:
                    print(f"[OK] 已寫入 JSON：{self.out}")
            else:
                self.out.parent.mkdir(parents=True, exist_ok=True)
                self.out.write_text(txt + "\n", encoding="utf-8")
                if verbose:
                    print(f"[OK] 已寫入 TXT：{self.out}  ->  {txt}")
        else:
            print(txt)

        if self.copy_to_cb:
            if _try_copy(txt):
                if verbose:
                    print("[OK] 已複製到剪貼簿：", txt)
            elif verbose:
                print("[WARN] 剪貼簿複製失敗（可安裝 pyperclip）。")
        return txt

    # -------- main loop --------
    def run(self):
        while True:
            k = cv2.waitKey(10) & 0xFF
            if k == ord('r'):
                self.roi = None; self._render()
            elif k == ord('s'):
                self._export()
            elif k in (10, 13):  # Enter
                self._export()
            elif k in (ord('q'), 27):  # q / Esc
                if self.out and self.roi:
                    self._export(verbose=False)
                break
        cv2.destroyAllWindows()


def parse_args():
    ap = argparse.ArgumentParser(description="單圖 ROI 繪製器（輸出 YOLO 歸一化座標）")
    ap.add_argument("image", type=Path, help="輸入影像路徑")
    ap.add_argument("-o", "--out", type=Path, default=None, help="輸出檔（.txt 或 .json）。省略則印到 stdout")
    ap.add_argument("--precision", type=int, default=4, help="小數點位數（YOLO 歸一化輸出）")
    ap.add_argument("--copy", action="store_true", help="輸出後複製到剪貼簿")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        picker = SingleROIPicker(args.image, args.out, args.precision, args.copy)
        picker.run()
    except KeyboardInterrupt:
        sys.exit(0)
