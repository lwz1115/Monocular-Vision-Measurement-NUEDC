# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import json
from collections import namedtuple

Box = namedtuple("Box", ["x", "y", "w", "h"])

# ---------------- 基本参数 ----------------
FRAME_WIDTH = 1280
FRAME_HEIGHT = 960

MIN_CONTOUR_AREA = 80
SHOW_INTERNAL_DEBUG = False  # 可选调试窗口

REAL_HEIGHT_CM = 26.0
REAL_WIDTH_CM  = 17.0
RECT_RATIO = REAL_HEIGHT_CM / REAL_WIDTH_CM  # 26/17 ≈ 1.529
RATIO_TOL = 0.4   # 放宽比例容差（旧逻辑保留，供其他地方使用）

PARAM_FILE = "./ref_params.json"

# ---------------- 参数保存 ----------------
def save_ref_params(distance_ref_cm, ref_metric, metric_mode):
    params = {
        "distance_ref_cm": distance_ref_cm,
        "ref_metric": ref_metric,
        "metric_mode": metric_mode,
    }
    with open(PARAM_FILE, "w", encoding="utf-8") as f:
        json.dump(params, f)

def load_ref_params():
    if not os.path.exists(PARAM_FILE):
        return None
    try:
        with open(PARAM_FILE, "r", encoding="utf-8") as f:
            params = json.load(f)
        return params
    except Exception as e:
        print("读取参考参数失败：", e)
        return None

# ---------------- 图像增强 ----------------
def preprocess(gray):
    # 提升亮度和对比度
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
    # CLAHE 局部对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # 锐化增强远距离边缘
    sharp_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    gray = cv2.filter2D(gray, -1, sharp_kernel)
    return gray

# ---------------- 自动 Canny 边缘 ----------------
def find_contours(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = preprocess(gray)
    median = np.median(gray)
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(gray, lower, upper)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return edges, cnts

# ---------------- 工具函数 ----------------
def circularity(c):
    a = cv2.contourArea(c)
    p = cv2.arcLength(c, True)
    if p == 0: 
        return 0.0
    return 4.0*np.pi*a/(p*p)

# ---------- 四边形直角性检查（新增） ----------
def _is_rightish(approx, tol_deg=20):
    """
    判断四边形是否近似直角：每个内角在 90±tol_deg 之间
    """
    if approx is None or len(approx) != 4:
        return False
    pts = approx.reshape(-1, 2).astype(np.float32)

    def ang(a, b, c):
        ba = a - b
        bc = c - b
        den = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosv = np.dot(ba, bc) / den
        return np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))

    angles = [ang(pts[(i-1) % 4], pts[i], pts[(i+1) % 4]) for i in range(4)]
    return all((90 - tol_deg) <= a <= (90 + tol_deg) for a in angles)

# ---------------- 外框检测（改进版） ----------------
def select_outer_rect(cnts, W, H):
    """
    更稳的外框选择：
    - 用 minAreaRect 的宽高比与 A4 比例比较（抗旋转）
    - 只接受四边形且近似直角
    - 面积大、居中性好得分高；贴边会扣分（但不直接淘汰）
    返回： (Box(x, y, w, h), contour) 或 (None, None)
    """
    # 策略参数（可微调）
    margin = 8                               # 贴边“容忍”像素
    min_area = 0.002 * W * H                 # 候选最小占画面比例
    max_area = 0.85  * W * H                 # 候选最大占画面比例
    ratio_rel_tol = 0.25                     # 宽高比相对误差容忍（0.25 ≈ ±25%）
    right_angle_tol_deg = 20                 # 直角容忍度 ±20°

    best_box, best_c, best_score = None, None, -1e9

    for c in cnts:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue

        # 旋转矩形：用它的 w/h 来判断比例，抗旋转/倾斜
        (cx, cy), (rw, rh), _ = cv2.minAreaRect(c)
        if rw <= 1 or rh <= 1:
            continue
        aspect_rot = max(rw, rh) / (min(rw, rh) + 1e-6)
        ratio_penalty = abs(aspect_rot / (RECT_RATIO + 1e-6) - 1.0)
        if ratio_penalty > ratio_rel_tol:
            continue  # A4 比例偏差太大

        # 多边形逼近，强制四边形 + 直角性检查
        per = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * per, True)
        if len(approx) != 4:
            # 试着对凸包再逼近一次
            hull = cv2.convexHull(c)
            approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        if len(approx) != 4:
            continue
        if not _is_rightish(approx, tol_deg=right_angle_tol_deg):
            continue

        # 轴对齐框仅用于面积/贴边评分与后续 scale_from_rect
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if not (min_area <= area <= max_area):
            continue

        # 贴边惩罚：允许贴边，但每贴一边扣一点分
        touch_edges = int(x <= margin) + int(y <= margin) + int(x + w >= W - margin) + int(y + h >= H - margin)
        edge_penalty = 0.4 * touch_edges

        # 居中程度（越靠中心越好）
        center_dist = np.hypot(cx - W / 2.0, cy - H / 2.0) / float(max(W, H))

        # 综合评分：面积占比↑、比例更准↑、更居中↑、少贴边↑
        area_score = area / float(W * H)
        score = 3.0 * area_score - 4.0 * ratio_penalty + 2.0 * (1.0 - center_dist) - edge_penalty

        if score > best_score:
            best_score = score
            best_box = Box(int(x), int(y), int(w), int(h))
            best_c = c

    return best_box, best_c

# ---------------- 尺度换算 ----------------
def scale_from_rect(rect_box: Box):
    if rect_box is None:
        return None, None, None
    cm_per_px_y = REAL_HEIGHT_CM / float(rect_box.h)
    cm_per_px_x = REAL_WIDTH_CM  / float(rect_box.w)
    cm_per_px   = 0.5*(cm_per_px_x + cm_per_px_y)
    return cm_per_px, cm_per_px_y, cm_per_px_x

def inside_rect(bx, by, bw, bh, rx, ry, rw, rh, margin=3):
    return (bx > rx+margin) and (by > ry+margin) and (bx+bw < rx+rw-margin) and (by+bh < ry+rh-margin)

def approx_polygon(c, eps_ratio=0.02):
    return cv2.approxPolyDP(c, eps_ratio*cv2.arcLength(c, True), True)

def classify_shape(c):
    approx = approx_polygon(c, 0.02)
    sides = len(approx)
    if sides == 3:
        return 'triangle', approx
    if sides == 4:
        pts = approx.reshape(-1,2)
        e = np.linalg.norm(np.roll(pts,-1,axis=0)-pts, axis=1)
        ratio = e.max()/(e.min()+1e-6)
        def ang(a,b,c):
            ba=a-b; bc=c-b
            cosv = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
            return np.degrees(np.arccos(np.clip(cosv,-1,1)))
        angs = [ang(pts[(i-1)%4], pts[i], pts[(i+1)%4]) for i in range(4)]
        rightish = all(70<=a<=110 for a in angs)
        if ratio < 1.15 and rightish:
            return 'square', approx
        else:
            return 'rectangle', approx
    if circularity(c) > 0.85:
        return 'circle', approx
    return 'polygon', approx

def polygon_side_lengths_cm(pts, cm_per_px_y, cm_per_px_x):
    pts_cm = np.stack([pts[:,0]*cm_per_px_x, pts[:,1]*cm_per_px_y], axis=1)
    nxt = np.roll(pts_cm, -1, axis=0)
    edges = np.linalg.norm(nxt - pts_cm, axis=1)
    return edges

def measure_circle(c, cm_per_px, rect_box: Box):
    (cx, cy), r = cv2.minEnclosingCircle(c)
    dia_px = 2.0*r
    dia_cm = dia_px * cm_per_px if cm_per_px is not None else None
    if rect_box is not None:
        short_edge = min(rect_box.w, rect_box.h)
        if not (0.10*short_edge <= dia_px <= 0.90*short_edge):
            return None, None, None, None
    return dia_px, dia_cm, (int(cx), int(cy)), int(r)

def compute_size_metric(rect_box: Box, use_mode: str):
    if rect_box is None: 
        return None
    return float(rect_box.h) if use_mode == "HEIGHT" else float(np.sqrt(rect_box.w * rect_box.h))

# ---------------- 备选：Hough 圆（保留接口） ----------------
def hough_find_circles(frame, rect_box):
    if rect_box is None:
        return []
    x, y, w, h = rect_box
    roi = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=60, param2=25, minRadius=10, maxRadius=min(w, h)//2
    )
    result = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cx, cy, r = i
            result.append((cx + x, cy + y, r))
    return result
