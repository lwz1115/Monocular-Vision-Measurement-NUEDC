# utils.py

import cv2
import numpy as np
import json
import os
from collections import namedtuple

Box = namedtuple("Box", ["x", "y", "w", "h"])

# 摄像头配置
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 960

# A4 尺寸
REAL_H_CM, REAL_W_CM = 26.0, 17.0
RECT_RATIO = REAL_H_CM / REAL_W_CM
RATIO_TOL = 0.4
PARAM_FILE = "ref_params.json"

# 检测参数
MIN_CONTOUR_AREA = 80
ANGLE_TOL = 25
RATIO_TOL_SQ = 0.35
MIN_SIDE_CM = 5.0
ROI_MARGIN_PX = 8

# 边缘补偿
EDGE_BIAS_AT_REF_PX = 1
EDGE_BIAS_EXP = 1

def load_params():
    if not os.path.exists(PARAM_FILE):
        raise RuntimeError("未找到标定文件 ref_params.json，请先运行主程序保存参数")
    with open(PARAM_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_distance(curr_metric, ref_metric, distance_ref_cm):
    if distance_ref_cm is None or ref_metric is None or curr_metric in (None, 0):
        return None
    return distance_ref_cm * (ref_metric / curr_metric)

def compute_size_metric(rect_box, mode="HEIGHT"):
    if rect_box is None:
        return None
    w, h = rect_box.w, rect_box.h
    return float(h) if mode == "HEIGHT" else float(np.sqrt(w * h))

def preprocess(gray):
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    sharp_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    gray = cv2.filter2D(gray, -1, sharp_kernel)
    return gray

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

def select_outer_rect(cnts, W, H):
    margin = 20
    min_area = 0.002 * W * H
    max_area = 0.60 * W * H
    best_box, best_score = None, -1e9
    for c in cnts:
        a = cv2.contourArea(c)
        if a < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if x <= margin or y <= margin or (x + w) >= (W - margin) or (y + h) >= (H - margin):
            continue
        area = w * h
        if not (min_area <= area <= max_area):
            continue
        aspect = max(w,h)/(min(w,h)+1e-6)
        approx = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)
        if len(approx) in (4,5,6):
            cx, cy = x+w/2.0, y+h/2.0
            center_dist = np.hypot(cx-W/2.0, cy-H/2.0)/max(W,H)
            score = (area)/(W*H)*3.0
            if h > w: score += 1.0
            score -= abs(aspect - RECT_RATIO)/max(RECT_RATIO,1.0)*2.0
            score += (1.0-center_dist)*2.0
            if score > best_score and abs(aspect - RECT_RATIO) < RATIO_TOL:
                best_score = score
                best_box = Box(x, y, w, h)
    return best_box

def is_square(approx):
    if len(approx) != 4:
        return False
    pts = approx.reshape(4,2).astype(np.float32)
    e = np.linalg.norm(np.roll(pts,-1,axis=0)-pts, axis=1)
    ratio = e.max()/(e.min()+1e-6)
    if ratio > (1.0 + RATIO_TOL_SQ):
        return False
    def ang(a,b,c):
        ba=a-b; bc=c-b
        cosv = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
        return np.degrees(np.arccos(np.clip(cosv,-1,1)))
    angs=[ang(pts[(i-1)%4], pts[i], pts[(i+1)%4]) for i in range(4)]
    return all(90-ANGLE_TOL<=a<=90+ANGLE_TOL for a in angs)
