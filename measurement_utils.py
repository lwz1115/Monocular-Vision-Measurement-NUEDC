import cv2
import numpy as np
import json
import os
from collections import namedtuple

Box = namedtuple("Box", ["x", "y", "w", "h"])

CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 960
REAL_H_CM, REAL_W_CM = 26.0, 17.0  # 标定纸实际尺寸（单位cm）
RECT_RATIO = REAL_H_CM / REAL_W_CM
RATIO_TOL = 0.4
MIN_CONTOUR_AREA = 80
PARAM_FILE = "ref_params.json"

def load_params():
    if not os.path.exists(PARAM_FILE):
        raise RuntimeError("未找到 ref_params.json，请先运行主程序标定参考参数。")
    with open(PARAM_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def preprocess(gray):
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
    clahe = cv2.createCLAHE(2.0, (8,8))
    gray = clahe.apply(gray)
    sharp_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(gray, -1, sharp_kernel)

def find_contours(frame):
    gray = preprocess(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    median = np.median(gray)
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(gray, lower, upper)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cnts

def select_outer_rect(cnts, W, H):
    margin = 20
    min_area = 0.002 * W * H
    max_area = 0.60 * W * H
    best_box, best_score = None, -1e9
    best_cnt = None
    for c in cnts:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if x <= margin or y <= margin or (x + w) >= (W - margin) or (y + h) >= (H - margin):
            continue
        area = w * h
        aspect = max(w,h)/(min(w,h)+1e-6)
        if not (min_area <= area <= max_area) or abs(aspect - RECT_RATIO) > RATIO_TOL:
            continue
        cx, cy = x+w/2.0, y+h/2.0
        center_dist = np.hypot(cx-W/2.0, cy-H/2.0)/max(W,H)
        score = area/(W*H)*3.0 + (1.0 - center_dist)*2.0 - abs(aspect - RECT_RATIO)*2.0
        if score > best_score:
            best_score = score
            best_box = Box(x, y, w, h)
            best_cnt = c
    return best_box, best_cnt

def compute_distance(rect_box, ref_metric, distance_ref_cm):
    """
    HEIGHT模式：用矩形的高度（像素）来估算距离
    rect_box: Box类型，代表检测到的矩形
    ref_metric: 标定时矩形的高度（像素）
    distance_ref_cm: 标定距离（cm）
    返回当前距离估算值（cm）
    """
    if rect_box is None:
        return None
    curr_height = rect_box.h
    if curr_height == 0:
        return None
    return distance_ref_cm * (ref_metric / curr_height)

def get_rect_corners_from_cnt(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4,2)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return np.int32(box)

def order_points(pts):
    pts = np.array(pts)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_perspective(img, src_pts, width, height):
    dst_pts = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warp = cv2.warpPerspective(img, M, (width, height))
    return warp, M

def find_shortest_extended_edge(sub_img):
    gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    poly = cv2.approxPolyDP(largest_contour, epsilon, True)
    points = poly.reshape(-1, 2)
    if len(points) < 2:
        return None, None, None
    def extend_line(p1, p2, length=5):
        direction = p2 - p1
        norm = np.linalg.norm(direction)
        if norm == 0:
            return np.int32([p1, p2])
        direction = direction / norm
        p1_ext = p1 - direction * length
        p2_ext = p2 + direction * length
        return np.int32([p1_ext, p2_ext])
    def is_crossing_white(p1, p2, mask, length=5):
        line = extend_line(p1, p2, length)
        p1_ext, p2_ext = line[0], line[1]
        def in_white(p):
            x, y = int(p[0]), int(p[1])
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                return mask[y, x] == 0
            return False
        return in_white(p1_ext) and in_white(p2_ext)
    min_length = float('inf')
    shortest_line = None
    shortest_points = None
    n = len(points)
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        if is_crossing_white(p1, p2, bin_img, length=5):
            extended = extend_line(p1, p2, length=5)
            line_length = np.linalg.norm(extended[1] - extended[0])
            if line_length < min_length:
                min_length = line_length
                shortest_line = extended
                shortest_points = (p1, p2)
    return shortest_line, shortest_points, points
