import cv2
import time
from collections import deque, Counter
import numpy as np
from core import (
    FRAME_WIDTH, FRAME_HEIGHT, SHOW_INTERNAL_DEBUG, MIN_CONTOUR_AREA,
    REAL_HEIGHT_CM, REAL_WIDTH_CM, RECT_RATIO, RATIO_TOL,
    save_ref_params, load_ref_params,
    preprocess, find_contours, select_outer_rect, scale_from_rect,
    inside_rect, approx_polygon, classify_shape, polygon_side_lengths_cm, measure_circle, compute_size_metric,
    # hough_find_circles  # 不再使用 Hough，保留这行也无妨
)

CAM_INDEX = 0

PRINT_INTERVAL = 0.5
AUTO_STOP_SECONDS = 5   # 自动运行 20 秒后关闭（可改）
_last_print_ts = 0.0

distance_ref_cm = None
ref_metric = None
metric_mode = "HEIGHT"       # 只保留 HEIGHT

# 滑动窗口相关
MAX_HISTORY_LEN = 5          # 可调
REJECT_DIFF_CM = 0.2         # 众数与均值偏差阈值（cm）

MIN_SIZE_CM = 8              # 小于该尺寸的形状一律过滤

WINDOW_NAME = "Measurement Viewer"  # 显示窗口名

# ========= 圆检测（轮廓法）参数 =========
CIRC_MIN_CIRCULARITY = 0.80     # 圆度阈值：越接近1越圆
CIRC_AREA_RATIO_TOL = 0.30      # 轮廓面积与外接圆面积的相对误差容忍度
CANNY_LO = 60
CANNY_HI = 160
BLUR_KSIZE = 5


# ========== 工具函数 ==========
def get_most_common_value(lst):
    if not lst:
        return None
    c = Counter(lst)
    return c.most_common(1)[0][0]


def reject_outlier_mode(mode_val, vals, max_diff=REJECT_DIFF_CM):
    if not vals or mode_val is None:
        return True   # 无值直接判为波动
    avg = sum(vals) / len(vals)
    return abs(mode_val - avg) > max_diff


def append_to_history(history, val):
    history.append(val)
    while len(history) > MAX_HISTORY_LEN:
        history.popleft()


def compute_distance(curr_metric):
    if distance_ref_cm is None or ref_metric is None or curr_metric in (None, 0):
        return None
    return distance_ref_cm * (ref_metric / curr_metric)


def draw_text(img, text, org, scale=0.6, color=(50, 230, 50), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def detect_circles_by_contour(frame, rect_box=None):
    """
    返回 [(cx, cy, r), ...] 像素坐标与半径，按半径从大到小排序
    逻辑：模糊 -> Canny -> 轮廓 -> 圆度与面积比过滤 -> minEnclosingCircle
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if BLUR_KSIZE > 1:
        gray = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)

    edges = cv2.Canny(gray, CANNY_LO, CANNY_HI)

    # 若限定在A4参考矩形区域内，则做个遮罩，避免外部干扰
    if rect_box is not None:
        rx, ry, rw, rh = rect_box
        mask = np.zeros_like(edges)
        cv2.rectangle(mask, (rx, ry), (rx + rw, ry + rh), 255, -1)
        edges = cv2.bitwise_and(edges, mask)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < max(MIN_CONTOUR_AREA, 20):  # 过滤很小的噪声
            continue

        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue

        circularity = 4.0 * np.pi * (area / (peri * peri))
        if circularity < CIRC_MIN_CIRCULARITY:
            continue

        # 最小外接圆一致性检查
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r <= 0:
            continue

        circle_area = np.pi * r * r
        if circle_area <= 0:
            continue
        ratio_err = abs(area - circle_area) / circle_area
        if ratio_err > CIRC_AREA_RATIO_TOL:
            continue

        # 通过所有检查，作为圆候选
        circles.append((int(round(cx)), int(round(cy)), int(round(r))))

    # 半径从大到小排序
    circles.sort(key=lambda t: t[2], reverse=True)
    return circles


# ========== 主逻辑 ==========
def main():
    global _last_print_ts, distance_ref_cm, ref_metric, metric_mode

    params = load_ref_params()
    if params:
        distance_ref_cm = params["distance_ref_cm"]
        ref_metric = params["ref_metric"]
        print(f"✅ 已自动载入上次参考参数: 距离={distance_ref_cm:.2f}cm, metric={ref_metric:.2f}px, mode=HEIGHT")
    else:
        # 若没有保存的参考参数，先询问一次距离
        distance_ref_cm = float(input("请输入 A4 纸到摄像机的参考距离 (cm)：").strip())

    # 打开摄像头
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    # cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    print(f"程序已启动，自动测量 {AUTO_STOP_SECONDS} 秒后关闭。按 'q' 可手动退出。")

    ref_saved = params is not None

    # 初始化滑动窗口
    distance_history = deque()
    circle_history = deque()
    triangle_history = deque()
    square_history = deque()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # 可缩放窗口
    cv2.resizeWindow(WINDOW_NAME, 960, 540)

    start_time = time.time()
    while True:
        now = time.time()
        # 自动停止
        if now - start_time >= AUTO_STOP_SECONDS:
            print(f"\n已自动运行 {AUTO_STOP_SECONDS} 秒，程序停止。")
            break

        ok, frame = cap.read()
        if not ok:
            # 读取失败也要让 UI 响应按键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        vis = frame.copy()  # 用于可视化的画面

        H, W = frame.shape[:2]
        edges, cnts = find_contours(frame)
        rect_box, rect_contour = select_outer_rect(cnts, W, H)
        cm_per_px, cm_per_px_y, cm_per_px_x = scale_from_rect(rect_box)

        # --- 可视化：外部参考矩形（A4） ---
        if rect_box is not None:
            rx, ry, rw, rh = rect_box
            cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
            draw_text(vis, "A4 reference", (rx, max(0, ry - 8)), 0.6, (0, 255, 255), 2)

        # --- 轮廓法圆检测（优先统计最大圆） ---
        contour_circles = detect_circles_by_contour(frame, rect_box)
        largest_circle = None  # (cx, cy, r, dia_cm)
        for (cx, cy, r) in contour_circles:
            # 画出所有找到的圆（细线）
            cv2.circle(vis, (cx, cy), r, (200, 200, 200), 1)
            cv2.circle(vis, (cx, cy), 2, (200, 200, 200), 2)

            if cm_per_px:
                dia_cm = 2 * r * cm_per_px
                if (largest_circle is None) or (dia_cm > largest_circle[3]):
                    largest_circle = (cx, cy, r, dia_cm)

        if largest_circle is not None and largest_circle[3] >= MIN_SIZE_CM:
            cx, cy, r, dia_cm = largest_circle
            # 高亮最大圆
            cv2.circle(vis, (cx, cy), r, (0, 200, 255), 3)
            draw_text(vis, f"Circle dia ~ {dia_cm:.2f} cm", (max(0, cx - r), max(0, cy - r - 10)), 0.7, (0, 200, 255), 2)
            append_to_history(circle_history, round(dia_cm, 2))

        # 距离参考（用外框高度作为 metric）
        if rect_box is not None:
            curr_metric = compute_size_metric(rect_box, "HEIGHT")
            if (ref_metric is None) and (curr_metric is not None):
                ref_metric = curr_metric
                print(f"✅ 已保存参考度量(HEIGHT) = {ref_metric:.2f} px，对应参考距离 {distance_ref_cm:.2f} cm")
                save_ref_params(distance_ref_cm, ref_metric, "HEIGHT")
                ref_saved = True

        # 检测内部三角形/正方形（取最大者）
        found = {'circle': None, 'triangle': None, 'square': None, 'rectangle': None}
        if rect_box:
            rx, ry, rw, rh = rect_box
            for c in cnts:
                if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                    continue
                bx, by, bw, bh = cv2.boundingRect(c)
                if not inside_rect(bx, by, bw, bh, rx, ry, rw, rh, margin=3):
                    continue
                cls, approx = classify_shape(c)
                if cls in found and (found[cls] is None or cv2.contourArea(c) > cv2.contourArea(found[cls])):
                    found[cls] = c

        # 三角形
        if found['triangle'] is not None and (cm_per_px_y is not None) and (cm_per_px_x is not None):
            approx = approx_polygon(found['triangle'])
            pts = approx.reshape(-1, 2).astype(np.int32)
            cv2.polylines(vis, [pts], True, (255, 180, 0), 3)
            fpts = approx.reshape(-1, 2).astype(np.float32)
            edges_cm = polygon_side_lengths_cm(fpts, cm_per_px_y, cm_per_px_x)
            if edges_cm.max() >= MIN_SIZE_CM:
                e_max, e_min = float(edges_cm.max()), float(edges_cm.min())
                ratio = e_max / (e_min + 1e-6)
                if ratio < 1.08:  # 等边
                    mean_side = float(edges_cm.mean())
                    append_to_history(triangle_history, round(mean_side, 2))
                    # 标注
                    cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                    draw_text(vis, f"Equilateral ~ {mean_side:.2f} cm", (max(0, cx - 80), max(0, cy - 10)), 0.6, (255, 180, 0), 2)

        # 正方形
        if found['square'] is not None and (cm_per_px_y is not None) and (cm_per_px_x is not None):
            approx = approx_polygon(found['square'])
            pts = approx.reshape(-1, 2).astype(np.int32)
            cv2.polylines(vis, [pts], True, (255, 0, 120), 3)
            fpts = approx.reshape(-1, 2).astype(np.float32)
            edges_cm = polygon_side_lengths_cm(fpts, cm_per_px_y, cm_per_px_x)
            side_cm = float(edges_cm.mean())
            if side_cm >= MIN_SIZE_CM:
                append_to_history(square_history, round(side_cm, 2))
                cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                draw_text(vis, f"Square ~ {side_cm:.2f} cm", (max(0, cx - 60), max(0, cy - 10)), 0.6, (255, 0, 120), 2)

        # 距离估计（由参考外框高度推算）
        if rect_box is not None and ref_metric is not None:
            curr_metric = compute_size_metric(rect_box, "HEIGHT")
            distance_cm = compute_distance(curr_metric)
            if distance_cm is not None:
                append_to_history(distance_history, round(distance_cm, 2))

        # --- 滑动窗口稳定输出（控制台） ---
        if now - _last_print_ts >= PRINT_INTERVAL:
            _last_print_ts = now
            print("-" * 60)
            print(time.strftime("[%H:%M:%S]"))
            # 距离
            if distance_history:
                dist_mode = get_most_common_value(distance_history)
                if dist_mode is not None and not reject_outlier_mode(dist_mode, distance_history):
                    print(f"[目标距离-全局众数] {dist_mode:.2f} cm")
            # 圆
            if circle_history:
                circle_mode = get_most_common_value(circle_history)
                if circle_mode is not None and not reject_outlier_mode(circle_mode, circle_history):
                    print(f"[圆直径-全局众数] {circle_mode:.2f} cm")
            # 三角形
            if triangle_history:
                tri_mode = get_most_common_value(triangle_history)
                if tri_mode is not None and not reject_outlier_mode(tri_mode, triangle_history):
                    print(f"[等边三角形边长-全局众数] {tri_mode:.2f} cm")
            # 正方形
            if square_history:
                square_mode = get_most_common_value(square_history)
                if square_mode is not None and not reject_outlier_mode(square_mode, square_history):
                    print(f"[正方形边长-全局众数] {square_mode:.2f} cm")

        # --- 叠加 HUD 信息（画面左上角） ---
        y = 24
        draw_text(vis, "Press 'q' to quit", (10, y), 0.6, (255, 255, 255), 2); y += 24
        if cm_per_px is not None:
            draw_text(vis, f"Scale ~ {cm_per_px:.4f} cm/px", (10, y), 0.6, (180, 255, 180), 2); y += 24
        if distance_history:
            dist_mode = get_most_common_value(distance_history)
            if dist_mode is not None and not reject_outlier_mode(dist_mode, distance_history):
                draw_text(vis, f"Distance ~ {dist_mode:.2f} cm", (10, y), 0.6, (180, 255, 255), 2); y += 24
        if circle_history:
            circle_mode = get_most_common_value(circle_history)
            if circle_mode is not None and not reject_outlier_mode(circle_mode, circle_history):
                draw_text(vis, f"Circle dia ~ {circle_mode:.2f} cm", (10, y), 0.6, (200, 220, 255), 2); y += 24
        if triangle_history:
            tri_mode = get_most_common_value(triangle_history)
            if tri_mode is not None and not reject_outlier_mode(tri_mode, triangle_history):
                draw_text(vis, f"Equilateral side ~ {tri_mode:.2f} cm", (10, y), 0.6, (255, 220, 180), 2); y += 24
        if square_history:
            square_mode = get_most_common_value(square_history)
            if square_mode is not None and not reject_outlier_mode(square_mode, square_history):
                draw_text(vis, f"Square side ~ {square_mode:.2f} cm", (10, y), 0.6, (255, 200, 220), 2); y += 24

        # 显示边缘图可选（按需开启）
        # edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # cv2.imshow("Edges", edges_vis)

        # === 关键：显示窗口 ===
        cv2.imshow(WINDOW_NAME, vis)

        # 键盘事件：按 'q' 退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("收到退出指令，正在关闭...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
