import cv2
import numpy as np
import time
from collections import Counter, deque
from utils1 import *

PRINT_INTERVAL = 0.5
AUTO_STOP_SECONDS = 5    # 自动停止秒数
MAX_HISTORY_LEN = 20     # 全局滑动窗口长度（建议10~30之间）

def get_most_common_value(lst):
    if not lst: return None
    c = Counter(lst)
    return c.most_common(1)[0][0]

def reject_outlier_mode(mode_val, vals, max_diff=0.2):
    if not vals or mode_val is None:
        return True   # 没值直接判为波动，不输出
    avg = sum(vals) / len(vals)
    return abs(mode_val - avg) > max_diff

def append_to_history(history, val):
    history.append(val)
    while len(history) > MAX_HISTORY_LEN:
        history.popleft()

def main():
    params = load_params()
    distance_ref_cm = float(params["distance_ref_cm"])
    ref_metric = float(params["ref_metric"])
    metric_mode = params.get("metric_mode", "HEIGHT")

    cap = cv2.VideoCapture(CAM_INDEX)
    # cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    square_history = deque()
    distance_history = deque()
    _last_print_ts = 0.0
    start_time = time.time()

    try:
        while True:
            now = time.time()
            # 自动退出
            if now - start_time >= AUTO_STOP_SECONDS:
                print(f"\n已自动运行 {AUTO_STOP_SECONDS} 秒，程序停止。")
                break

            ok, frame = cap.read()
            if not ok:
                # 也给窗口一个机会响应关闭
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n已按 q 退出。")
                    break
                continue

            # 复制一份用于可视化
            display_frame = frame.copy()

            # 寻找外接矩形
            edges, cnts = find_contours(frame)
            rect_box = select_outer_rect(cnts, FRAME_W, FRAME_H)
            if not rect_box:
                # 没找到时也显示原始画面，便于调试
                cv2.putText(display_frame, "No outer rect found",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow("Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n已按 q 退出。")
                    break
                # 周期性打印众数（即便没找到矩形）
                if now - _last_print_ts >= PRINT_INTERVAL:
                    _last_print_ts = now
                    if distance_history:
                        dist_mode = get_most_common_value(distance_history)
                        if dist_mode is not None and not reject_outlier_mode(dist_mode, distance_history, 0.2):
                            print(f"[参考目标距离-全局众数] {dist_mode:.2f} cm")
                    if square_history:
                        mode_val = get_most_common_value(square_history)
                        if mode_val is not None and not reject_outlier_mode(mode_val, square_history, 0.2):
                            print(f"[最小正方形边长-全局众数] {mode_val:.2f} cm")
                continue

            x, y, w, h = rect_box
            # 画出参考外接矩形
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # 像素-厘米比例估计
            cm_per_px_y = REAL_H_CM / float(h)
            cm_per_px_x = REAL_W_CM / float(w)
            cm_per_px_avg = 0.5*(cm_per_px_x + cm_per_px_y)

            # 计算度量与距离
            curr_metric = compute_size_metric(rect_box, metric_mode)
            distance_cm = compute_distance(curr_metric, ref_metric, distance_ref_cm)
            append_to_history(distance_history, round(distance_cm, 2))

            # 边缘补偿
            scale = curr_metric / max(ref_metric, 1e-6)
            delta_px = EDGE_BIAS_AT_REF_PX * (1.0 / max(scale, 1e-3))**EDGE_BIAS_EXP
            edge_comp_cm = 2.0 * delta_px * cm_per_px_avg

            # ROI
            margin = max(ROI_MARGIN_PX, int(0.01*min(w, h)))
            rx, ry = x+margin, y+margin
            rw, rh = w-2*margin, h-2*margin
            rx, ry = max(rx,0), max(ry,0)
            rw, rh = max(rw,1), max(rh,1)
            roi = frame[ry:ry+rh, rx:rx+rw]

            # 二值化与形态学
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray,(3,3),0)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            ks = 1 if min(rw, rh) < 220 else 2
            kernel = np.ones((ks, ks), np.uint8)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, 1)

            extra = max(0, int(round((1.0 / max(scale,1e-3)) - 1.0)))
            if extra > 0:
                th = cv2.dilate(th, np.ones((1,1), np.uint8), iterations=min(2, extra))

            cnts,_ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # 面积与边长阈值
            min_side_px = max(6, int(MIN_SIDE_CM / max(cm_per_px_avg, 1e-6)))
            min_area_px = max(MIN_CONTOUR_AREA, int(0.5 * min_side_px * min_side_px))

            squares = []
            for c in cnts:
                if cv2.contourArea(c) < min_area_px:
                    continue
                approx_s = cv2.approxPolyDP(c, 0.015*cv2.arcLength(c, True), True)
                if not is_square(approx_s):
                    continue
                approx_g = approx_s + np.array([[[rx, ry]]], dtype=np.int32)  # 回到全图坐标
                pts = approx_g.reshape(-1,2).astype(np.float32)
                wpx = np.linalg.norm(pts[0]-pts[1])
                hpx = np.linalg.norm(pts[1]-pts[2])
                side_cm_raw = 0.5*(wpx*cm_per_px_x + hpx*cm_per_px_y)
                side_cm = side_cm_raw + edge_comp_cm
                if side_cm >= MIN_SIDE_CM:
                    squares.append((side_cm, approx_g))

            # 可视化检测到的正方形，并记录最小边长
            if squares:
                # 最小边长的正方形
                smin, amin = min(squares, key=lambda x: x[0])
                append_to_history(square_history, round(smin, 2))
                # 画全部正方形
                for side_cm, pts in squares:
                    cv2.polylines(display_frame, [pts], True, (0, 0, 255), 2)
                    cv2.putText(display_frame, f"{side_cm:.2f}cm",
                                tuple(pts[0][0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # 叠加文字信息（当前距离与全局众数）
            overlay_lines = [f"Distance: {distance_cm:.2f} cm"]
            dist_mode = get_most_common_value(distance_history) if distance_history else None
            if dist_mode is not None and not reject_outlier_mode(dist_mode, distance_history, 0.2):
                overlay_lines.append(f"Dist(mode): {dist_mode:.2f} cm")

            sq_mode = get_most_common_value(square_history) if square_history else None
            if sq_mode is not None and not reject_outlier_mode(sq_mode, square_history, 0.2):
                overlay_lines.append(f"Square min side(mode): {sq_mode:.2f} cm")

            # 左上角绘制文本
            y0 = 30
            for i, txt in enumerate(overlay_lines):
                cv2.putText(display_frame, txt, (10, y0 + i*28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # 显示窗口
            cv2.imshow("Detection", display_frame)
            # 若想同时查看 ROI 或二值图，可取消下面注释：
            # cv2.imshow("ROI", roi)
            # cv2.imshow("ROI_Thresh", th)

            # 键盘处理：q 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n已按 q 退出。")
                break

            # 周期性打印终端输出（抗波动后的众数）
            if now - _last_print_ts >= PRINT_INTERVAL:
                _last_print_ts = now
                if distance_history:
                    dist_mode = get_most_common_value(distance_history)
                    if dist_mode is not None and not reject_outlier_mode(dist_mode, distance_history, 0.2):
                        print(f"[参考目标距离-全局众数] {dist_mode:.2f} cm")
                if square_history:
                    mode_val = get_most_common_value(square_history)
                    if mode_val is not None and not reject_outlier_mode(mode_val, square_history, 0.2):
                        print(f"[最小正方形边长-全局众数] {mode_val:.2f} cm")

    except KeyboardInterrupt:
        print("\n已手动退出。")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
