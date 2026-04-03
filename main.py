#!/usr/bin/env python3
import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTreeWidgetItem, QStyledItemDelegate, QFileDialog, QWidget,QTableWidget, \
    QTableWidgetItem, QButtonGroup, QHeaderView, QCompleter, QMessageBox, QComboBox, QLineEdit, QPushButton,QAbstractItemView,QItemDelegate
from PyQt5.QtCore import Qt, QCoreApplication, QThread, pyqtSignal, QStringListModel, QObject, QEvent, QDate, QVariant
from PyQt5.QtGui import QColor, QBrush,QDoubleValidator
import re
import sensor_interface
from PyQt5.QtWidgets import QFileDialog,QMainWindow, QMessageBox, QHBoxLayout, QPushButton, QDialog,QTableWidgetItem, QApplication, QWidget
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import Qt, pyqtSignal, QCoreApplication,QThread, QTimer
from winUI import Ui_MainWindow
import threading
from PyQt5.QtCore import Qt
import ddddocr
import cv2
import time
from collections import deque, Counter
import numpy as np
from queue import Queue

# 线程池JC1
class Thread_JC1(QThread):
    # 定义一个自定义信号，用于向主线程发送消息
    images_signal = pyqtSignal(np.ndarray)
    msg_signal =  pyqtSignal(bool)
    print_signal =  pyqtSignal(str)
    from core import (
        FRAME_WIDTH, FRAME_HEIGHT, MIN_CONTOUR_AREA
    )

    CAM_INDEX = 0

    PRINT_INTERVAL = 0.5
    AUTO_STOP_SECONDS = 5  # 自动运行 20 秒后关闭（可改）
    _last_print_ts = 0.0

    distance_ref_cm = None
    ref_metric = None
    metric_mode = "HEIGHT"  # 只保留 HEIGHT

    # 滑动窗口相关
    MAX_HISTORY_LEN = 5  # 可调
    REJECT_DIFF_CM = 0.2  # 众数与均值偏差阈值（cm）

    MIN_SIZE_CM = 8  # 小于该尺寸的形状一律过滤

    WINDOW_NAME = "Measurement Viewer"  # 显示窗口名

    # ========= 圆检测（轮廓法）参数 =========
    CIRC_MIN_CIRCULARITY = 0.80  # 圆度阈值：越接近1越圆
    CIRC_AREA_RATIO_TOL = 0.30  # 轮廓面积与外接圆面积的相对误差容忍度
    CANNY_LO = 60
    CANNY_HI = 160
    BLUR_KSIZE = 5


    def __init__(self, stop_status,parent=None):
        super().__init__(parent)  # 必须调用父类初始化
        self.stop_status = stop_status

    # ========== 工具函数 ==========
    def get_most_common_value(self,lst):
        if not lst:
            return None
        c = Counter(lst)
        return c.most_common(1)[0][0]

    def reject_outlier_mode(self,mode_val, vals, max_diff=REJECT_DIFF_CM):
        if not vals or mode_val is None:
            return True  # 无值直接判为波动
        avg = sum(vals) / len(vals)
        return abs(mode_val - avg) > max_diff

    def append_to_history(self,history, val):
        history.append(val)
        while len(history) > self.MAX_HISTORY_LEN:
            history.popleft()

    def compute_distance(self,curr_metric):
        if self.distance_ref_cm is None or self.ref_metric is None or curr_metric in (None, 0):
            return None
        return self.distance_ref_cm * (self.ref_metric / curr_metric)

    def draw_text(self,img, text, org, scale=0.6, color=(50, 230, 50), thick=2):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def detect_circles_by_contour(self,frame, rect_box=None):
        """
        返回 [(cx, cy, r), ...] 像素坐标与半径，按半径从大到小排序
        逻辑：模糊 -> Canny -> 轮廓 -> 圆度与面积比过滤 -> minEnclosingCircle
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.BLUR_KSIZE > 1:
            gray = cv2.GaussianBlur(gray, (self.BLUR_KSIZE, self.BLUR_KSIZE), 0)

        edges = cv2.Canny(gray, self.CANNY_LO, self.CANNY_HI)

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
            if area < max(self.MIN_CONTOUR_AREA, 20):  # 过滤很小的噪声
                continue

            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue

            circularity = 4.0 * np.pi * (area / (peri * peri))
            if circularity < self.CIRC_MIN_CIRCULARITY:
                continue

            # 最小外接圆一致性检查
            (cx, cy), r = cv2.minEnclosingCircle(c)
            if r <= 0:
                continue

            circle_area = np.pi * r * r
            if circle_area <= 0:
                continue
            ratio_err = abs(area - circle_area) / circle_area
            if ratio_err > self.CIRC_AREA_RATIO_TOL:
                continue

            # 通过所有检查，作为圆候选
            circles.append((int(round(cx)), int(round(cy)), int(round(r))))

        # 半径从大到小排序
        circles.sort(key=lambda t: t[2], reverse=True)
        return circles

    def run(self):
        from core import load_ref_params,approx_polygon
        params = load_ref_params()
        if params:
            self.distance_ref_cm = params["distance_ref_cm"]
            self.ref_metric = params["ref_metric"]
            print(f"✅ 已自动载入上次参考参数: 距离={self.distance_ref_cm:.2f}cm, metric={self.ref_metric:.2f}px, mode=HEIGHT")
            self.print_signal.emit(f"✅ 已自动载入上次参考参数: 距离={self.distance_ref_cm:.2f}cm, metric={self.ref_metric:.2f}px, mode=HEIGHT")
        else:
            # self.value_input.setEnabled(True)
            # self.value_input.setPlaceholderText("请输入 A4 纸到摄像机的参考距离 (cm)")
            # # 若没有保存的参考参数，先询问一次距离
            # if self.input_distance:
            #     # A4 纸到摄像机的参考距离
            #     self.distance_ref_cm = float(self.input_distance.strip())
            # else:
            #     self.print_signal.emit("请输入 A4 纸到摄像机的参考距离 (cm)")
            #     # 结束程序，恢复按钮
            #     self.msg_signal.emit(True)
            #     return
            self.print_signal.emit("请到后台终端输入 A4 纸到摄像机的参考距离 (cm)")
            self.distance_ref_cm = float(input("请输入 A4 纸到摄像机的参考距离 (cm)：").strip())
            # distance_ref_cm = float(self.value_input.text().strip())

        # 打开摄像头
        cap = cv2.VideoCapture(self.CAM_INDEX, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError("无法打开摄像头")

        print(f"程序已启动，自动测量 {self.AUTO_STOP_SECONDS} 秒后关闭。按 'q' 可手动退出。")
        self.print_signal.emit(f"程序已启动，自动测量 {self.AUTO_STOP_SECONDS} 秒后关闭。")
        ref_saved = params is not None

        # 初始化滑动窗口
        distance_history = deque()
        circle_history = deque()
        triangle_history = deque()
        square_history = deque()

        # cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)  # 可缩放窗口
        # cv2.resizeWindow(self.WINDOW_NAME, 960, 540)

        start_time = time.time()
        while True:
            now = time.time()
            # 自动停止
            if now - start_time >= self.AUTO_STOP_SECONDS:
                print(f"\n已自动运行 {self.AUTO_STOP_SECONDS} 秒，程序停止。")
                self.print_signal.emit(f"\n已自动运行 {self.AUTO_STOP_SECONDS} 秒，程序停止。")
                # 结束程序，恢复按钮
                self.msg_signal.emit(True)
                break

            ok, frame = cap.read()
            if not ok:
                # # 读取失败也要让 UI 响应按键
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                # 结束程序，恢复按钮
                self.msg_signal.emit(True)
                break
                # continue

            vis = frame.copy()  # 用于可视化的画面

            H, W = frame.shape[:2]
            from core import find_contours
            edges, cnts = find_contours(frame)
            from core import select_outer_rect
            rect_box, rect_contour = select_outer_rect(cnts, W, H)
            from core import scale_from_rect
            cm_per_px, cm_per_px_y, cm_per_px_x = scale_from_rect(rect_box)

            # --- 可视化：外部参考矩形（A4） ---
            if rect_box is not None:
                rx, ry, rw, rh = rect_box
                cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
                self.draw_text(vis, "A4 reference", (rx, max(0, ry - 8)), 0.6, (0, 255, 255), 2)

            # --- 轮廓法圆检测（优先统计最大圆） ---
            contour_circles = self.detect_circles_by_contour(frame, rect_box)
            largest_circle = None  # (cx, cy, r, dia_cm)
            for (cx, cy, r) in contour_circles:
                # 画出所有找到的圆（细线）
                cv2.circle(vis, (cx, cy), r, (200, 200, 200), 1)
                cv2.circle(vis, (cx, cy), 2, (200, 200, 200), 2)

                if cm_per_px:
                    dia_cm = 2 * r * cm_per_px
                    if (largest_circle is None) or (dia_cm > largest_circle[3]):
                        largest_circle = (cx, cy, r, dia_cm)

            if largest_circle is not None and largest_circle[3] >= self.MIN_SIZE_CM:
                cx, cy, r, dia_cm = largest_circle
                # 高亮最大圆
                cv2.circle(vis, (cx, cy), r, (0, 200, 255), 3)
                self.draw_text(vis, f"Circle dia ~ {dia_cm:.2f} cm", (max(0, cx - r), max(0, cy - r - 10)), 0.7,
                          (0, 200, 255), 2)
                self.append_to_history(circle_history, round(dia_cm, 2))

            # 距离参考（用外框高度作为 metric）
            if rect_box is not None:
                from core import compute_size_metric
                curr_metric = compute_size_metric(rect_box, "HEIGHT")
                if (self.ref_metric is None) and (curr_metric is not None):
                    self.ref_metric = curr_metric
                    print(f"✅ 已保存参考度量(HEIGHT) = {self.ref_metric:.2f} px，对应参考距离 {self.distance_ref_cm:.2f} cm")
                    self.print_signal.emit(f"✅ 已保存参考度量(HEIGHT) = {self.ref_metric:.2f} px，对应参考距离 {self.distance_ref_cm:.2f} cm")
                    from core import save_ref_params
                    save_ref_params(self.distance_ref_cm, self.ref_metric, "HEIGHT")
                    ref_saved = True

            # 检测内部三角形/正方形（取最大者）
            found = {'circle': None, 'triangle': None, 'square': None, 'rectangle': None}
            if rect_box:
                rx, ry, rw, rh = rect_box
                for c in cnts:
                    if cv2.contourArea(c) < self.MIN_CONTOUR_AREA:
                        continue
                    bx, by, bw, bh = cv2.boundingRect(c)
                    from core import inside_rect
                    if not inside_rect(bx, by, bw, bh, rx, ry, rw, rh, margin=3):
                        continue
                    from core import classify_shape
                    cls, approx = classify_shape(c)
                    if cls in found and (found[cls] is None or cv2.contourArea(c) > cv2.contourArea(found[cls])):
                        found[cls] = c

            # 三角形
            if found['triangle'] is not None and (cm_per_px_y is not None) and (cm_per_px_x is not None):
                approx = approx_polygon(found['triangle'])
                pts = approx.reshape(-1, 2).astype(np.int32)
                cv2.polylines(vis, [pts], True, (255, 180, 0), 3)
                fpts = approx.reshape(-1, 2).astype(np.float32)
                from core import polygon_side_lengths_cm
                edges_cm = polygon_side_lengths_cm(fpts, cm_per_px_y, cm_per_px_x)
                if edges_cm.max() >= self.MIN_SIZE_CM:
                    e_max, e_min = float(edges_cm.max()), float(edges_cm.min())
                    ratio = e_max / (e_min + 1e-6)
                    if ratio < 1.08:  # 等边
                        mean_side = float(edges_cm.mean())
                        self.append_to_history(triangle_history, round(mean_side, 2))
                        # 标注
                        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                        self.draw_text(vis, f"Equilateral ~ {mean_side:.2f} cm", (max(0, cx - 80), max(0, cy - 10)), 0.6,
                                  (255, 180, 0), 2)

            # 正方形
            if found['square'] is not None and (cm_per_px_y is not None) and (cm_per_px_x is not None):
                approx = approx_polygon(found['square'])
                pts = approx.reshape(-1, 2).astype(np.int32)
                cv2.polylines(vis, [pts], True, (255, 0, 120), 3)
                fpts = approx.reshape(-1, 2).astype(np.float32)
                from core import polygon_side_lengths_cm
                edges_cm = polygon_side_lengths_cm(fpts, cm_per_px_y, cm_per_px_x)
                side_cm = float(edges_cm.mean())
                if side_cm >= self.MIN_SIZE_CM:
                    self.append_to_history(square_history, round(side_cm, 2))
                    cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                    self.draw_text(vis, f"Square ~ {side_cm:.2f} cm", (max(0, cx - 60), max(0, cy - 10)), 0.6, (255, 0, 120),
                              2)

            # 距离估计（由参考外框高度推算）
            if rect_box is not None and self.ref_metric is not None:
                from core import compute_size_metric
                curr_metric = compute_size_metric(rect_box, "HEIGHT")
                distance_cm = self.compute_distance(curr_metric)
                if distance_cm is not None:
                    self.append_to_history(distance_history, round(distance_cm, 2))

            # --- 滑动窗口稳定输出（控制台） ---
            if now - self._last_print_ts >= self.PRINT_INTERVAL:
                self._last_print_ts = now
                print("-" * 60)
                self.print_signal.emit("-" * 60)
                print(time.strftime("[%H:%M:%S]"))
                self.print_signal.emit(time.strftime("[%H:%M:%S]"))
                # 距离
                if distance_history:
                    dist_mode = self.get_most_common_value(distance_history)
                    if dist_mode is not None and not self.reject_outlier_mode(dist_mode, distance_history):
                        print(f"[目标距离-全局众数] {dist_mode:.2f} cm")
                        self.print_signal.emit(f"[目标距离-全局众数] {dist_mode:.2f} cm")
                # 圆
                if circle_history:
                    circle_mode = self.get_most_common_value(circle_history)
                    if circle_mode is not None and not self.reject_outlier_mode(circle_mode, circle_history):
                        print(f"[圆直径-全局众数] {circle_mode:.2f} cm")
                        self.print_signal.emit(f"[圆直径-全局众数] {circle_mode:.2f} cm")
                # 三角形
                if triangle_history:
                    tri_mode = self.get_most_common_value(triangle_history)
                    if tri_mode is not None and not self.reject_outlier_mode(tri_mode, triangle_history):
                        print(f"[等边三角形边长-全局众数] {tri_mode:.2f} cm")
                        self.print_signal.emit(f"[等边三角形边长-全局众数] {tri_mode:.2f} cm")
                # 正方形
                if square_history:
                    square_mode = self.get_most_common_value(square_history)
                    if square_mode is not None and not self.reject_outlier_mode(square_mode, square_history):
                        print(f"[正方形边长-全局众数] {square_mode:.2f} cm")
                        self.print_signal.emit(f"[正方形边长-全局众数] {square_mode:.2f} cm")

            # --- 叠加 HUD 信息（画面左上角） ---
            y = 24
            self.draw_text(vis, "Press 'q' to quit", (10, y), 0.6, (255, 255, 255), 2);
            y += 24
            if cm_per_px is not None:
                self.draw_text(vis, f"Scale ~ {cm_per_px:.4f} cm/px", (10, y), 0.6, (180, 255, 180), 2);
                y += 24
            if distance_history:
                dist_mode = self.get_most_common_value(distance_history)
                if dist_mode is not None and not self.reject_outlier_mode(dist_mode, distance_history):
                    self.draw_text(vis, f"Distance ~ {dist_mode:.2f} cm", (10, y), 0.6, (180, 255, 255), 2);
                    y += 24
            if circle_history:
                circle_mode = self.get_most_common_value(circle_history)
                if circle_mode is not None and not self.reject_outlier_mode(circle_mode, circle_history):
                    self.draw_text(vis, f"Circle dia ~ {circle_mode:.2f} cm", (10, y), 0.6, (200, 220, 255), 2);
                    y += 24
            if triangle_history:
                tri_mode = self.get_most_common_value(triangle_history)
                if tri_mode is not None and not self.reject_outlier_mode(tri_mode, triangle_history):
                    self.draw_text(vis, f"Equilateral side ~ {tri_mode:.2f} cm", (10, y), 0.6, (255, 220, 180), 2);
                    y += 24
            if square_history:
                square_mode = self.get_most_common_value(square_history)
                if square_mode is not None and not self.reject_outlier_mode(square_mode, square_history):
                    self.draw_text(vis, f"Square side ~ {square_mode:.2f} cm", (10, y), 0.6, (255, 200, 220), 2);
                    y += 24

            # 显示边缘图可选（按需开启）
            # edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            # cv2.imshow("Edges", edges_vis)

            # === 关键：显示窗口 ===
            # cv2.imshow(self.WINDOW_NAME, vis)

            # 转换为BGRA格式

            # 回传到主程序
            self.images_signal.emit(vis)
            # # 键盘事件：按 'q' 退出
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord('q'):
            #     print("收到退出指令，正在关闭...")
            #     break

            if self.stop_status.is_set():
                print("收到退出指令，正在关闭...")
                self.print_signal.emit("收到退出指令，正在关闭...")
                # 结束程序，恢复按钮
                self.msg_signal.emit(True)
                break

        cap.release()
        # cv2.destroyAllWindows()


# 线程池FH1
class Thread_FH1(QThread):
    # 定义一个自定义信号，用于向主线程发送消息
    images_signal = pyqtSignal(np.ndarray)
    msg_signal =  pyqtSignal(bool)
    print_signal =  pyqtSignal(str)


    PRINT_INTERVAL = 0.5
    AUTO_STOP_SECONDS = 5  # 自动停止秒数
    MAX_HISTORY_LEN = 20  # 全局滑动窗口长度（建议10~30之间）


    def __init__(self, stop_status,parent=None):
        super().__init__(parent)  # 必须调用父类初始化
        self.stop_status = stop_status


    def get_most_common_value(self,lst):
        if not lst: return None
        c = Counter(lst)
        return c.most_common(1)[0][0]

    def reject_outlier_mode(self,mode_val, vals, max_diff=0.2):
        if not vals or mode_val is None:
            return True  # 没值直接判为波动，不输出
        avg = sum(vals) / len(vals)
        return abs(mode_val - avg) > max_diff

    def append_to_history(self,history, val):
        history.append(val)
        while len(history) > self.MAX_HISTORY_LEN:
            history.popleft()

    def run(self):
        try:
            from utils1 import load_params
            params = load_params()
            distance_ref_cm = float(params["distance_ref_cm"])
            ref_metric = float(params["ref_metric"])
            metric_mode = params.get("metric_mode", "HEIGHT")
            from utils1 import CAM_INDEX,FRAME_W,FRAME_H
            cap = cv2.VideoCapture(CAM_INDEX)
            # cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

            square_history = deque()
            distance_history = deque()
            _last_print_ts = 0.0
            start_time = time.time()

            # try:
            while True:
                now = time.time()
                # 自动退出
                if now - start_time >= self.AUTO_STOP_SECONDS:
                    print(f"\n已自动运行 {self.AUTO_STOP_SECONDS} 秒，程序停止。")
                    self.print_signal.emit(f"\n已自动运行 {self.AUTO_STOP_SECONDS} 秒，程序停止。")
                    # 结束程序，恢复按钮
                    self.msg_signal.emit(True)
                    break

                ok, frame = cap.read()
                if not ok:
                    # # 也给窗口一个机会响应关闭
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     print("\n已按 q 退出。")
                    #     break
                    # 结束程序，恢复按钮
                    self.msg_signal.emit(True)
                    break
                    # continue

                # 复制一份用于可视化
                display_frame = frame.copy()

                # 寻找外接矩形
                from utils1 import find_contours
                edges, cnts = find_contours(frame)
                from utils1 import select_outer_rect
                rect_box = select_outer_rect(cnts, FRAME_W, FRAME_H)
                if not rect_box:
                    # 没找到时也显示原始画面，便于调试
                    cv2.putText(display_frame, "No outer rect found",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    # cv2.imshow("Detection", display_frame)

                    # 回传到主程序
                    self.images_signal.emit(display_frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     print("\n已按 q 退出。")
                    #     break

                    if self.stop_status.is_set():
                        print("收到退出指令，正在关闭...")
                        self.print_signal.emit("收到退出指令，正在关闭...")
                        # 结束程序，恢复按钮
                        self.msg_signal.emit(True)
                        break

                    # 周期性打印众数（即便没找到矩形）
                    if now - _last_print_ts >= self.PRINT_INTERVAL:
                        _last_print_ts = now
                        if distance_history:
                            dist_mode = self.get_most_common_value(distance_history)
                            if dist_mode is not None and not self.reject_outlier_mode(dist_mode, distance_history, 0.2):
                                print(f"[参考目标距离-全局众数] {dist_mode:.2f} cm")
                                self.print_signal.emit(f"[参考目标距离-全局众数] {dist_mode:.2f} cm")
                        if square_history:
                            mode_val = self.get_most_common_value(square_history)
                            if mode_val is not None and not self.reject_outlier_mode(mode_val, square_history, 0.2):
                                print(f"[最小正方形边长-全局众数] {mode_val:.2f} cm")
                                self.print_signal.emit(f"[最小正方形边长-全局众数] {mode_val:.2f} cm")
                    continue

                x, y, w, h = rect_box
                # 画出参考外接矩形
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                from utils1 import REAL_H_CM, REAL_W_CM, FRAME_H
                # 像素-厘米比例估计
                cm_per_px_y = REAL_H_CM / float(h)
                cm_per_px_x = REAL_W_CM / float(w)
                cm_per_px_avg = 0.5 * (cm_per_px_x + cm_per_px_y)
                from utils1 import compute_size_metric,compute_distance
                # 计算度量与距离
                curr_metric = compute_size_metric(rect_box, metric_mode)
                distance_cm = compute_distance(curr_metric, ref_metric, distance_ref_cm)
                self.append_to_history(distance_history, round(distance_cm, 2))

                # 边缘补偿
                from utils1 import EDGE_BIAS_AT_REF_PX,EDGE_BIAS_EXP,ROI_MARGIN_PX,MIN_SIDE_CM,MIN_CONTOUR_AREA
                scale = curr_metric / max(ref_metric, 1e-6)
                delta_px = EDGE_BIAS_AT_REF_PX * (1.0 / max(scale, 1e-3)) ** EDGE_BIAS_EXP
                edge_comp_cm = 2.0 * delta_px * cm_per_px_avg

                # ROI
                margin = max(ROI_MARGIN_PX, int(0.01 * min(w, h)))
                rx, ry = x + margin, y + margin
                rw, rh = w - 2 * margin, h - 2 * margin
                rx, ry = max(rx, 0), max(ry, 0)
                rw, rh = max(rw, 1), max(rh, 1)
                roi = frame[ry:ry + rh, rx:rx + rw]

                # 二值化与形态学
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                ks = 1 if min(rw, rh) < 220 else 2
                kernel = np.ones((ks, ks), np.uint8)
                th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, 1)

                extra = max(0, int(round((1.0 / max(scale, 1e-3)) - 1.0)))
                if extra > 0:
                    th = cv2.dilate(th, np.ones((1, 1), np.uint8), iterations=min(2, extra))

                cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                # 面积与边长阈值
                min_side_px = max(6, int(MIN_SIDE_CM / max(cm_per_px_avg, 1e-6)))
                min_area_px = max(MIN_CONTOUR_AREA, int(0.5 * min_side_px * min_side_px))

                squares = []
                for c in cnts:
                    if cv2.contourArea(c) < min_area_px:
                        continue
                    approx_s = cv2.approxPolyDP(c, 0.015 * cv2.arcLength(c, True), True)
                    from utils1 import is_square
                    if not is_square(approx_s):
                        continue
                    approx_g = approx_s + np.array([[[rx, ry]]], dtype=np.int32)  # 回到全图坐标
                    pts = approx_g.reshape(-1, 2).astype(np.float32)
                    wpx = np.linalg.norm(pts[0] - pts[1])
                    hpx = np.linalg.norm(pts[1] - pts[2])
                    side_cm_raw = 0.5 * (wpx * cm_per_px_x + hpx * cm_per_px_y)
                    side_cm = side_cm_raw + edge_comp_cm
                    if side_cm >= MIN_SIDE_CM:
                        squares.append((side_cm, approx_g))

                # 可视化检测到的正方形，并记录最小边长
                if squares:
                    # 最小边长的正方形
                    smin, amin = min(squares, key=lambda x: x[0])
                    self.append_to_history(square_history, round(smin, 2))
                    # 画全部正方形
                    for side_cm, pts in squares:
                        cv2.polylines(display_frame, [pts], True, (0, 0, 255), 2)
                        cv2.putText(display_frame, f"{side_cm:.2f}cm",
                                    tuple(pts[0][0]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # 叠加文字信息（当前距离与全局众数）
                overlay_lines = [f"Distance: {distance_cm:.2f} cm"]
                dist_mode = self.get_most_common_value(distance_history) if distance_history else None
                if dist_mode is not None and not self.reject_outlier_mode(dist_mode, distance_history, 0.2):
                    overlay_lines.append(f"Dist(mode): {dist_mode:.2f} cm")

                sq_mode = self.get_most_common_value(square_history) if square_history else None
                if sq_mode is not None and not self.reject_outlier_mode(sq_mode, square_history, 0.2):
                    overlay_lines.append(f"Square min side(mode): {sq_mode:.2f} cm")

                # 左上角绘制文本
                y0 = 30
                for i, txt in enumerate(overlay_lines):
                    cv2.putText(display_frame, txt, (10, y0 + i * 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # # 显示窗口
                # cv2.imshow("Detection", display_frame)

                # 回传到主程序
                self.images_signal.emit(display_frame)
                # 若想同时查看 ROI 或二值图，可取消下面注释：
                # cv2.imshow("ROI", roi)
                # cv2.imshow("ROI_Thresh", th)

                # # 键盘处理：q 退出
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     print("\n已按 q 退出。")
                #     break
                if self.stop_status.is_set():
                    print("收到退出指令，正在关闭...")
                    self.print_signal.emit("收到退出指令，正在关闭...")
                    # 结束程序，恢复按钮
                    self.msg_signal.emit(True)
                    break
                # 周期性打印终端输出（抗波动后的众数）
                if now - _last_print_ts >= self.PRINT_INTERVAL:
                    _last_print_ts = now
                    if distance_history:
                        dist_mode = self.get_most_common_value(distance_history)
                        if dist_mode is not None and not self.reject_outlier_mode(dist_mode, distance_history, 0.2):
                            print(f"[参考目标距离-全局众数] {dist_mode:.2f} cm")
                            self.print_signal.emit(f"[参考目标距离-全局众数] {dist_mode:.2f} cm")
                    if square_history:
                        mode_val = self.get_most_common_value(square_history)
                        if mode_val is not None and not self.reject_outlier_mode(mode_val, square_history, 0.2):
                                print(f"[最小正方形边长-全局众数] {mode_val:.2f} cm")
                                self.print_signal.emit(f"[最小正方形边长-全局众数] {mode_val:.2f} cm")

            # except KeyboardInterrupt:
            #     print("\n已手动退出。")
            # finally:
            #     cap.release()
            #     # cv2.destroyAllWindows()
            cap.release()
        except Exception as e:
            print(e)


# 线程池FH2
class Thread_FH2(QThread):
    # 定义一个自定义信号，用于向主线程发送消息
    images_signal = pyqtSignal(np.ndarray)
    msg_signal =  pyqtSignal(bool)
    print_signal =  pyqtSignal(str)


    def __init__(self, stop_status,parent=None):
        super().__init__(parent)  # 必须调用父类初始化
        self.stop_status = stop_status
    def run(self):
        from measurement_utils import load_params,CAM_INDEX,FRAME_W,FRAME_H,find_contours,select_outer_rect,get_rect_corners_from_cnt,order_points,compute_distance,warp_perspective,find_shortest_extended_edge,REAL_W_CM,REAL_H_CM
        params = load_params()
        distance_ref_cm = params["distance_ref_cm"]
        ref_metric = params["ref_metric"]

        # cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

        # 透视矫正后的目标尺寸
        WARP_W, WARP_H = 850, 1300

        print("📏 透视矫正 + 实时测量（5秒后自动退出，按 q 或 ESC 手动退出）")
        self.print_signal.emit("📏 透视矫正 + 实时测量（5秒后自动退出，按 q 或 ESC 手动退出）")
        start_time = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                # # 避免画面卡死，仍然响应键盘
                # if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                #     print("手动退出")
                # 结束程序，恢复按钮
                self.msg_signal.emit(True)
                break
                # continue

            H, W = frame.shape[:2]
            cnts = find_contours(frame)
            rect_box, rect_cnt = select_outer_rect(cnts, W, H)

            vis = frame.copy()
            distance_cm = None
            real_length_cm = None

            if rect_box and rect_cnt is not None:
                x, y, w, h = rect_box
                rect_corners = get_rect_corners_from_cnt(rect_cnt)
                rect_corners_ordered = order_points(rect_corners)

                # 绘制外轮廓
                cv2.polylines(vis, [rect_corners_ordered.astype(np.int32)], True, (0, 255, 0), 2)

                # 计算距离
                distance_cm = compute_distance(rect_box, ref_metric, distance_ref_cm)
                if distance_cm:
                    cv2.putText(vis, f"{distance_cm:.2f} cm", (x, max(30, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # 透视矫正
                warp, M = warp_perspective(frame, rect_corners_ordered, WARP_W, WARP_H)

                # 找到最短延伸边
                shortest_line, _, corner_points = find_shortest_extended_edge(warp)

                # 标注角点
                if corner_points is not None:
                    for i, p in enumerate(corner_points):
                        pt_src = cv2.perspectiveTransform(
                            np.array([[p]], dtype='float32'), np.linalg.inv(M)
                        )[0][0]
                        cv2.circle(vis, tuple(np.round(pt_src).astype(int)), 4, (0, 0, 255), -1)
                        cv2.putText(vis, str(i), tuple(np.round(pt_src + 5).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # 标注最短边
                if shortest_line is not None:
                    p1_src = cv2.perspectiveTransform(
                        np.array([[shortest_line[0]]], dtype='float32'), np.linalg.inv(M)
                    )[0][0]
                    p2_src = cv2.perspectiveTransform(
                        np.array([[shortest_line[1]]], dtype='float32'), np.linalg.inv(M)
                    )[0][0]

                    cv2.line(vis, tuple(np.round(p1_src).astype(int)), tuple(np.round(p2_src).astype(int)), (0, 255, 0),
                             2)

                    # 计算实际长度
                    px_len = np.linalg.norm(shortest_line[1] - shortest_line[0])
                    cm_per_px = (REAL_W_CM / WARP_W + REAL_H_CM / WARP_H) / 2
                    real_length_cm = px_len * cm_per_px

                    mid_pt = (shortest_line[0] + shortest_line[1]) // 2
                    mid_pt_src = cv2.perspectiveTransform(
                        np.array([[mid_pt]], dtype='float32'), np.linalg.inv(M)
                    )[0][0]

                    cv2.putText(vis, f"{real_length_cm:.2f} cm", tuple(np.round(mid_pt_src).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # 终端打印
            print_str = ""
            if distance_cm is not None:
                print_str += f"测距: {distance_cm:.2f} cm  "
            if real_length_cm is not None:
                print_str += f"最短边: {real_length_cm:.2f} cm"
            if print_str:
                print(print_str)
                self.print_signal.emit(print_str)
            # 显示窗口（缩放一半，避免过大）
            display_scale = 0.5
            vis_resized = cv2.resize(vis, (0, 0), fx=display_scale, fy=display_scale)
            # cv2.imshow("Distance + Shortest Length (Rectified)", vis_resized)
            self.images_signal.emit(vis_resized)

            # 自动退出
            if time.time() - start_time > 5:
                print("程序已自动退出（5秒到）")
                self.print_signal.emit("程序已自动退出（5秒到）")
                # 结束程序，恢复按钮
                self.msg_signal.emit(True)
                break

            # # 按 q / ESC 手动退出
            # key = cv2.waitKey(1) & 0xFF
            # if key in (27, ord('q')):
            #     print("手动退出")
            #     self.print_signal.emit("手动退出")
            #     break
            if self.stop_status.is_set():
                print("收到退出指令，正在关闭...")
                self.print_signal.emit("收到退出指令，正在关闭...")
                # 结束程序，恢复按钮
                self.msg_signal.emit(True)
                break

        cap.release()
        # cv2.destroyAllWindows()

# 线程池FH3A
class Thread_FH3A(QThread):
    # 定义一个自定义信号，用于向主线程发送消息
    images_signal = pyqtSignal(np.ndarray)
    msg_signal =  pyqtSignal(bool)
    btn_signal =  pyqtSignal(bool)
    print_signal =  pyqtSignal(str)


    def __init__(self, stop_status,input_key,parent=None):
        super().__init__(parent)  # 必须调用父类初始化
        self.stop_status = stop_status
        self.input_key = input_key



    def try_ocr_with_crop(self,roi, ocr, crop=10):
        """先直接识别，若不准则收缩边缘再识别一次"""
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_bin = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 7)
        _, buf = cv2.imencode('.png', roi_bin)
        result = ocr.classification(buf.tobytes()).strip()
        if not (result.isdigit() and len(result) == 1):
            h, w = roi_bin.shape[:2]
            crop = min(crop, w // 4, h // 4)
            if crop > 0 and w - 2 * crop > 0 and h - 2 * crop > 0:
                roi_crop = roi_bin[crop:h - crop, crop:w - crop]
                _, buf2 = cv2.imencode('.png', roi_crop)
                result2 = ocr.classification(buf2.tobytes()).strip()
                if result2.isdigit() and len(result2) == 1:
                    result = result2
        return result

    def run(self):
        self.print_signal.emit("请选择你要高亮的数字(0~9)")
        target_digit = self.input_key.get()  # 阻塞直到收到新值
        print(f"收到新 input_key: {target_digit}")
        self.print_signal.emit(f"所选数字为：{target_digit}")
        # ======== 1. 启动时输入目标数字 ========
        from utils import load_params,CAM_INDEX,FRAME_W,FRAME_H,find_contours,select_outer_rect,get_rect_corners_from_cnt,order_points,warp_perspective,find_valid_edges,build_square_from_edge,REAL_H_CM
        params = load_params()
        distance_ref_cm = float(params["distance_ref_cm"])  # 标定距离(cm)
        ref_height = float(params["ref_metric"])  # 标定时的纸张高度（像素）只用高度！
        # 不再考虑面积模式，不存在 area 标定

        # cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        WARP_W, WARP_H = 850, 1300

        ocr = ddddocr.DdddOcr(beta=True)
        self.btn_signal.emit(True)
        while True:
            ok, frame = cap.read()
            if not ok:
                # 结束程序，恢复按钮
                self.msg_signal.emit(True)
                break

            H, W = frame.shape[:2]
            cnts = find_contours(frame)
            rect_box, rect_cnt = select_outer_rect(cnts, W, H)
            vis = frame.copy()
            debug_digit_list = []
            highlight_boxes = []

            if rect_box and rect_cnt is not None:
                x, y, w, h = rect_box
                rect_corners = get_rect_corners_from_cnt(rect_cnt)
                rect_corners_ordered = order_points(rect_corners)
                cv2.polylines(vis, [rect_corners_ordered.astype(np.int32)], True, (0, 255, 0), 2)

                # HEIGHT模式距离测量，只用高度像素！
                curr_height = h
                if curr_height > 0:
                    distance_cm = distance_ref_cm * (ref_height / curr_height)
                    txt = f"{distance_cm:.2f} cm"
                    cv2.putText(vis, txt, (x, max(30, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    distance_cm = None

                warp, M = warp_perspective(frame, rect_corners_ordered, WARP_W, WARP_H)
                if M is None or np.abs(np.linalg.det(M)) < 1e-5:
                    continue

                try:
                    inv_M = np.linalg.inv(M)
                except np.linalg.LinAlgError:
                    continue

                valid_edges, corner_points = find_valid_edges(warp)
                used_points = set()
                roi_cnt = 0
                for edge, (p1, p2) in valid_edges:
                    if roi_cnt >= 8: break
                    key = tuple(sorted((tuple(p1), tuple(p2))))
                    if key in used_points:
                        continue
                    used_points.add(key)
                    square = build_square_from_edge(p1, p2, inward=False)

                    # warp后像素转厘米，只用HEIGHT逻辑
                    side_len_px = np.linalg.norm(square[0] - square[1])
                    cm_per_px = REAL_H_CM / WARP_H  # 只用标定高
                    side_len_cm = side_len_px * cm_per_px

                    # 投影到原图做高亮框
                    pts = np.array([square], dtype=np.float32)
                    pts_src = cv2.perspectiveTransform(pts, inv_M)[0]
                    pts_src_int = np.round(pts_src).astype(int)

                    # ROI处理和数字识别
                    mask = np.zeros((H, W), dtype=np.uint8)
                    cv2.fillPoly(mask, [pts_src_int], 255)
                    x1, y1, w1, h1 = cv2.boundingRect(pts_src_int)
                    if w1 < 10 or h1 < 10: continue
                    roi = cv2.bitwise_and(frame, frame, mask=mask)[y1:y1 + h1, x1:x1 + w1]

                    # 用ddddocr识别数字
                    if roi.shape[0] > 0 and roi.shape[1] > 0:
                        result_digit = self.try_ocr_with_crop(roi, ocr, crop=10)
                        roi_show = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (48, 48))
                        roi_show = cv2.cvtColor(roi_show, cv2.COLOR_GRAY2BGR)
                        cv2.putText(roi_show, result_digit, (3, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        debug_digit_list.append(roi_show)
                        # ==== 高亮目标数字 ====
                        if result_digit == target_digit:
                            highlight_boxes.append((pts_src_int, side_len_cm))
                    roi_cnt += 1

            # ==== 绘制高亮框和边长 ====
            for box_pts, side_len in highlight_boxes:
                cv2.polylines(vis, [box_pts], True, (0, 0, 255), 3)
                x_, y_ = box_pts[0][0], box_pts[0][1]
                cv2.putText(vis, f"L={side_len:.2f}cm", (x_, y_ - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # if debug_digit_list:
            #     debug_preview = np.hstack(debug_digit_list) if len(debug_digit_list) > 1 else debug_digit_list[0]
            #     cv2.imshow("Debug Digit", debug_preview)
            # else:
            #     cv2.imshow("Debug Digit", np.zeros((48, 48, 3), dtype=np.uint8))

            scale = 0.5
            vis_small = cv2.resize(vis, (0, 0), fx=scale, fy=scale)
            # cv2.imshow("Distance + Square", vis_small)
            # 回传到主程序
            self.images_signal.emit(vis_small)

            if self.stop_status.is_set():
                print("收到退出指令，正在关闭...")
                self.print_signal.emit("收到退出指令，正在关闭...")
                # 结束程序，恢复按钮
                self.msg_signal.emit(True)
                break

            # key = cv2.waitKey(1) & 0xFF
            # if key in (27, ord('q')):
            #     break

        cap.release()


# 线程池FH3B
class Thread_FH3B(QThread):
    # 定义一个自定义信号，用于向主线程发送消息
    images_signal = pyqtSignal(np.ndarray)
    msg_signal =  pyqtSignal(bool)
    btn_signal =  pyqtSignal(bool)
    print_signal =  pyqtSignal(str)

    PRINT_INTERVAL = 0.5
    AUTO_STOP_SECONDS = 66  # 自动停止秒数
    MAX_HISTORY_LEN = 40  # 全局滑动窗口长度（建议10~30之间）

    def __init__(self, stop_status,input_key,parent=None):
        super().__init__(parent)  # 必须调用父类初始化
        self.stop_status = stop_status
        self.input_key = input_key

    def get_most_common_value(self,lst):
        if not lst: return None
        c = Counter(lst)
        return c.most_common(1)[0][0]

    def reject_outlier_mode(self,mode_val, vals, max_diff=0.2):
        if not vals or mode_val is None:
            return True  # 没值直接判为波动，不输出
        avg = sum(vals) / len(vals)
        return abs(mode_val - avg) > max_diff

    def append_to_history(self,history, val):
        history.append(val)
        while len(history) > self.MAX_HISTORY_LEN:
            history.popleft()

    def try_ocr_with_crop(self,roi, ocr, crop=10):
        """先直接识别，若不准则收缩边缘再识别一次"""
        _, buf = cv2.imencode('.png', roi)
        result = ocr.classification(buf.tobytes()).strip()

        if not result.isdigit() or len(result) != 1:
            h, w = roi.shape[:2]
            crop = min(crop, w // 4, h // 4)
            if crop > 0 and w - 2 * crop > 0 and h - 2 * crop > 0:
                roi_crop = roi[crop:h - crop, crop:w - crop]
                _, buf2 = cv2.imencode('.png', roi_crop)
                result2 = ocr.classification(buf2.tobytes()).strip()
                if result2.isdigit() and len(result2) == 1:
                    result = result2
        return result


    def run(self):
        from utils1 import load_params,CAM_INDEX,FRAME_W,FRAME_H,find_contours,select_outer_rect,REAL_H_CM,REAL_W_CM,compute_size_metric,compute_distance,EDGE_BIAS_AT_REF_PX,EDGE_BIAS_EXP,ROI_MARGIN_PX,MIN_SIDE_CM,MIN_CONTOUR_AREA,is_square
        params = load_params()
        distance_ref_cm = float(params["distance_ref_cm"])
        ref_metric = float(params["ref_metric"])
        metric_mode = params.get("metric_mode", "HEIGHT")

        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

        square_history = deque()
        distance_history = deque()
        _last_print_ts = 0.0
        start_time = time.time()

        # 初始化 OCR
        ocr = ddddocr.DdddOcr(beta=True, show_ad=False)
        # target_digit = input("请输入要检测的数字：").strip()

        self.print_signal.emit("请选择你要高亮的数字(0~9)")
        target_digit = self.input_key.get()  # 阻塞直到收到新值
        print(f"收到新 input_key: {target_digit}")
        self.print_signal.emit(f"所选数字为：{target_digit}")


        try:
            while True:
                now = time.time()
                if now - start_time >= self.AUTO_STOP_SECONDS:
                    print(f"\n已自动运行 {self.AUTO_STOP_SECONDS} 秒，程序停止。")
                    self.print_signal.emit(f"\n已自动运行 {self.AUTO_STOP_SECONDS} 秒，程序停止。")
                    # 结束程序，恢复按钮
                    self.msg_signal.emit(True)
                    break

                ok, frame = cap.read()
                if not ok:
                    # 结束程序，恢复按钮
                    self.msg_signal.emit(True)
                    break

                edges, cnts = find_contours(frame)
                rect_box = select_outer_rect(cnts, FRAME_W, FRAME_H)
                if not rect_box:
                    # 回传到主程序
                    self.images_signal.emit(frame)
                    # cv2.imshow("frame", frame)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                    if self.stop_status.is_set():
                        print("收到退出指令，正在关闭...")
                        self.print_signal.emit("收到退出指令，正在关闭...")
                        # 结束程序，恢复按钮
                        self.msg_signal.emit(True)
                        break
                    continue

                x, y, w, h = rect_box

                cm_per_px_y = REAL_H_CM / float(h)
                cm_per_px_x = REAL_W_CM / float(w)
                cm_per_px_avg = 0.5 * (cm_per_px_x + cm_per_px_y)

                curr_metric = compute_size_metric(rect_box, metric_mode)
                distance_cm = compute_distance(curr_metric, ref_metric, distance_ref_cm)
                self.append_to_history(distance_history, round(distance_cm, 2))

                scale = curr_metric / max(ref_metric, 1e-6)
                delta_px = EDGE_BIAS_AT_REF_PX * (1.0 / max(scale, 1e-3)) ** EDGE_BIAS_EXP
                edge_comp_cm = 2.0 * delta_px * cm_per_px_avg

                margin = max(ROI_MARGIN_PX, int(0.01 * min(w, h)))
                rx, ry = x + margin, y + margin
                rw, rh = w - 2 * margin, h - 2 * margin
                rx, ry = max(rx, 0), max(ry, 0)
                rw, rh = max(rw, 1), max(rh, 1)
                roi = frame[ry:ry + rh, rx:rx + rw]

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                ks = 1 if min(rw, rh) < 220 else 2
                kernel = np.ones((ks, ks), np.uint8)
                th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, 1)

                extra = max(0, int(round((1.0 / max(scale, 1e-3)) - 1.0)))
                if extra > 0:
                    th = cv2.dilate(th, np.ones((1, 1), np.uint8), iterations=min(2, extra))

                cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                min_side_px = max(6, int(MIN_SIDE_CM / max(cm_per_px_avg, 1e-6)))
                min_area_px = max(MIN_CONTOUR_AREA, int(0.5 * min_side_px * min_side_px))

                squares = []
                for c in cnts:
                    if cv2.contourArea(c) < min_area_px:
                        continue
                    approx_s = cv2.approxPolyDP(c, 0.015 * cv2.arcLength(c, True), True)
                    if not is_square(approx_s):
                        continue
                    approx_g = approx_s + np.array([[[rx, ry]]], dtype=np.int32)
                    pts = approx_g.reshape(-1, 2).astype(np.float32)
                    wpx = np.linalg.norm(pts[0] - pts[1])
                    hpx = np.linalg.norm(pts[1] - pts[2])
                    side_cm_raw = 0.5 * (wpx * cm_per_px_x + hpx * cm_per_px_y)
                    side_cm = side_cm_raw + edge_comp_cm
                    if side_cm >= MIN_SIDE_CM:
                        squares.append((side_cm, approx_g))

                # 检测数字并绘制
                if squares:
                    for side_cm, box in squares:
                        pts = box.reshape(-1, 2).astype(np.int32)
                        x0, y0 = np.min(pts, axis=0)
                        x1, y1 = np.max(pts, axis=0)
                        x0, y0 = max(int(x0), 0), max(int(y0), 0)
                        x1, y1 = min(int(x1), frame.shape[1]), min(int(y1), frame.shape[0])

                        roi_digit = frame[y0:y1, x0:x1]
                        if roi_digit.shape[0] > 0 and roi_digit.shape[1] > 0:
                            result_digit = self.try_ocr_with_crop(roi_digit, ocr, crop=10)

                            if result_digit == target_digit:
                                print(f"检测到目标数字 {result_digit}，边长为 {side_cm:.2f} cm")
                                self.append_to_history(square_history, round(side_cm, 2))

                                # 在画面上画框和数字
                                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                                cv2.putText(frame, f"{result_digit} ({side_cm:.1f}cm)",
                                            (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.8, (0, 255, 0), 2, cv2.LINE_AA)

                # 定期打印全局统计
                if now - _last_print_ts >= self.PRINT_INTERVAL:
                    _last_print_ts = now

                    if distance_history:
                        dist_mode = self.get_most_common_value(distance_history)
                        if dist_mode is not None and not self.reject_outlier_mode(dist_mode, distance_history, 0.2):
                            print(f"[参考目标距离-全局众数] {dist_mode:.2f} cm")

                    if square_history:
                        mode_val = self.get_most_common_value(square_history)
                        if mode_val is not None and not self.reject_outlier_mode(mode_val, square_history, 0.2):
                            print(f"[目标数字正方形边长-全局众数] {mode_val:.2f} cm")

                # # 显示窗口
                # cv2.imshow("frame", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                # 回传到主程序
                self.images_signal.emit(frame)

                if self.stop_status.is_set():
                    print("收到退出指令，正在关闭...")
                    self.print_signal.emit("收到退出指令，正在关闭...")
                    # 结束程序，恢复按钮
                    self.msg_signal.emit(True)
                    break


        except KeyboardInterrupt:
            print("\n已手动退出。")
        finally:
            cap.release()
            # cv2.destroyAllWindows()

# 线程池FH4
class Thread_FH4(QThread):
    # 定义一个自定义信号，用于向主线程发送消息
    images_signal = pyqtSignal(np.ndarray)
    msg_signal =  pyqtSignal(bool)
    print_signal =  pyqtSignal(str)

    # 实际矩形框宽高
    REAL_W, REAL_H = 20, 31  # 改成你的实际宽和高
    RECT_W, RECT_H = 200, 310  # 校正输出尺寸，比例一致即可



    def __init__(self, stop_status,parent=None):
        super().__init__(parent)  # 必须调用父类初始化
        self.stop_status = stop_status
    def run(self):
        # cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        while True:
            ret, img = cap.read()
            if not ret:
                # 结束程序，恢复按钮
                self.msg_signal.emit(True)
                break

            display = img.copy()
            rect_pts = self.find_largest_rect(img)
            ratio, real_area_square, edge_length = 0, 0, 0

            if rect_pts is not None:
                rect_pts = self.order_points(rect_pts)
                dst_pts = np.array([[0, 0], [self.RECT_W - 1, 0], [self.RECT_W - 1, self.RECT_H - 1], [0, self.RECT_H - 1]],
                                   dtype="float32")
                M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
                warp = cv2.warpPerspective(img, M, (self.RECT_W, self.RECT_H))

                gray_warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
                h, w = gray_warp.shape
                roi = gray_warp[h // 5: h * 4 // 5, w // 5: w * 4 // 5]
                _, mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                square_area = 0
                square_rect = None
                for c in contours:
                    area = cv2.contourArea(c)
                    x, y, ww, hh = cv2.boundingRect(c)
                    if 0.1 * w * w < area < 0.7 * w * w and 0.7 < ww / hh < 1.3:
                        square_area = area
                        square_rect = (x, y, ww, hh)
                        cv2.rectangle(warp, (w // 5 + x, h // 5 + y), (w // 5 + x + ww, h // 5 + y + hh), (255, 0, 0),
                                      2)
                        break

                total_area = self.RECT_W * self.RECT_H
                ratio = square_area / total_area if total_area > 0 else 0
                real_area_total = self.REAL_W * self.REAL_H
                real_area_square = ratio * real_area_total
                edge_length = np.sqrt(real_area_square) if real_area_square > 0 else 0

                # ✅ 终端输出
                print(f"比例: {ratio:.3f} | 方形面积: {real_area_square:.2f} cm² | 边长: {edge_length:.2f} cm")
                self.print_signal.emit(f"比例: {ratio:.3f} | 方形面积: {real_area_square:.2f} cm² | 边长: {edge_length:.2f} cm")
                # 画面标注
                text = f"Ratio: {ratio:.3f} | Edge: {edge_length:.2f}"
                cv2.putText(display, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                cv2.polylines(display, [rect_pts.astype(int)], True, (0, 255, 0), 2)
                # cv2.imshow("Warped Rectified", warp)
            else:
                cv2.putText(display, "No rect detected!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

            scale = 0.5
            display_small = cv2.resize(display, (0, 0), fx=scale, fy=scale)
            # cv2.imshow("Camera", display_small)
            # 回传到主程序
            self.images_signal.emit(display_small)
            # if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            #     break
            if self.stop_status.is_set():
                print("收到退出指令，正在关闭...")
                self.print_signal.emit("收到退出指令，正在关闭...")
                # 结束程序，恢复按钮
                self.msg_signal.emit(True)
                break

        cap.release()


    def find_largest_rect(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 50, 150)
        cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                return np.array(approx).reshape(4, 2)
        return None

    def order_points(self,pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):  # 构造方法
        super(MainWindow, self).__init__()  # 运行父类的构造方法
        self.setupUi(self)
        # 键入数字
        self.input_key = Queue()
        # 停止标识符
        self.JC1_stop_status = None
        self.FH1_stop_status = None
        self.FH2_stop_status = None
        self.FH3A_stop_status = None
        self.FH3B_stop_status = None
        self.FH4_stop_status = None
        # 绑定
        self.pushButton_JC1.clicked.connect(self.JC1_fun)
        self.pushButton_FH1.clicked.connect(self.FH1_fun)
        self.pushButton_FH2.clicked.connect(self.FH2_fun)
        self.pushButton_FH3A.clicked.connect(self.FH3A_fun)
        self.pushButton_FH3B.clicked.connect(self.FH3B_fun)
        self.pushButton_FH4.clicked.connect(self.FH4_fun)
        self.pushButton_0.clicked.connect(self.keyboard_fun)
        self.pushButton_1.clicked.connect(self.keyboard_fun)
        self.pushButton_2.clicked.connect(self.keyboard_fun)
        self.pushButton_3.clicked.connect(self.keyboard_fun)
        self.pushButton_4.clicked.connect(self.keyboard_fun)
        self.pushButton_5.clicked.connect(self.keyboard_fun)
        self.pushButton_6.clicked.connect(self.keyboard_fun)
        self.pushButton_7.clicked.connect(self.keyboard_fun)
        self.pushButton_8.clicked.connect(self.keyboard_fun)
        self.pushButton_9.clicked.connect(self.keyboard_fun)
        # 创建定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ina226)
        self.timer.start(500)  # 5000毫秒=5秒
        self.max_power_val = 0.0

    def update_ina226(self):
        curr, pwr, pwr_float = sensor_interface.read_ina226_current_power()
        self.label_current_var.setText(f"{curr}")
        self.label_power_var.setText(f"{pwr}")
        if pwr_float > self.max_power_val:
            self.max_power_val = pwr_float
        if self.max_power_val > 0:
            self.label_max_power_var.setText(f"{self.max_power_val:.3f}")
        else:
            self.label_max_power_var.setText("--")


    # 键盘输入
    def keyboard_fun(self):
        sender = self.sender()
        # 键入数字
        self.input_key.put(sender.objectName()[-1])

    def update_image(self, frame):
        """更新 QLabel 显示"""
        # 转换为BGRA格式
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        # 显示图像
        QtImg = QImage(frame.data, frame.shape[1], frame.shape[0],
                       QImage.Format_RGB32)
        pixmap = QPixmap.fromImage(QtImg)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))


    """JC1 程序"""

    def JC1_fun(self):
        """JC1 按钮"""
        sender = self.sender()
        if sender.text() == "停止":
            sender.setText("JC1")
            # 停止
            self.JC1_stop_status.set()
        else:
            sender.setText("停止")
            # 清空输出
            self.textEdit.clear()
            # 停止标识符
            self.JC1_stop_status = threading.Event()
            # 禁用其他按钮
            self.pushButton_FH1.setEnabled(False)
            self.pushButton_FH2.setEnabled(False)
            self.pushButton_FH3A.setEnabled(False)
            self.pushButton_FH3B.setEnabled(False)
            self.pushButton_FH4.setEnabled(False)
            # 多线程
            self.thread = Thread_JC1(self.JC1_stop_status)
            self.thread.images_signal.connect(self.update_image)
            self.thread.msg_signal.connect(self.JC1_signal)
            self.thread.print_signal.connect(lambda x:self.textEdit.append(x))
            self.thread.start()


    def JC1_signal(self,msg):
        if msg:

            # 停止
            self.JC1_stop_status.set()
            self.pushButton_JC1.setText("JC1")
            # 恢复其他按钮
            self.pushButton_FH1.setEnabled(True)
            self.pushButton_FH2.setEnabled(True)
            self.pushButton_FH3A.setEnabled(True)
            self.pushButton_FH3B.setEnabled(True)
            self.pushButton_FH4.setEnabled(True)
            self.image_label.setText("暂无图像")



    """FH1 程序"""
    def FH1_fun(self):
        """FH1 按钮"""
        sender = self.sender()
        if sender.text() == "停止":
            sender.setText("FH1")
            # 停止
            self.FH1_stop_status.set()
        else:
            sender.setText("停止")
            # 清空输出
            self.textEdit.clear()
            # 停止标识符
            self.FH1_stop_status = threading.Event()
            # 禁用其他按钮
            self.pushButton_JC1.setEnabled(False)
            self.pushButton_FH2.setEnabled(False)
            self.pushButton_FH3A.setEnabled(False)
            self.pushButton_FH3B.setEnabled(False)
            self.pushButton_FH4.setEnabled(False)
            # 多线程
            self.thread = Thread_FH1(self.FH1_stop_status)
            self.thread.images_signal.connect(self.update_image)
            self.thread.msg_signal.connect(self.FH1_signal)
            self.thread.print_signal.connect(lambda x:self.textEdit.append(x))
            self.thread.start()


    def FH1_signal(self,msg):
        if msg:

            # 停止
            self.FH1_stop_status.set()
            self.pushButton_FH1.setText("FH1")
            # 恢复其他按钮
            self.pushButton_JC1.setEnabled(True)
            self.pushButton_FH2.setEnabled(True)
            self.pushButton_FH3A.setEnabled(True)
            self.pushButton_FH3B.setEnabled(True)
            self.pushButton_FH4.setEnabled(True)
            self.image_label.setText("暂无图像")


    """FH2 程序"""
    def FH2_fun(self):
        """FH2 按钮"""
        sender = self.sender()
        if sender.text() == "停止":
            sender.setText("FH2")
            # 停止
            self.FH2_stop_status.set()
        else:
            sender.setText("停止")
            # 清空输出
            self.textEdit.clear()
            # 停止标识符
            self.FH2_stop_status = threading.Event()
            # 禁用其他按钮
            self.pushButton_JC1.setEnabled(False)
            self.pushButton_FH1.setEnabled(False)
            self.pushButton_FH3A.setEnabled(False)
            self.pushButton_FH3B.setEnabled(False)
            self.pushButton_FH4.setEnabled(False)
            # 多线程
            self.thread = Thread_FH2(self.FH2_stop_status)
            self.thread.images_signal.connect(self.update_image)
            self.thread.msg_signal.connect(self.FH2_signal)
            self.thread.print_signal.connect(lambda x:self.textEdit.append(x))
            self.thread.start()


    def FH2_signal(self,msg):
        if msg:
            # 停止
            self.FH2_stop_status.set()
            self.pushButton_FH2.setText("FH2")
            # 恢复其他按钮
            self.pushButton_JC1.setEnabled(True)
            self.pushButton_FH1.setEnabled(True)
            self.pushButton_FH3A.setEnabled(True)
            self.pushButton_FH3B.setEnabled(True)
            self.pushButton_FH4.setEnabled(True)
            self.image_label.setText("暂无图像")


    """FH3A 程序"""
    def FH3A_fun(self):
        """FH3A 按钮"""
        sender = self.sender()
        if sender.text() == "停止":
            sender.setText("FH3A")
            # 停止
            self.FH3A_stop_status.set()
        else:
            sender.setText("停止")
            # 清空输出
            self.textEdit.clear()
            # 停止标识符
            self.FH3A_stop_status = threading.Event()
            # 禁用其他按钮
            self.pushButton_JC1.setEnabled(False)
            self.pushButton_FH1.setEnabled(False)
            self.pushButton_FH2.setEnabled(False)
            self.pushButton_FH3B.setEnabled(False)
            self.pushButton_FH4.setEnabled(False)
            self.pushButton_FH3A.setEnabled(False)
            # 多线程
            self.thread = Thread_FH3A(self.FH3A_stop_status,self.input_key)
            self.thread.images_signal.connect(self.update_image)
            self.thread.btn_signal.connect(lambda x:self.pushButton_FH3A.setEnabled(x))
            self.thread.msg_signal.connect(self.FH3A_signal)
            self.thread.print_signal.connect(lambda x:self.textEdit.append(x))
            self.thread.start()


    def FH3A_signal(self,msg):
        if msg:
            # 停止
            self.FH3A_stop_status.set()
            self.pushButton_FH3A.setText("FH3A")
            # 恢复其他按钮
            self.pushButton_JC1.setEnabled(True)
            self.pushButton_FH1.setEnabled(True)
            self.pushButton_FH2.setEnabled(True)
            self.pushButton_FH3B.setEnabled(True)
            self.pushButton_FH4.setEnabled(True)
            self.image_label.setText("暂无图像")

    """FH3B 程序"""
    def FH3B_fun(self):
        """FH3B 按钮"""
        sender = self.sender()
        if sender.text() == "停止":
            sender.setText("FH3B")
            # 停止
            self.FH3B_stop_status.set()
        else:
            sender.setText("停止")
            # 清空输出
            self.textEdit.clear()
            # 停止标识符
            self.FH3B_stop_status = threading.Event()
            # 禁用其他按钮
            self.pushButton_JC1.setEnabled(False)
            self.pushButton_FH1.setEnabled(False)
            self.pushButton_FH2.setEnabled(False)
            self.pushButton_FH3A.setEnabled(False)
            self.pushButton_FH4.setEnabled(False)
            # 多线程
            self.thread = Thread_FH3B(self.FH3B_stop_status,self.input_key)
            self.thread.images_signal.connect(self.update_image)
            self.thread.msg_signal.connect(self.FH3B_signal)
            self.thread.print_signal.connect(lambda x:self.textEdit.append(x))
            self.thread.start()


    def FH3B_signal(self,msg):
        if msg:
            # 停止
            self.FH3B_stop_status.set()
            self.pushButton_FH3B.setText("FH3B")
            # 恢复其他按钮
            self.pushButton_JC1.setEnabled(True)
            self.pushButton_FH1.setEnabled(True)
            self.pushButton_FH2.setEnabled(True)
            self.pushButton_FH3A.setEnabled(True)
            self.pushButton_FH4.setEnabled(True)
            self.image_label.setText("暂无图像")
    """FH4 程序"""
    def FH4_fun(self):
        """FH4 按钮"""
        sender = self.sender()
        if sender.text() == "停止":
            sender.setText("FH4")
            # 停止
            self.FH4_stop_status.set()
        else:
            sender.setText("停止")
            # 清空输出
            self.textEdit.clear()
            # 停止标识符
            self.FH4_stop_status = threading.Event()
            # 禁用其他按钮
            self.pushButton_JC1.setEnabled(False)
            self.pushButton_FH1.setEnabled(False)
            self.pushButton_FH2.setEnabled(False)
            self.pushButton_FH3B.setEnabled(False)
            self.pushButton_FH3A.setEnabled(False)
            # 多线程
            self.thread = Thread_FH4(self.FH4_stop_status)
            self.thread.images_signal.connect(self.update_image)
            self.thread.msg_signal.connect(self.FH4_signal)
            self.thread.print_signal.connect(lambda x:self.textEdit.append(x))
            self.thread.start()


    def FH4_signal(self,msg):
        if msg:
            # 停止
            self.FH4_stop_status.set()
            self.pushButton_FH3A.setText("FH4")
            # 恢复其他按钮
            self.pushButton_JC1.setEnabled(True)
            self.pushButton_FH1.setEnabled(True)
            self.pushButton_FH2.setEnabled(True)
            self.pushButton_FH3B.setEnabled(True)
            self.pushButton_FH3A.setEnabled(True)
            self.image_label.setText("暂无图像")

# 入口函数
if __name__ == '__main__':
    # 分辨率自适应
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
