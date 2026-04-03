import cv2
import numpy as np
import time
from measurement_utils import *

def main():
    params = load_params()
    distance_ref_cm = params["distance_ref_cm"]
    ref_metric = params["ref_metric"]

    # cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap = cv2.VideoCapture(CAM_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    # 透视矫正后的目标尺寸
    WARP_W, WARP_H = 850, 1300

    print("📏 透视矫正 + 实时测量（5秒后自动退出，按 q 或 ESC 手动退出）")

    start_time = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            # 避免画面卡死，仍然响应键盘
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                print("手动退出")
                break
            continue

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

                cv2.line(vis, tuple(np.round(p1_src).astype(int)), tuple(np.round(p2_src).astype(int)), (0, 255, 0), 2)

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

        # 显示窗口（缩放一半，避免过大）
        display_scale = 0.5
        vis_resized = cv2.resize(vis, (0, 0), fx=display_scale, fy=display_scale)
        cv2.imshow("Distance + Shortest Length (Rectified)", vis_resized)

        # 自动退出
        if time.time() - start_time > 5:
            print("程序已自动退出（5秒到）")
            break

        # 按 q / ESC 手动退出
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            print("手动退出")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
