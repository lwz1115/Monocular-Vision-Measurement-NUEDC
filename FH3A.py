import cv2
import numpy as np
import ddddocr
from utils import *

def try_ocr_with_crop(roi, ocr, crop=10):
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

def main():
    # ======== 1. 启动时输入目标数字 ========
    try:
        target_digit = input("请输入你要高亮的数字(0~9): ").strip()
        assert target_digit.isdigit() and 0 <= int(target_digit) <= 9
    except Exception:
        print("输入错误，请输入0-9之间的数字")
        return

    params = load_params()
    distance_ref_cm = float(params["distance_ref_cm"])    # 标定距离(cm)
    ref_height = float(params["ref_metric"])              # 标定时的纸张高度（像素）只用高度！
    # 不再考虑面积模式，不存在 area 标定

    # cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    WARP_W, WARP_H = 850, 1300

    ocr = ddddocr.DdddOcr(beta=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

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
            cv2.polylines(vis, [rect_corners_ordered.astype(np.int32)], True, (0,255,0), 2)
            
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
                cm_per_px = REAL_H_CM / WARP_H   # 只用标定高
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
                roi = cv2.bitwise_and(frame, frame, mask=mask)[y1:y1+h1, x1:x1+w1]

                # 用ddddocr识别数字
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    result_digit = try_ocr_with_crop(roi, ocr, crop=10)
                    roi_show = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (48, 48))
                    roi_show = cv2.cvtColor(roi_show, cv2.COLOR_GRAY2BGR)
                    cv2.putText(roi_show, result_digit, (3, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    debug_digit_list.append(roi_show)
                    # ==== 高亮目标数字 ====
                    if result_digit == target_digit:
                        highlight_boxes.append((pts_src_int, side_len_cm))
                roi_cnt += 1

        # ==== 绘制高亮框和边长 ====
        for box_pts, side_len in highlight_boxes:
            cv2.polylines(vis, [box_pts], True, (0,0,255), 3)
            x_, y_ = box_pts[0][0], box_pts[0][1]
            cv2.putText(vis, f"L={side_len:.2f}cm", (x_, y_-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        if debug_digit_list:
            debug_preview = np.hstack(debug_digit_list) if len(debug_digit_list) > 1 else debug_digit_list[0]
            cv2.imshow("Debug Digit", debug_preview)
        else:
            cv2.imshow("Debug Digit", np.zeros((48, 48, 3), dtype=np.uint8))

        scale = 0.5
        vis_small = cv2.resize(vis, (0, 0), fx=scale, fy=scale)
        cv2.imshow("Distance + Square", vis_small)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
