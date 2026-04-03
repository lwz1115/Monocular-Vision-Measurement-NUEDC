import cv2
import numpy as np

# 实际矩形框宽高
REAL_W, REAL_H = 20, 31    # 改成你的实际宽和高
RECT_W, RECT_H = 200, 310  # 校正输出尺寸，比例一致即可

def find_largest_rect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return np.array(approx).reshape(4,2)
    return None

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

while True:
    ret, img = cap.read()
    if not ret:
        continue

    display = img.copy()
    rect_pts = find_largest_rect(img)
    ratio, real_area_square, edge_length = 0, 0, 0

    if rect_pts is not None:
        rect_pts = order_points(rect_pts)
        dst_pts = np.array([[0,0],[RECT_W-1,0],[RECT_W-1,RECT_H-1],[0,RECT_H-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
        warp = cv2.warpPerspective(img, M, (RECT_W, RECT_H))

        gray_warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
        h, w = gray_warp.shape
        roi = gray_warp[h//5: h*4//5, w//5: w*4//5]
        _, mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        square_area = 0
        square_rect = None
        for c in contours:
            area = cv2.contourArea(c)
            x, y, ww, hh = cv2.boundingRect(c)
            if 0.1*w*w < area < 0.7*w*w and 0.7 < ww/hh < 1.3:
                square_area = area
                square_rect = (x, y, ww, hh)
                cv2.rectangle(warp, (w//5+x, h//5+y), (w//5+x+ww, h//5+y+hh), (255,0,0), 2)
                break

        total_area = RECT_W * RECT_H
        ratio = square_area / total_area if total_area > 0 else 0
        real_area_total = REAL_W * REAL_H
        real_area_square = ratio * real_area_total
        edge_length = np.sqrt(real_area_square) if real_area_square > 0 else 0

        # ✅ 终端输出
        print(f"比例: {ratio:.3f} | 方形面积: {real_area_square:.2f} cm² | 边长: {edge_length:.2f} cm")

        # 画面标注
        text = f"Ratio: {ratio:.3f} | Edge: {edge_length:.2f}"
        cv2.putText(display, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3)
        cv2.polylines(display, [rect_pts.astype(int)], True, (0,255,0), 2)
        cv2.imshow("Warped Rectified", warp)
    else:
        cv2.putText(display, "No rect detected!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3)

    scale = 0.5
    display_small = cv2.resize(display, (0, 0), fx=scale, fy=scale)
    cv2.imshow("Camera", display_small)

    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
