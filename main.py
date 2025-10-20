import cv2
cap = cv2.VideoCapture("highway.mp4")

# inisialisasi background subtractor
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[340:720, 500:800]  # Dimensi lebar dari 500 - 800 & tinggi dari 340 - 720

    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # a

    for cnt in contours:  # b
        area = cv2.contourArea(cnt)
        if area > 100:  # c
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)  # c
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(30)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
