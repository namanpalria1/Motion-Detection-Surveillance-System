import cv2
import imutils
import time

cap = cv2.VideoCapture("E:/desktop/car_survillence_system/video.mp4")
time.sleep(2)
first_frame = None

SKIP_FRAMES = 1  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    frame_delta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1500:              continue

        (x, y, w, h) = cv2.boundingRect(contour)

        aspect_ratio = w / float(h)
        if aspect_ratio < 1.2 or aspect_ratio > 4:  
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Vehicle Detected", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print("Moving vehicle detected!")

    cv2.imshow("Parking Surveillance", frame)

    key = cv2.waitKey(15) & 0xFF
    if key == ord('q'):
        break

    for _ in range(SKIP_FRAMES - 1):
        cap.read()

cap.release()
cv2.destroyAllWindows()

