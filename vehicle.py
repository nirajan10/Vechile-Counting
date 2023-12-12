import cv2
import numpy as np

# Importing video
cap = cv2.VideoCapture("./video/traffic_-_27260.mp4")          

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

count_line_position = height - 190  # counter line for vehicle
min_width_rect = int(width * 0.05)  # min width rectangle
min_height_rect = int(height * 0.05)  # min height rectangle


# Initializing detector algorithm
algo = cv2.createBackgroundSubtractorMOG2()


def center_handle(x, y, w, h):
    cx = int((x + x + w) / 2)
    cy = int((y + y + h) / 2)
    return (cx, cy)


detect = []
offset = 12  # Allowable error between pixels (due to low fps)
counter = 0


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Applying algorithm on each frame
    img_sub = algo.apply(blur)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened1 = cv2.morphologyEx(img_sub, cv2.MORPH_OPEN, kernel1)

    _, thres_img = cv2.threshold(opened1, 50, 255, cv2.THRESH_BINARY)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8 , 8))
    opened2 = cv2.morphologyEx(thres_img, cv2.MORPH_OPEN, kernel2)

    dilate1 = cv2.dilate(opened2, np.ones((5,5)))
    dilate2 = cv2.dilate(dilate1, np.ones((5,5)))

    counterShape, h = cv2.findContours(dilate2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame, (25, count_line_position), (width - 25, count_line_position), (0, 0, 255), 1)

    for (i, c) in enumerate(counterShape):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_rect) and (h >= min_height_rect)
        if not validate_counter:
            continue
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'VEHICLE: {str(counter)}', (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 0, 0), 2)

        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)
        print(detect)

    temp_detect = detect.copy()
    for pt in temp_detect:
        if pt[1]<(count_line_position + offset) and pt[1]>(count_line_position - offset):
            counter += 1
            cv2.line(frame, (25, count_line_position), (width - 25, count_line_position), (255, 0, 0), 1)
        detect.remove(pt)
        print(f'Vehicle Counter: {str(counter)}')

    cv2.putText(frame, f'VEHICLE COUNTER: {str(counter)}', (width - 400, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Traffic video", frame)

    key = cv2.waitKey(fps)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()



    