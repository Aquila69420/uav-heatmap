import cv2

import numpy as np

from ultralytics import YOLO, solutions

def splitFrame(frame):
    cropped_frames = []
    height, width, _ = frame.shape
    for i in range(0, height, 1280):
        row = []
        for j in range(0, width, 1280):
            row.append(np.ascontiguousarray(frame[i:i+1280, j:j+1280]))
        cropped_frames.append(row)
    return cropped_frames

model = YOLO("head.pt") # https://github.com/AbelKidaneHaile/Reports
cap = cv2.VideoCapture("people.webm")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

line_points = [(20, 400), (1080, 400)]
classes_to_count = [0] 

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 1280))

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=line_points,
    names=model.names,
    draw_tracks=True,
    line_thickness=2,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    im0Split = splitFrame(im0);
    
    tracks = model.track(im0Split[0][0], persist=True, show=False)

    im0 = counter.start_counting(im0Split[0][0], tracks)
    video_writer.write(im0Split[0][0])

cap.release()
video_writer.release()
cv2.destroyAllWindows()