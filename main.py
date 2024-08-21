import cv2

import numpy as np

from ultralytics import YOLO, solutions
from ultralytics.solutions import heatmap

def splitFrame(frame):
    cropped_frames = []
    height, width, _ = frame.shape
    for i in range(0, height, 1280):
        row = []
        for j in range(0, width, 1280):
            row.append(np.ascontiguousarray(frame[i:i+1280, j:j+1280]))
        cropped_frames.append(row)
    return cropped_frames

def get_object_coordinates(results):
    center_coordinates = []
    for result in results: 
        if result.boxes: 
            for box in result.boxes: 
                x1, y1, x2, y2 = box.xyxy[0] 
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                center_coordinates.append([center_x, center_y])
    return center_coordinates

def count_coordinates_in_grid(coordinates, image_width, image_height):
    grid_rows = image_height // 160 
    grid_cols = image_width // 160
    grid_square_width = image_width / grid_cols
    grid_square_height = image_height / grid_rows
    grid_counts = np.zeros((grid_rows, grid_cols), dtype=int)
    for x, y in coordinates:
        row_index = int(y // grid_square_height)
        col_index = int(x // grid_square_width)
        grid_counts[row_index, col_index] += 1

    return grid_counts

model = YOLO("head.pt") # https://github.com/AbelKidaneHaile/Reports
cap = cv2.VideoCapture("people2.webm")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

line_points = [(20, 400), (1080, 400)]
classes_to_count = [0]

# Initialize the heatmap object
heatmap_obj = heatmap.Heatmap(
    names=model.names,
    colormap=cv2.COLORMAP_JET,  # You can change the colormap as needed
    imw=640,  # Width of your input image/video
    imh=480,  # Height of your input image/video
    view_img=True,
    shape="circle",
    decay_factor=0,
    heatmap_alpha=1
)

# Video writer
# video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 1280))

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    im0Split = splitFrame(im0);
    
    tracks = model.track(im0Split[1][1], persist=True, show=False)
    
    heatmap_array = count_coordinates_in_grid(get_object_coordinates(tracks), 1280, 880)
    
    heatmap_normalised = (heatmap_array - np.min(heatmap_array)) / (np.max(heatmap_array) - np.min(heatmap_array))
    
    heatmap_coloured = cv2.applyColorMap((heatmap_normalised * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    
    heatmap_coloured_resized = cv2.resize(heatmap_coloured, (1280, 880), interpolation=cv2.INTER_LINEAR)
    
    combined_frame = cv2.addWeighted(im0Split[1][1], 0.5, heatmap_coloured_resized, 0.5, 0)
    
    cv2.imshow("show", combined_frame)
    cv2.waitKey(1)
    # video_writer.write(combined_frame)

cap.release()
# video_writer.release()
cv2.destroyAllWindows()