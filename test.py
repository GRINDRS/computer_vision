import cv2
from ultralytics import YOLO
import os
import tkinter as tk
from tkinter import simpledialog
import time
import sys

def main():
    # Load YOLO model
    try:
        model = YOLO('yolo11l.pt')
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        sys.exit(1)
    
    # Known object widths (in cm)
    known_widths = {
        "person": 50.0, "bottle": 7.0, "book": 15.0,
        "cell phone": 7.0, "laptop": 33.0, "tv": 100.0
    }
    FOCAL_LENGTH = 800
    
    # Setup output directory
    output_dir = os.path.expanduser("~/Desktop/Video_Output")
    os.makedirs(output_dir, exist_ok=True)
    temp_output_path = os.path.join(output_dir, "temp_output.mp4")
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    
    # Video writer setup
    out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                          fps, (frame_width, frame_height))
    
    # Exit button coordinates
    EXIT_BUTTON = {
        'x1': frame_width - 160,
        'y1': 20,
        'x2': frame_width - 20,
        'y2': 60
    }
    
    def check_exit_button(x, y):
        return (EXIT_BUTTON['x1'] <= x <= EXIT_BUTTON['x2'] and 
                EXIT_BUTTON['y1'] <= y <= EXIT_BUTTON['y2'])
    
    def on_mouse(event, x, y, flags, param):
        print(f"Mouse event: {event}, Coordinates: ({x}, {y})")
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Left mouse button clicked")
            if check_exit_button(x, y):
                print("EXIT BUTTON DETECTED!")
                return True
        return False
    
    cv2.namedWindow("YOLO + Distance")
    cv2.setMouseCallback("YOLO + Distance", on_mouse)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            results = model(frame)
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0].item())
                label = model.names[class_id]
                conf = float(box.conf[0].item())
                
                pixel_width = x2 - x1
                real_width = known_widths.get(label, 50.0)
                distance_cm = (real_width * FOCAL_LENGTH) / pixel_width if pixel_width > 0 else 0
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f} {distance_cm:.1f} cm",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
            
            # Draw EXIT button
            cv2.rectangle(frame, 
                          (EXIT_BUTTON['x1'], EXIT_BUTTON['y1']), 
                          (EXIT_BUTTON['x2'], EXIT_BUTTON['y2']), 
                          (0, 0, 255), -1)
            cv2.putText(frame, "EXIT", 
                        (EXIT_BUTTON['x1'] + 25, EXIT_BUTTON['y2'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("YOLO + Distance", frame)
            out.write(frame)
            
            # Verbose key handling
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("ESC key pressed")
                break
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
