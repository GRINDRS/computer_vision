import cv2
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import json

def main():
    # === MQTT Setup ===
    broker = "localhost"
    port = 1883
    topic = "yolo/detections"
    client = mqtt.Client()
    client.connect(broker, port, 60)

    # === Load the YOLO model ===
    model = YOLO('/Users/grigorcrandon/Desktop/deco3801_computer_vision/runs/detect/train5/weights/best.pt')

    # === Start webcam ===
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame, conf=0.10)

        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = model.names[class_id]

            # Draw on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Append detection for MQTT
            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            })

        # === Send over MQTT ===
        if detections:
            payload = json.dumps({
                "detections": detections
            })
            client.publish(topic, payload)
            print("📤 MQTT Sent:", payload)

        cv2.imshow("YOLO Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

if __name__ == "__main__":
    main()

