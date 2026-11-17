from ultralytics import YOLO
import cv2

model = YOLO("yolo26l-objv1-150.pt")
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    raise IOError("Cannot open camera")

try:
    while True:
        ret, frame = cam.read()

        if not ret:
            break

        results = model.predict(
            source=frame,
            imgsz=640,
            verbose=False
        )

        annotated = results[0].plot()
        cv2.imshow("YOLO26", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cam.release()
    cv2.destroyAllWindows()