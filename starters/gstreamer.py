import cv2

# Gebruik een GStreamer-pipeline die rechtstreeks werkt met libcamera
gst_pipeline = (
    "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)

camera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not camera.isOpened():
    print("Kan de camera niet openen met GStreamer")
    exit()

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Kan het frame niet lezen")
            break
        cv2.imshow('Raspberry Pi Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Gestopt door gebruiker")

finally:
    camera.release()
    cv2.destroyAllWindows()

