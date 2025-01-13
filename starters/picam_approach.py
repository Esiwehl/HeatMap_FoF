from picamera2 import Picamera2
from libcamera import controls
import time
import cv2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (1280, 960)}))
picam2.start()

picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous}) # 0 manual, 1 = continuous, 2 = auto

time.sleep(2)

#picam2.set_controls({"LensPosition": 0.5})
#picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 0.0})


try:
    while True:
        frame = picam2.capture_array()
        cv2.imshow('Picam Frame', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Afsluiten door gebruiker")

finally:
    picam2.stop()
    picam2.close()

