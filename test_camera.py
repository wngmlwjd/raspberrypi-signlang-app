from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()

config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

time.sleep(1)

while True:
    frame = picam2.capture_array()
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
