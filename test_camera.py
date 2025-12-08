import cv2
import time

print("Opening camera...")
cap = cv2.VideoCapture(0)

# 강제로 MJPG 설정 (라즈베리파이 카메라의 필수 설정)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(1.0)

print("Camera opened:", cap.isOpened())

# warmup
for i in range(5):
    ret, frame = cap.read()
    print(f"Warmup {i}:", ret)

if not cap.isOpened():
    print("Failed to open camera")
    exit()

while True:
    ret, frame = cap.read()
    print("read:", ret)
    if not ret:
        continue

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
