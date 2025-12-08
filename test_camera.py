import cv2

print("opening cam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ CV2 camera open failed")
    exit()

print("✅ CV2 camera opened")

while True:
    ret, frame = cap.read()
    print("read:", ret)
    if not ret:
        break

    # GUI 없는 환경에서 강제 종료 방지
    try:
        cv2.imshow("frame", frame)
    except:
        print("⚠️ imshow failed (no GUI)")
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
