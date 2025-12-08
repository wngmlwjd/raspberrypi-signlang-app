import cv2

# 640x480 30fps 스트림
gst_pipeline = (
    "rpicam-vid --inline --nopreview -t 0 --width 640 --height 480 --framerate 30 "
    "-o - | "
    "ffmpeg -i pipe:0 -vcodec rawvideo -pix_fmt bgr24 -f rawvideo -"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("카메라 열기 실패")
    exit()

while True:
    ret, frame = cap.read()
    print("ret:", ret)
    if ret:
        cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
