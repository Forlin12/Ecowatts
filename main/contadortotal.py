import cv2
import datetime
import numpy as np
from ultralytics import solutions
import matplotlib.pyplot as plt


kk = 0
sensor = np.zeros(10000, dtype=int)
total = np.zeros(10000, dtype=int)# Pre-alocar um array de tamanho fixo para economizar memória

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(r"C:\Users\Forlin\Ovoflow\resources\eggesteira.mp4")


assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
region_points = [(400, 0), (400, 1080)]  # For line counting

# Init Object Counter
counter = solutions.ObjectCounter(
    show=True,  # Display the output
    region=region_points,  # Pass region points
    model=r"C:\Users\Forlin\Ecowatts\treinamento\runs\detect\train2\weights\best.pt",  # Model path
    show_in=True,  # Display in counts
    show_out=True,  # Display out counts
    line_width=0,  # Adjust the line width for bounding boxes and text display
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    rec_time = datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")

    cv2.putText(im0, f"FPS: {cap.get(cv2.CAP_PROP_FPS)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(im0, f"WxH: {im0.shape[1]} x {im0.shape[0]}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    im0 = counter.count(im0)
    print(f"Inward count: {counter.in_count}, Outward count: {counter.out_count}, quantidadeeee: {len(counter.track_ids)} ")

    kk += 1
    sensor[kk] = len(counter.track_ids)
    total[kk] = counter.out_count

    key = cv2.waitKey(30)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.plot(sensor[:kk+1], label="Object Count")
plt.xlabel('Frame')
plt.ylabel('Count')
plt.title('Object Count Over Time')
plt.legend()
plt.show()

cap.release()
cv2.destroyAllWindows()
plt.plot(total[:kk+1], label="Object Count")
plt.xlabel('Frame')
plt.ylabel('Count')
plt.title('Object Count Over Time')
plt.legend()
plt.show()