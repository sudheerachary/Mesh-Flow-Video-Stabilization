import sys
import cv2

TIME= 7
file_name = sys.argv[1]
cap = cv2.VideoCapture(file_name)
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(TIME*frame_rate)

# total frame count 
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# generate resized video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('small-shaky-4.avi', fourcc, frame_rate, (int(frame_width/2), int(frame_height/2)))

frame_num = 0
while frame_num < frame_count:
    try:
        ret, frame = cap.read()
        new_frame = cv2.resize(frame, (int(frame_width/2), int(frame_height/2)), interpolation=cv2.INTER_CUBIC)
        out.write(new_frame)
        frame_num += 1
    except:
        break
        
cap.release()
out.release()