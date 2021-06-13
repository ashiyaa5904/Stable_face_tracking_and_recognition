#  center values is used to rotate, area values is used to move forward and backward
import numpy as np


# function to track the face
def trackface(info, w, pid, previousError,fbRange,drone):
    area = info[1]
    x, y = info[0]
    fb = 0
    # how far the face is from the center
    error = x - w // 2
    speed = pid[0] * error + pid[1] * (error - previousError)
    #  speed range
    speed = int(np.clip(speed, -100, 100))

    #  if area is within the range, then drone stays static
    if fbRange[0] < area < fbRange[1]:
        fb = 0
    #  if area is big, move backward
    elif area > fbRange[1]:
        fb = -15
    #  if area is small and not zero, move forward
    elif area < fbRange[0] and area != 0:
        fb = 15
    #  if no face, then nothing happens
    if x == 0:
        speed = 0
        error = 0
    print(speed, fb)
    drone.send_rc_control(0, fb, 0, speed)
    # return error since previous error is used
    return error
