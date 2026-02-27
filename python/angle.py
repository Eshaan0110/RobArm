import cv2
import mediapipe as mp
import time
import math
from pyfirmata import Arduino, SERVO
import inspect

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

port = 'COM7'
board = Arduino(port)

# Servo pin setup
left_servos = {'shoulder': 9, 'elbow': 10, 'rotation': 11, 'grip': 5, 'tilt': 3}
right_servos = {'shoulder': 6, 'elbow': 7, 'rotation': 8, 'grip': 4, 'tilt': 2}

for pin in list(left_servos.values()) + list(right_servos.values()):
    board.digital[pin].mode = SERVO

def rotate(pin, angle):
    board.digital[pin].write(angle)

def motion(servopin, target, current):
    if target != current:
        diff = abs(target - current)
        step = min(5, diff)
        new_angle = current + step if target > current else current - step
        rotate(servopin, new_angle)
        return new_angle
    return current

# Mediapipe setup
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Initial values
pTime = 0
current_angles = {
    'left': {'shoulder': 85, 'elbow': 134, 'rotation': 90, 'grip': 30, 'tilt': 55},
    'right': {'shoulder': 85, 'elbow': 134, 'rotation': 90, 'grip': 30, 'tilt': 55}
}

# Geometry helpers
def anglefinder(C, A=50, B=50):
    cos_a = (-(A**2)+(B**2)+(C**2))/(2*B*C)
    cos_b = ((A**2)-(B**2)+(C**2))/(2*A*C)
    cos_c = ((A**2)+(B**2)-(C**2))/(2*A*B)
    alpha = math.degrees(math.acos(cos_a))
    beta = math.degrees(math.acos(cos_b))
    gamma = math.degrees(math.acos(cos_c))
    return [gamma, alpha, beta, C]

def anglecorrector(y, angle, maxangle=90, range=400):
    return int(((y / range) * maxangle) + angle)

def distcalc(p1, p2, ignore_z=False):
    x2, y2, z2 = p2
    x1, y1, z1 = p1
    if ignore_z:
        z1 = z2 = 0
    return (((x2 - x1)**2) + ((y2 - y1)**2) + ((z2 - z1)**2)) ** 0.5

def gripangle(p1, p2):
    val = int(distcalc(p1, p2, ignore_z=True)) + 20
    return min(val, 130)

def bruhh(x):  # elbow angle
    return int((((100 - x) / 60) * 90) + 125)

def planeanglefinder(a, b, c):
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c
    v1 = [bx - ax, by - ay, bz - az]
    v2 = [cx - ax, cy - ay, cz - az]
    normal = [v1[1]*v2[2] - v1[2]*v2[1],
              v1[2]*v2[0] - v1[0]*v2[2],
              v1[0]*v2[1] - v1[1]*v2[0]]
    c = normal[2]
    mag = (normal[0]**2 + normal[1]**2 + normal[2]**2)**0.5
    cosa = (c*1300) / (1300 * mag)
    angle = math.degrees(math.acos(cosa))
    angle = abs(int(((angle - 89) * 10000) - 9960))
    return angle

def tilt(a, b):
    return min(int(distcalc(a, b, ignore_z=True)), 180)

# Frame loop
while True:
    success, img = cap.read()
    if not success:
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    try:
        if results.multi_hand_landmarks:
            for idx, handLms in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label  # 'Left' or 'Right'
                hand_side = 'left' if handedness == "Left" else 'right'
                servos = left_servos if hand_side == 'left' else right_servos
                angles = current_angles[hand_side]

                h, w, _ = img.shape
                landmarks = {}

                for id, lm in enumerate(handLms.landmark):
                    cx, cy, cz = (int(lm.x * w) - 300) * -1, (int(lm.y * h) - 500) * -1, int(lm.z * 5000 * -1)
                    landmarks[id] = [cx, cy, cz]

                if 9 in landmarks:
                    init = [2, 33, 69]
                    dist = distcalc(landmarks[9], init)
                    yolo = anglefinder(dist / 5)
                    angles['shoulder'] = max(0, int(yolo[0]) - 15)
                    angles['elbow'] = bruhh(int(anglecorrector(landmarks[9][1], yolo[1])))
                    angles['rotation'] = (int(anglecorrector(landmarks[9][0], 0, 90, 300))) + 90

                if all(k in landmarks for k in [5, 17, 1]):
                    angles['tilt'] = tilt(landmarks[5], landmarks[12])
                if all(k in landmarks for k in [4, 8]):
                    angles['grip'] = gripangle(landmarks[4], landmarks[8])

                # Apply angles via motion
                for joint in angles:
                    angles[joint] = motion(servos[joint], angles[joint], angles[joint])

                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    except Exception as e:
        print("Error:", e)
        continue

    # Show FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime != pTime else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Dual Arm Tracking", cv2.flip(img, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
