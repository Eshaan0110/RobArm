#  Vision-Based Robotic Arm Control System

##  Overview

This project implements a real-time robotic arm control system using computer vision and hand tracking.

A webcam captures live hand movements, MediaPipe extracts hand landmarks, joint angles are computed using vector mathematics, and commands are transmitted to an Arduino to actuate servo motors.

This project demonstrates:

- Real-time computer vision
- Geometric angle computation
- Serial communication
- Embedded systems integration
- Hardware-software coordination

---

##  System Pipeline

Webcam  
↓  
MediaPipe Hand Landmark Detection  
↓  
Angle Calculation (angle.py)  
↓  
Main Control Logic (code.py)  
↓  
Serial Communication  
↓  
Arduino  
↓  
Servo Motors  

---

## Project Structure

```
ROBARM/
│
├── assets/                # (Optional) Media / visuals
│
├── python/
│   ├── angle.py           # Angle computation logic
│   └── code.py            # Main hand tracking + serial control script
│
├── requirement.txt        # Python dependencies
└── README.md
```

---

##  Core Working Principle

###  Hand Detection
MediaPipe detects 21 hand landmarks in real time.

###  Angle Computation
The `angle.py` module computes joint angles using the vector dot product formula:

angle = arccos( (v1 · v2) / (|v1| |v2|) )

These angles represent finger or wrist movement.

### Mapping & Transmission
- Angles are mapped to servo-safe range (0°–180°)
- Values are transmitted to Arduino via serial communication
- Arduino actuates servo motors accordingly

---

##  Installation

###  Clone Repository

```bash
git clone https://github.com/yourusername/ROBARM.git
cd ROBARM
```

### Install Dependencies

```bash
pip install -r requirement.txt
```

(If that doesn’t work, rename it to `requirements.txt` — that is the standard name.)

### Run the System

```bash
python python/code.py
```

---

##  Hardware Requirements

- Arduino (Uno / Nano)
- 3 Servo Motors (or more depending on your build)
- External Power Supply (recommended)
- USB Cable
- Webcam

---

##  Features

- Real-time hand tracking
- Modular angle computation
- Clean separation of logic (angle.py & code.py)
- Low-latency serial communication
- Extendable architecture
- Ready for dual-arm upgrade

