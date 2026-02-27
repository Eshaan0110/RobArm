#  Vision-Based 3DOF Robotic Arm Control

##  Overview

This project implements a real-time 3DOF robotic arm control system using computer vision.

A webcam captures hand movements, MediaPipe extracts hand landmarks, joint angles are computed using vector mathematics, and commands are transmitted to an Arduino to actuate servo motors.

This project demonstrates:
- Real-time computer vision
- Angle computation using geometry
- Hardware-software integration
- Serial communication
- Embedded control

---

## System Pipeline

Webcam  
↓  
MediaPipe Hand Landmark Detection  
↓  
Joint Angle Computation  
↓  
Angle Mapping (0°–180°)  
↓  
Serial Communication  
↓  
Arduino  
↓  
Servo Motors  

---

##  Core Logic

- MediaPipe detects 21 hand landmarks.
- Specific landmark triplets are used to compute joint angles.
- Angles are calculated using vector dot product formula.
- Values are mapped to servo-safe range.
- Data is transmitted over serial to Arduino.

---

##  Project Structure
