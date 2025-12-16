# Vision-Based Intelligent System

This project is an upgrade of a sensor-based automatic lighting system to a **vision-based human detection system**. Instead of using ultrasonic or PIR sensors, a camera is used to explicitly detect **human presence**, and the detected output is used to control a lamp via a microcontroller.

The current phase focuses on **human detection using a laptop webcam**, which acts as a replacement for traditional motion sensors.

---

## Project Objective

* Detect **human presence** using a camera
* Avoid false triggers from animals or moving objects
* Generate a simple **ON / OFF control signal** based on detection
* Use this signal to control a lamp through a microcontroller (Arduino)
* Build a scalable foundation for future college-level deployment

---

## System Overview

Camera → Human Detection Model → Decision Logic → Microcontroller → Lamp

In this phase:

* Camera: Laptop inbuilt webcam
* Processing: Python-based computer vision and deep learning
* Output: Binary human-presence signal

---

## Repository Structure

```
Vision-Based-Intelligent-System/
│
├── README.md
├── requirements.txt
├── models/              # Pretrained or fine-tuned detection models
├── src/
│   ├── camera_interface.py   # Webcam capture
│   ├── human_detection.py    # Human detection logic
│   └── lamp_controller.py    # Arduino / relay control
```

---

## Technologies Used

* Python
* OpenCV (camera handling)
* PyTorch
* YOLO (Ultralytics)
* Arduino (for lamp control)

---

## Installation

1. Clone the repository

```
git clone https://github.com/SouvikPal001/Vision-Based-Intelligent-System.git
cd Vision-Based-Intelligent-System
```

2. Create and activate a virtual environment (recommended)

3. Install dependencies

```
pip install -r requirements.txt
```

---

## Current Status

* Project initialized
* Repository structure finalized
* Dependency setup completed
* Human detection module under development

---

## Future Scope

* Face detection and attendance automation
* Edge device deployment (Jetson / Raspberry Pi)
* Smart campus lighting integration
* Energy usage analytics

---

## Author

Souvik Pal

---

This repository is intended for academic and research purposes.
