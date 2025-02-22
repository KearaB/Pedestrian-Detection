# 🚶‍♂️ Pedestrian Detection using OpenCV & HOG  

A simple and efficient pedestrian detection system using **OpenCV** and **Histogram of Oriented Gradients (HOG)**.  
This project can detect pedestrians in both **images and videos** and allows users to test their own files for optimization.  


## 🛠 Tools & Technologies Used

<p align="left">
  <img src="https://img.shields.io/badge/OS-Windows%20%7C%20Linux%20%7C%20Mac-9cf?style=for-the-badge&logo=windows&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/OpenCV-4.5+-red?style=for-the-badge&logo=opencv" />
  <img src="https://img.shields.io/badge/Imutils-0.5.4-green?style=for-the-badge" />
  <p align="center"> 
</p>


## 📌 Features
✅ Detect pedestrians in **images** and **videos**  
✅ Uses **pre-trained HOG + SVM detector** for accuracy  
✅ **Supports custom** images/videos for testing  
✅ **Optimized processing** for real-time detection  
✅ Saves output with detected pedestrians  


## 🎯 How It Works

#### 🔍 **Pedestrian Detection Logic**
1. **Load an image or video**  
2. **Preprocess the input** (resize for efficiency)  
3. **Apply HOG + SVM detection**  
4. **Draw bounding boxes around pedestrians**  
5. **Display and save the output**  

#### 🔍 **Detection Model**
- Uses **HOGDescriptor + SVM** to recognize pedestrians  
- Optimized with **winStride, padding, and scale tuning**  
- Works well for **real-time video processing**  


## ⚙️ Setup & Installation

### **1️⃣ Install Dependencies**
Make sure you have Python **3.8+** installed. Then run:  
```bash
pip install opencv-python imutils
```

### **2️⃣ Clone Repository**
``` bash
git clone https://github.com/your-repo/pedestrian-detection.git
cd pedestrian-detection
```

### **3️⃣ Run Pedestrian Detection on Images**
```bash
python Pedestrian Detection - Image.py --image data/sample_image.png
```

### **4️⃣ Run Pedestrian Detection on Videos**
```bash
python Pedestrian Detection - Video .py --video input/sample_video.mp4
```

### ⏹ How to Stop the Video Detection

The pedestrian detection script allows two ways to **exit the video processing**:

1. **Manually:** Press **`q`** on your keyboard at any time to stop the detection.
2. **Automatically:** The script will **stop after 12 seconds** to prevent running indefinitely.

```python
# Stop if 'q' is pressed
if cv2.waitKey(1) & 0xFF == ord("q"):
    print("Stopping: 'q' key pressed.")
    break

# Automatically stop after 12 seconds
if time.time() - start_time > 12:
    print("Stopping: 12 seconds elapsed.")
    break
```


## 📂 Testing Your Own Files

You can replace the sample image or video with your own files:

* Image Detection: Replace  "data/image.png" with your own pedestrian image
* Video Detection: Replace "input/video.mp4" with any MP4 video


## 📌 References
* [OpenCV Docs](https://docs.opencv.org/4.x/d1/dfb/intro.html)
* [Sample Image Source](https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/CrossWalk_%285465840138%29.jpg/1280px-CrossWalk_%285465840138%29.jpg)
* [Sample Video Source](https://www.pexels.com/)

## 🚀 Future Improvements
* Use YOLOv5 for real-time detection
* Implement multi-object tracking (MOT)
* Enhance accuracy with deep learning models