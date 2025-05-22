🌿 ComputerVisionPlantAnalysisDashboard
A comprehensive, real-time dashboard for plant health monitoring and greenhouse environmental tracking, combining computer vision and sensor integration in a unified Python application.

🔍 Overview
ComputerVisionPlantAnalysisDashboard is a dual-purpose GUI application built with Python, offering:

Plant Analysis System: Computer vision-based plant health diagnostics and morphological analysis

Environmental Monitoring System: Real-time visualization of environmental sensor data

Designed for greenhouse settings, this system empowers agronomists, researchers, and horticultural engineers to observe, analyze, and respond proactively to plant and environmental conditions.

📦 Key Technologies
Component	Purpose
OpenCV	Real-time camera input & image processing
PlantCV	Advanced plant morphological and physiological analysis
Tkinter	GUI development
PySerial	Serial communication with sensors
Matplotlib	Graphing and historical data visualization
Pandas	Data manipulation and tabular display
Pillow	Image display and enhancements

🚀 Features
🌱 Plant Detection Tab
Camera Management

Select up to three cameras

Adjustable streaming quality and FPS

Independent start/stop controls per feed

Analysis Options

ROI (Region of Interest) detection

Photosynthetic analysis (Fv/Fm, Fq'/Fm', NPQ)

Health scoring

Growth rate measurement

Nutrient deficiency estimation

ML-based detection framework

Full morphology scan: stems, leaves, tip/branch points, curvature, etc.

User Interface

Zoom, pan, and scroll capabilities

Real-time data tables and graphs

On-demand frame analysis for performance optimization

🌡️ Environmental Monitoring Tab
Serial Integration

Auto-detect available COM ports

Configurable baud rate

Safe connect/disconnect interface

Real-Time Visualization

Live circular gauges for:

🌡️ Temperature (°C)

💧 Humidity (%)

🫁 CO2 (ppm)

💡 Light (lux)

🌱 Soil Moisture (%)

Timestamped historical graphs

Data Management

Adjustable history window

Clear/reset functionality

Thread-safe continuous monitoring

🧪 Installation
Requirements
Python 3.6+

Install dependencies:

bash
Copy
Edit
pip install opencv-python pillow matplotlib pandas pyserial plantcv
🖥️ Usage Guide
Launching the App
bash
Copy
Edit
python app.py
Plant Detection Tab
Select available cameras

Adjust quality and FPS

Click Start Stream

Click Capture Frame for analysis

Choose an analysis type and click Analyze Current Frame

Review results in real-time tables and graphs

Environmental Monitoring Tab
Select a COM port and baud rate

Click Connect

Monitor gauges and graphs

Click Clear Data to reset tracking

📊 Plant Analysis Details
Analysis Type	Description
ROI Info	Detects plant outlines and regions of interest
Photosynthetic Metrics	Fv/Fm, Fq'/Fm', NPQ values
Health Analysis	Classifies plant health condition
Growth Rate	Detects size and growth over time
Nutrient Deficiency	Identifies visual symptoms
Machine Learning	Integration-ready framework
Morphology	Tip/stem/branch/leaf identification and shape analysis

🌎 Sensor Data Format (Expected via Serial)
Each message should include:

json
Copy
Edit
{
  "temperature": 24.5,
  "humidity": 55,
  "co2": 400,
  "light": 750,
  "soil_moisture": 38
}
⚙️ Performance Tips
For best results:

Use 1–5 FPS for smoother system performance

Adjust resolution/quality for lower CPU usage

Capturing frames on-demand is recommended for analysis

Environmental monitoring runs in the background regardless of tab

📈 Future Enhancements
✅ Machine Learning: Integrate CNNs for real-time classification

📤 Data Export: Export historical and analysis results as CSV/Excel

🔐 User Authentication: Secure login and role-based access

🌐 Web Dashboard: Remote access via Flask or Django API

🤖 Automated Greenhouse Control: Trigger fans, lights, watering systems

📄 License
MIT License – open for personal, educational, and commercial use. See LICENSE for more details.

👨‍💻 Author
Adam Dabdoub – CTI Developer
Specializing in full-stack development, plant science, and greenhouse automation.

