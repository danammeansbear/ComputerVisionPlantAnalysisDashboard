# ComputerVisionPlantAnalysisDashboard
A rewrite of a python project I wrote years ago. A dashboard to help you run analysis of plants in your greenhouse. 

Overview

The Plant Detection Dashboard is a GUI application developed using Python to monitor plant health and analyze various plant parameters. It uses multiple cameras to capture plant images and perform several analyses, such as photosynthetic analysis, nutrient deficiency detection, plant morphology analysis, and more. The application uses the following main libraries:

OpenCV: For accessing camera feeds.

Tkinter: For building the graphical user interface (GUI).

PlantCV: For image processing and plant analysis.

Pandas: For handling and analyzing tabular data.

Matplotlib: For graphing analysis results.

Pillow: For image display.

Features

Camera Selection: The user can select up to three cameras to be used for analysis.

Analysis Options: Users can choose different types of plant analyses, including:

Get ROI Info

Photosynthetic Analysis

Health Status Analysis

Growth Rate Analysis

Nutrient Deficiency Detection

Machine Learning Detection

Plant Morphology Analysis

Refresh Button: Analysis is only performed when the user manually clicks the refresh button, allowing the user to control when updates occur.

Zoom: The user can zoom in and out on the scrollable frame to view details more clearly.

Scrollable Interface: The application has a scrollable frame that can accommodate different analyses for multiple grids.

Logging: Logging is configured to provide detailed information on the progress and any issues encountered.

Installation

To use this application, you will need to install the following dependencies:

Python 3.6+

Required libraries:

pip install opencv-python plantcv Pillow matplotlib pandas

How to Use

Run the Application: Start the application by running the script using the command:

python plant_detection_dashboard.py

Select Cameras: For each grid, select an available camera using the drop-down menu.

Select Analysis: Choose the type of plant analysis you want to perform from the analysis drop-down menu.

Refresh Analysis: Click the "Refresh" button to update the analysis for each grid.

Zoom In/Out: Use the zoom buttons to adjust the view within the scrollable frame.

Plant Analyses

Get ROI Info: Provides information on regions of interest (ROIs) detected in the plant image.

Photosynthetic Analysis: Displays photosynthetic metrics like Fv/Fm, Fq'/Fm', and NPQ.

Health Status Analysis: Analyzes the health of the plant and displays a status (e.g., Healthy).

Growth Rate Analysis: Displays the growth rate of the plant based on detected metrics.

Nutrient Deficiency Detection: Attempts to identify any nutrient deficiencies in the plant.

Machine Learning Detection: Placeholder for machine learning analysis on plant data.

Plant Morphology Analysis: Analyzes the plant's morphological features, such as branch points, tip points, stem count, leaf count, etc. Multiple attempts are made to get an accurate skeleton if initial analysis fails.

Error Handling

If an error occurs during the plant morphology analysis, the application will make up to three attempts to obtain the skeleton and perform analysis.

Errors encountered during analysis are logged to help troubleshoot issues.

Controls and Navigation

Camera Dropdown: Select which camera to use for each grid.

Analysis Dropdown: Select the type of analysis to perform.

Refresh Button: Click to manually refresh and update the analysis for all grids.

Zoom Buttons: Click "Zoom In" or "Zoom Out" to adjust the view of the application.

Mouse Wheel Zoom: Hold the Ctrl key and use the mouse wheel to zoom in or out within the scrollable frame.

Notes

Once an analysis completes successfully, it will not update until the "Refresh" button is clicked.

The application is configured to use up to three camera feeds, each with a dedicated analysis grid.

Future Enhancements

Machine Learning Integration: Implement actual machine learning models for plant health detection and analysis.

Data Export: Add functionality to export analysis data to CSV or other formats.

User Authentication: Introduce user authentication for controlled access.

License

This project is licensed under the MIT License.

