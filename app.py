import cv2
import tkinter as tk
from tkinter import ttk
from plantcv import plantcv as pcv
from PIL import Image, ImageTk
import pandas as pd
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from tkinter import filedialog
import os
import threading
import time
import queue
import serial
import serial.tools.list_ports
from datetime import datetime, timedelta


class ScrollableFrame(tk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self)
        self.scrollbar_y = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollbar_x = tk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.scrollable_frame = tk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar_y.pack(side="right", fill="y")
        self.scrollbar_x.pack(side="bottom", fill="x")

        self.zoom_scale = 1.0

    def zoom(self, scale_factor):
        self.zoom_scale *= scale_factor
        self.canvas.scale("all", 0, 0, scale_factor, scale_factor)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


class CircularGauge(tk.Canvas):
    """Custom circular gauge widget for displaying environmental data"""
    def __init__(self, parent, title, min_value, max_value, **kwargs):
        super().__init__(parent, **kwargs)
        self.title = title
        self.min_value = min_value
        self.max_value = max_value
        self.value = min_value
        self.width = kwargs.get('width', 150)
        self.height = kwargs.get('height', 150)
        self.configure(width=self.width, height=self.height, bg='white', highlightthickness=0)
        self.draw()

    def set_value(self, value):
        """Update the gauge value and redraw"""
        self.value = max(self.min_value, min(self.max_value, value))
        self.draw()

    def draw(self):
        """Draw the gauge with current value"""
        self.delete("all")
        
        # Draw background
        self.create_oval(10, 10, self.width-10, self.height-10, outline='gray', width=2)
        
        # Draw title
        self.create_text(self.width/2, 20, text=self.title, font=('Arial', 12, 'bold'))
        
        # Draw scale
        center_x, center_y = self.width/2, self.height/2
        radius = (self.width - 40) / 2
        start_angle = 135
        end_angle = 405  # 45 degrees past the bottom
        
        # Calculate angle for current value
        angle_range = end_angle - start_angle
        ratio = (self.value - self.min_value) / (self.max_value - self.min_value)
        current_angle = start_angle + (ratio * angle_range)
        
        # Draw value arc
        self.create_arc(center_x-radius, center_y-radius, center_x+radius, center_y+radius,
                        start=start_angle, extent=current_angle-start_angle, outline='', 
                        style='pieslice', fill='#0080ff')
        
        # Draw value text
        self.create_text(center_x, center_y+radius/2, text=f"{self.value:.1f}", 
                         font=('Arial', 14, 'bold'))
        
        # Draw min and max labels
        self.create_text(center_x-radius*0.9, center_y+radius*0.4, 
                         text=f"{self.min_value}", font=('Arial', 8))
        self.create_text(center_x+radius*0.9, center_y+radius*0.4, 
                         text=f"{self.max_value}", font=('Arial', 8))


class PlantDetectionDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Detection & Environmental Dashboard")
        
        # Create tabbed interface
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self.plant_tab = ttk.Frame(self.notebook)
        self.env_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.plant_tab, text='Plant Detection')
        self.notebook.add(self.env_tab, text='Environmental Data')
        
        # Plant Detection Tab Variables
        self.cameras = self.get_available_cameras()
        self.camera_streams = [None] * 3
        self.is_running = [False] * 3
        self.grids = []
        self.last_update_time = [datetime.min] * 3
        self.analysis_completed = [False] * 3
        self.update_threads = [None] * 3
        
        # Stream optimization variables
        self.frame_queues = [queue.Queue(maxsize=2) for _ in range(3)]
        self.stream_quality = [50] * 3
        self.stream_fps = [5] * 3
        self.last_frame_time = [0] * 3
        self.current_frames = [None] * 3
        self.stream_active = [False] * 3
        
        # Environmental Tab Variables
        self.serial_port = None
        self.serial_thread = None
        self.is_serial_running = False
        self.env_data = {
            'temperature': [],
            'humidity': [],
            'co2': [],
            'light': [],
            'soil_moisture': [],
            'time': []
        }
        self.env_data_lock = threading.Lock()
        self.max_history_points = 100  # Store last 100 readings
        
        # Initialize PlantCV debug parameters
        pcv.params.debug = None
        pcv.params.dpi = 100

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Build Plant Detection UI
        self.create_plant_detection_widgets()
        
        # Build Environmental Data UI
        self.create_environmental_widgets()

        # Bind mouse wheel for zoom in Plant Detection tab
        self.root.bind("<Control-MouseWheel>", self.mouse_wheel_zoom)
        
        # Handle application close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_plant_detection_widgets(self):
        # Add scrollable frame
        self.main_frame = ScrollableFrame(self.plant_tab)
        self.main_frame.pack(fill="both", expand=True)

        # Configure scrollable frame layout
        self.scrollable_frame = self.main_frame.scrollable_frame
        self.scrollable_frame.columnconfigure([0, 1, 2], weight=1)
        self.scrollable_frame.rowconfigure([0, 1, 2], weight=1)

        # Zoom buttons
        zoom_frame = tk.Frame(self.plant_tab)
        zoom_frame.pack(fill="x", padx=10, pady=5)
        zoom_in_button = tk.Button(zoom_frame, text="Zoom In", command=lambda: self.main_frame.zoom(1.1))
        zoom_out_button = tk.Button(zoom_frame, text="Zoom Out", command=lambda: self.main_frame.zoom(0.9))
        zoom_in_button.pack(side="left", padx=5)
        zoom_out_button.pack(side="left", padx=5)

        # Refresh button
        refresh_button = tk.Button(zoom_frame, text="Refresh", command=self.refresh_all)
        refresh_button.pack(side="right", padx=5)

        # Machine Learning buttons
        ml_frame = tk.Frame(self.plant_tab)
        ml_frame.pack(fill="x", padx=10, pady=5)
        create_dataset_button = tk.Button(ml_frame, text="Create New Dataset", command=self.create_dataset)
        train_model_button = tk.Button(ml_frame, text="Train Model", command=self.train_model)
        create_dataset_button.pack(side="left", padx=5)
        train_model_button.pack(side="left", padx=5)

        for i in range(3):
            frame = tk.Frame(self.scrollable_frame, relief=tk.RAISED, borderwidth=2)
            frame.grid(row=0, column=i, padx=10, pady=10, sticky="nsew")
            frame.columnconfigure([0, 1], weight=1)
            frame.rowconfigure([0, 1, 2, 3, 4, 5, 6, 7, 8], weight=1)

            # Camera selection
            tk.Label(frame, text=f"Grid {i + 1}: Select Camera").grid(row=0, column=0, sticky="w", padx=5)
            camera_dropdown = ttk.Combobox(frame, values=self.cameras, state="readonly")
            camera_dropdown.grid(row=0, column=1, sticky="ew", padx=5)
            camera_dropdown.bind("<<ComboboxSelected>>", lambda event, idx=i: self.select_camera(idx))

            # Stream controls
            stream_control_frame = tk.Frame(frame)
            stream_control_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
            
            # Stream toggle button
            stream_toggle = tk.Button(stream_control_frame, text="Start Stream", 
                                     command=lambda idx=i: self.toggle_stream(idx))
            stream_toggle.pack(side="left", padx=5)
            
            # Capture single frame button
            capture_button = tk.Button(stream_control_frame, text="Capture Frame", 
                                      command=lambda idx=i: self.capture_single_frame(idx))
            capture_button.pack(side="left", padx=5)
            
            # Quality slider
            tk.Label(stream_control_frame, text="Quality:").pack(side="left", padx=(10,0))
            quality_slider = tk.Scale(stream_control_frame, from_=10, to=100, orient="horizontal",
                                     command=lambda val, idx=i: self.set_stream_quality(idx, val))
            quality_slider.set(self.stream_quality[i])
            quality_slider.pack(side="left", padx=5)
            
            # FPS slider
            tk.Label(stream_control_frame, text="FPS:").pack(side="left", padx=(10,0))
            fps_slider = tk.Scale(stream_control_frame, from_=1, to=30, orient="horizontal",
                                 command=lambda val, idx=i: self.set_stream_fps(idx, val))
            fps_slider.set(self.stream_fps[i])
            fps_slider.pack(side="left", padx=5)

            # Analysis selection
            tk.Label(frame, text=f"Grid {i + 1}: Select Analysis").grid(row=2, column=0, sticky="w", padx=5)
            analysis_options = ["Get ROI Info", "Photosynthetic Analysis", "Health Status Analysis",
                                "Growth Rate Analysis", "Nutrient Deficiency Detection",
                                "Machine Learning Detection", "Plant Morphology Analysis"]
            analysis_dropdown = ttk.Combobox(frame, values=analysis_options, state="readonly")
            analysis_dropdown.grid(row=2, column=1, sticky="ew", padx=5)
            analysis_dropdown.current(0)
            
            # Analyze button
            analyze_button = tk.Button(frame, text="Analyze Current Frame", 
                                     command=lambda idx=i: self.analyze_current_frame(idx))
            analyze_button.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

            # Image display
            image_display = tk.Label(frame, text="Camera Feed", bg="black")
            image_display.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=5)

            # Analysis Option Data Table
            tk.Label(frame, text="Analysis Option Data Table").grid(row=5, column=0, sticky="w", padx=5)
            tree = ttk.Treeview(frame, show="headings", height=5)
            tree.grid(row=6, column=0, columnspan=2, sticky="nsew", padx=5)

            # Graph display
            graph_frame = tk.Frame(frame)
            graph_frame.grid(row=7, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

            self.grids.append({
                "frame": frame,
                "camera_dropdown": camera_dropdown,
                "analysis_dropdown": analysis_dropdown,
                "image_display": image_display,
                "tree": tree,
                "graph_frame": graph_frame,
                "stream_toggle": stream_toggle,
                "capture_button": capture_button,
                "quality_slider": quality_slider,
                "fps_slider": fps_slider,
                "analyze_button": analyze_button
            })

    def create_environmental_widgets(self):
        # Main container for environmental tab
        env_main = tk.Frame(self.env_tab)
        env_main.pack(fill='both', expand=True, padx=10, pady=10)
        env_main.columnconfigure(0, weight=1)
        env_main.columnconfigure(1, weight=1)
        env_main.rowconfigure(0, weight=0)  # Serial controls
        env_main.rowconfigure(1, weight=1)  # Gauges
        env_main.rowconfigure(2, weight=1)  # Graphs
        
        # Serial connection controls
        serial_frame = tk.LabelFrame(env_main, text="Serial Connection", padx=10, pady=10)
        serial_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        
        # Get available serial ports
        ports = self.get_serial_ports()
        
        # Serial port selection
        tk.Label(serial_frame, text="Port:").grid(row=0, column=0, sticky="w", padx=5)
        self.port_dropdown = ttk.Combobox(serial_frame, values=ports, state="readonly", width=30)
        self.port_dropdown.grid(row=0, column=1, sticky="ew", padx=5)
        if ports:
            self.port_dropdown.current(0)
            
        # Baud rate selection
        tk.Label(serial_frame, text="Baud Rate:").grid(row=0, column=2, sticky="w", padx=5)
        baud_rates = ['9600', '19200', '38400', '57600', '115200']
        self.baud_dropdown = ttk.Combobox(serial_frame, values=baud_rates, state="readonly", width=10)
        self.baud_dropdown.grid(row=0, column=3, sticky="ew", padx=5)
        self.baud_dropdown.current(0)  # Default to 9600
        
        # Connect/Disconnect button
        self.serial_connect_btn = tk.Button(serial_frame, text="Connect", 
                                          command=self.toggle_serial_connection)
        self.serial_connect_btn.grid(row=0, column=4, padx=10)
        
        # Refresh ports button
        refresh_btn = tk.Button(serial_frame, text="Refresh Ports", 
                               command=self.refresh_serial_ports)
        refresh_btn.grid(row=0, column=5, padx=5)
        
        # Clear data button
        clear_btn = tk.Button(serial_frame, text="Clear Data", 
                             command=self.clear_env_data)
        clear_btn.grid(row=0, column=6, padx=5)
        
        # Current readings display
        gauge_frame = tk.LabelFrame(env_main, text="Current Readings", padx=10, pady=10)
        gauge_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        # Create gauges for each environmental parameter
        gauge_frame.columnconfigure(0, weight=1)
        gauge_frame.columnconfigure(1, weight=1)
        gauge_frame.columnconfigure(2, weight=1)
        gauge_frame.columnconfigure(3, weight=1)
        gauge_frame.columnconfigure(4, weight=1)
        
        self.gauges = {
            'temperature': CircularGauge(gauge_frame, "Temperature (°C)", 0, 50, width=150, height=150),
            'humidity': CircularGauge(gauge_frame, "Humidity (%)", 0, 100, width=150, height=150),
            'co2': CircularGauge(gauge_frame, "CO2 (ppm)", 0, 2000, width=150, height=150),
            'light': CircularGauge(gauge_frame, "Light (lux)", 0, 10000, width=150, height=150),
            'soil_moisture': CircularGauge(gauge_frame, "Soil Moisture (%)", 0, 100, width=150, height=150)
        }
        
        # Position gauges
        self.gauges['temperature'].grid(row=0, column=0, padx=5, pady=5)
        self.gauges['humidity'].grid(row=0, column=1, padx=5, pady=5)
        self.gauges['co2'].grid(row=0, column=2, padx=5, pady=5)
        self.gauges['light'].grid(row=0, column=3, padx=5, pady=5)
        self.gauges['soil_moisture'].grid(row=0, column=4, padx=5, pady=5)
        
        # Last updated timestamp
        self.last_updated_label = tk.Label(gauge_frame, text="Last Updated: Never")
        self.last_updated_label.grid(row=1, column=0, columnspan=5, sticky="ew", pady=5)
        
        # Historical data graphs
        graph_frame = tk.LabelFrame(env_main, text="Historical Data", padx=10, pady=10)
        graph_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        
        # Create figure for graphs
        self.env_fig = plt.figure(figsize=(10, 6))
        self.env_canvas = FigureCanvasTkAgg(self.env_fig, master=graph_frame)
        self.env_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        
        # Connection status indicator
        self.status_label = tk.Label(env_main, text="Disconnected", fg="red", 
                                   font=('Arial', 10, 'bold'))
        self.status_label.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        
        # Initialize graphs
        self.update_env_graphs()

    def get_serial_ports(self):
        """Get list of available serial ports"""
        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append(port.device)
        return ports

    def refresh_serial_ports(self):
        """Refresh the list of available serial ports"""
        ports = self.get_serial_ports()
        self.port_dropdown['values'] = ports
        if ports:
            self.port_dropdown.current(0)
        logging.info(f"Refreshed serial ports: {len(ports)} ports found")

    def toggle_serial_connection(self):
        """Connect to or disconnect from the selected serial port"""
        if self.is_serial_running:
            # Disconnect
            self.is_serial_running = False
            if self.serial_thread and self.serial_thread.is_alive():
                self.serial_thread.join(timeout=1.0)
            if self.serial_port:
                self.serial_port.close()
                self.serial_port = None
            
            self.serial_connect_btn.config(text="Connect")
            self.status_label.config(text="Disconnected", fg="red")
            logging.info("Disconnected from serial port")
        else:
            # Connect
            try:
                port = self.port_dropdown.get()
                baud = int(self.baud_dropdown.get())
                
                if not port:
                    logging.error("No serial port selected")
                    return
                    
                self.serial_port = serial.Serial(port, baud, timeout=0.5)
                self.is_serial_running = True
                
                # Start reading thread
                self.serial_thread = threading.Thread(target=self.read_serial_data)
                self.serial_thread.daemon = True
                self.serial_thread.start()
                
                self.serial_connect_btn.config(text="Disconnect")
                self.status_label.config(text=f"Connected to {port} at {baud} baud", fg="green")
                logging.info(f"Connected to {port} at {baud} baud")
                
            except Exception as e:
                logging.error(f"Error connecting to serial port: {e}")
                self.status_label.config(text=f"Connection error: {str(e)}", fg="red")

    def read_serial_data(self):
        """Read and process data from the serial port"""
        buffer = ""
        
        while self.is_serial_running and self.serial_port:
            try:
                # Read from serial port
                data = self.serial_port.read(100).decode('utf-8', errors='replace')
                
                if data:
                    buffer += data
                    
                    # Process complete messages
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        self.process_serial_message(line.strip())
                        
                # Small delay to prevent CPU hogging
                time.sleep(0.01)
                
            except Exception as e:
                logging.error(f"Error reading from serial port: {e}")
                self.is_serial_running = False
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Serial error: {str(e)}", fg="red"))
                break

    def process_serial_message(self, message):
        """Process a complete message from the serial port"""
        try:
            # Expected format: temp=25.5,humidity=60.2,co2=450,light=8500,moisture=75.3
            if not message:
                return
                
            # Parse message into key-value pairs
            parts = message.split(',')
            data = {}
            
            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    try:
                        data[key.strip()] = float(value.strip())
                    except ValueError:
                        data[key.strip()] = value.strip()
            
            # Check for the environmental data values we're interested in
            now = datetime.now()
            
            with self.env_data_lock:
                if 'temp' in data or 'temperature' in data:
                    temp = data.get('temp', data.get('temperature', 0))
                    self.env_data['temperature'].append(temp)
                    # Update gauge in main thread
                    self.root.after(0, lambda: self.gauges['temperature'].set_value(temp))
                
                if 'humidity' in data:
                    humidity = data.get('humidity', 0)
                    self.env_data['humidity'].append(humidity)
                    self.root.after(0, lambda: self.gauges['humidity'].set_value(humidity))
                
                if 'co2' in data:
                    co2 = data.get('co2', 0)
                    self.env_data['co2'].append(co2)
                    self.root.after(0, lambda: self.gauges['co2'].set_value(co2))
                
                if 'light' in data:
                    light = data.get('light', 0)
                    self.env_data['light'].append(light)
                    self.root.after(0, lambda: self.gauges['light'].set_value(light))
                
                if 'moisture' in data or 'soil_moisture' in data:
                    moisture = data.get('moisture', data.get('soil_moisture', 0))
                    self.env_data['soil_moisture'].append(moisture)
                    self.root.after(0, lambda: self.gauges['soil_moisture'].set_value(moisture))
                
                # Add timestamp and limit data points
                self.env_data['time'].append(now)
                
                # Trim data if we exceed max history
                if len(self.env_data['time']) > self.max_history_points:
                    for key in self.env_data:
                        self.env_data[key] = self.env_data[key][-self.max_history_points:]
            
            # Update the timestamp and graphs in the main thread
            self.root.after(0, lambda: self.last_updated_label.config(
                text=f"Last Updated: {now.strftime('%Y-%m-%d %H:%M:%S')}"))
            self.root.after(0, self.update_env_graphs)
            
            logging.debug(f"Processed environmental data: {data}")
            
        except Exception as e:
            logging.error(f"Error processing serial message '{message}': {e}")

    def update_env_graphs(self):
        """Update the environmental data graphs"""
        try:
            # Clear the figure
            self.env_fig.clear()
            
            with self.env_data_lock:
                # Skip if no data
                if not self.env_data['time']:
                    # Draw empty plots
                    axes = self.env_fig.subplots(2, 3, sharex=True)
                    axes = axes.flatten()
                    for i, key in enumerate(['temperature', 'humidity', 'co2', 'light', 'soil_moisture']):
                        axes[i].set_title(key.replace('_', ' ').title())
                        axes[i].set_ylabel('Value')
                        axes[i].grid(True)
                    self.env_fig.tight_layout()
                    self.env_canvas.draw()
                    return
                    
                # Format timestamps for x-axis
                times = [t.strftime('%H:%M:%S') for t in self.env_data['time']]
                
                # Create subplots
                axes = self.env_fig.subplots(2, 3, sharex=True)
                axes = axes.flatten()  # Flatten for easier indexing
                
                # Plot each environmental parameter
                for i, key in enumerate(['temperature', 'humidity', 'co2', 'light', 'soil_moisture']):
                    if self.env_data[key]:  # Check if we have data for this parameter
                        axes[i].plot(times, self.env_data[key], 'o-', linewidth=2)
                        axes[i].set_title(key.replace('_', ' ').title())
                        axes[i].set_ylabel('Value')
                        axes[i].grid(True)
                        
                        # Only show some x-axis labels to avoid crowding
                        if len(times) > 10:
                            skip = len(times) // 10
                            axes[i].set_xticks(times[::skip])
                            axes[i].set_xticklabels(times[::skip], rotation=45)
                
                # Hide the last unused subplot if there is one
                if len(axes) > 5:
                    axes[5].set_visible(False)
                
                self.env_fig.tight_layout()
                self.env_canvas.draw()
                
        except Exception as e:
            logging.error(f"Error updating environmental graphs: {e}")

    def clear_env_data(self):
        """Clear all environmental data"""
        with self.env_data_lock:
            for key in self.env_data:
                self.env_data[key] = []
        
        # Reset gauges
        for gauge in self.gauges.values():
            gauge.set_value(0)
            
        # Reset last updated label
        self.last_updated_label.config(text="Last Updated: Never")
        
        # Update graphs
        self.update_env_graphs()
        logging.info("Environmental data cleared")

    def on_close(self):
        """Handle application close"""
        # Stop all streams
        for i in range(3):
            self.stop_stream(i)
            
        # Stop serial connection
        self.is_serial_running = False
        if self.serial_port:
            self.serial_port.close()
            
        # Destroy root window
        self.root.destroy()

    # The rest of your existing methods remain unchanged
    # (get_available_cameras, select_camera, toggle_stream, capture_single_frame, etc.)

    def get_available_cameras(self):
        cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append(f"Camera {i}")
                cap.release()
            else:
                cap.release()
        return cameras

    def select_camera(self, idx):
        camera_idx = self.grids[idx]["camera_dropdown"].current()
        if camera_idx is not None:
            # Stop any existing stream first
            self.stop_stream(idx)
            
            # Configure new camera
            self.camera_streams[idx] = cv2.VideoCapture(camera_idx)
            self.is_running[idx] = True
            self.analysis_completed[idx] = False
            
            # Don't start streaming automatically, wait for user to click "Start Stream"
            logging.info(f"Camera {camera_idx} selected for Grid {idx + 1}. Click 'Start Stream' to begin streaming.")

    def toggle_stream(self, idx):
        if self.stream_active[idx]:
            # Stop streaming
            self.stream_active[idx] = False
            self.grids[idx]["stream_toggle"].config(text="Start Stream")
            logging.info(f"Streaming stopped for Grid {idx + 1}")
        else:
            # Start streaming
            if self.camera_streams[idx] is None or not self.camera_streams[idx].isOpened():
                logging.error(f"No camera selected for Grid {idx + 1}")
                return
                
            self.stream_active[idx] = True
            self.grids[idx]["stream_toggle"].config(text="Stop Stream")
            logging.info(f"Streaming started for Grid {idx + 1}")
            
            # Start stream thread if not already running
            if self.update_threads[idx] is None or not self.update_threads[idx].is_alive():
                self.update_threads[idx] = threading.Thread(target=self.stream_frames, args=(idx,))
                self.update_threads[idx].daemon = True
                self.update_threads[idx].start()
                
            # Start display thread if not already running
            display_thread = threading.Thread(target=self.display_stream, args=(idx,))
            display_thread.daemon = True
            display_thread.start()

    def capture_single_frame(self, idx):
        if self.camera_streams[idx] is None or not self.camera_streams[idx].isOpened():
            logging.error(f"No camera selected for Grid {idx + 1}")
            return
            
        # Capture a single frame without streaming
        ret, frame = self.camera_streams[idx].read()
        if not ret:
            logging.error(f"Failed to capture frame from Camera {idx}")
            return
            
        # Store the captured frame
        self.current_frames[idx] = frame
        
        # Display the frame (without analysis overlay)
        self.display_image(frame, frame.copy(), idx)
        logging.info(f"Frame captured for Grid {idx + 1}")

    def stream_frames(self, idx):
        """Thread function to capture frames continuously at specified FPS"""
        while self.is_running[idx] and self.stream_active[idx]:
            if self.camera_streams[idx] is None or not self.camera_streams[idx].isOpened():
                break
                
            # Control frame rate
            current_time = time.time()
            time_diff = current_time - self.last_frame_time[idx]
            target_diff = 1.0 / self.stream_fps[idx]
            
            if time_diff < target_diff:
                time.sleep(target_diff - time_diff)
                
            # Capture new frame
            ret, frame = self.camera_streams[idx].read()
            self.last_frame_time[idx] = time.time()
            
            if not ret:
                logging.error(f"Failed to capture frame from Camera {idx}")
                self.stream_active[idx] = False
                break
                
            # Store the frame for analysis
            self.current_frames[idx] = frame.copy()
            
            # Resize for display based on quality setting
            # Lower quality = smaller size = better performance
            scale_factor = self.stream_quality[idx] / 100.0
            if scale_factor < 1.0:
                width = int(frame.shape[1] * scale_factor)
                height = int(frame.shape[0] * scale_factor)
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            
            # Put in queue, remove old frames if queue is full
            if self.frame_queues[idx].full():
                try:
                    self.frame_queues[idx].get_nowait()
                except queue.Empty:
                    pass
            
            try:
                self.frame_queues[idx].put_nowait(frame)
            except queue.Full:
                pass  # Skip frame if queue is still full
    
    def display_stream(self, idx):
        """Thread function to display frames from the queue"""
        while self.is_running[idx] and self.stream_active[idx]:
            try:
                # Get frame with timeout to avoid blocking forever
                frame = self.frame_queues[idx].get(timeout=0.5)
                
                # Use PIL for more efficient image conversion and display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(image=img)
                
                # Update UI in main thread
                self.root.after(0, lambda img=photo, idx=idx: self.update_display(img, idx))
                
                # Short sleep to prevent UI updates from overwhelming the system
                time.sleep(0.01)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in display stream for Grid {idx + 1}: {e}")
                break
    
    def update_display(self, photo, idx):
        """Update the image display label (called from main thread)"""
        try:
            display = self.grids[idx]["image_display"]
            display.configure(image=photo)
            display.image = photo  # Keep reference to prevent garbage collection
        except Exception as e:
            logging.error(f"Error updating display for Grid {idx + 1}: {e}")

    def analyze_current_frame(self, idx):
        """Analyze the currently displayed frame"""
        if self.current_frames[idx] is None:
            logging.error(f"No frame available for analysis in Grid {idx + 1}")
            return
            
        frame = self.current_frames[idx].copy()
        analysis_type = self.grids[idx]['analysis_dropdown'].get()
        
        try:
            # Show "Processing..." in the UI
            self.grids[idx]["analyze_button"].config(text="Processing...", state="disabled")
            self.root.update()
            
            # Perform analysis in a separate thread to avoid freezing UI
            analysis_thread = threading.Thread(
                target=self.perform_analysis, 
                args=(idx, frame, analysis_type)
            )
            analysis_thread.daemon = True
            analysis_thread.start()
        except Exception as e:
            logging.error(f"Error starting analysis for Grid {idx + 1}: {e}")
            self.grids[idx]["analyze_button"].config(text="Analyze Current Frame", state="normal")

    def perform_analysis(self, idx, frame, analysis_type):
        """Run analysis in background thread"""
        try:
            processed_img, analysis_data, photo_data, overlay_img = self.process_image(frame, analysis_type)
            
            # Update UI elements from the main thread
            self.root.after(0, lambda: self.display_image(frame, overlay_img, idx))
            self.root.after(0, lambda: self.update_table(idx, analysis_data, photo_data))
            self.root.after(0, lambda: self.update_graph(idx, analysis_data, photo_data))
            self.root.after(0, lambda: self.grids[idx]["analyze_button"].config(
                text="Analyze Current Frame", state="normal"))
            
            self.analysis_completed[idx] = True
        except Exception as e:
            logging.error(f"Error in analysis for Grid {idx + 1}: {e}")
            self.root.after(0, lambda: self.grids[idx]["analyze_button"].config(
                text="Analyze Current Frame", state="normal"))

    def set_stream_quality(self, idx, value):
        """Set the quality of the stream (resolution scaling)"""
        self.stream_quality[idx] = int(value)
        logging.info(f"Stream quality for Grid {idx + 1} set to {value}%")

    def set_stream_fps(self, idx, value):
        """Set the target FPS for the stream"""
        self.stream_fps[idx] = int(value)
        logging.info(f"Stream FPS for Grid {idx + 1} set to {value}")

    def stop_stream(self, idx):
        """Stop streaming and release resources"""
        self.stream_active[idx] = False
        self.is_running[idx] = False
        
        # Clear the frame queue
        while not self.frame_queues[idx].empty():
            try:
                self.frame_queues[idx].get_nowait()
            except queue.Empty:
                break
        
        # Join thread if it exists
        if self.update_threads[idx] is not None and self.update_threads[idx].is_alive():
            self.update_threads[idx].join(timeout=1.0)
        
        # Release camera
        if self.camera_streams[idx] is not None:
            self.camera_streams[idx].release()
            self.camera_streams[idx] = None
        
        # Update button text
        if idx < len(self.grids) and self.grids[idx].get("stream_toggle"):
            self.grids[idx]["stream_toggle"].config(text="Start Stream")

    def refresh_all(self):
        """Refresh all analysis without restarting streams"""
        for i in range(3):
            if self.current_frames[i] is not None:
                self.analyze_current_frame(i)

    def process_image(self, frame, analysis_type):
        overlay_img = frame.copy()
        analysis_data = []
        photo_data = []

        try:
            # Detect ROIs in the image
            lab_a = pcv.rgb2gray_lab(rgb_img=frame, channel="a")
            a_thresh = pcv.threshold.binary(lab_a, 134, object_type="light")
            a_fill = pcv.fill(a_thresh, size=200)
            labeled_mask, num_rois = pcv.create_labels(a_fill)

            # Iterate through each ROI and perform analysis
            for roi_idx in range(1, num_rois + 1):
                mask = labeled_mask == roi_idx
                overlay_img[mask] = [0, 255, 0]  # Highlight ROI

                # Perform analysis based on the selected type
                if analysis_type == 'Get ROI Info':
                    analysis_data.append({"ROI": f"ROI {roi_idx}", "Parameter": "Detected", "Value": "Yes"})

                elif analysis_type == 'Photosynthetic Analysis':
                    # Placeholder photosynthesis metrics
                    fvfm = 0.83
                    fqfm = 0.75
                    npq = 0.5
                    analysis_data.extend([
                        {"ROI": f"ROI {roi_idx}", "Parameter": "Fv/Fm", "Value": fvfm},
                        {"ROI": f"ROI {roi_idx}", "Parameter": "Fq'/Fm'", "Value": fqfm},
                        {"ROI": f"ROI {roi_idx}", "Parameter": "NPQ", "Value": npq},
                    ])

                elif analysis_type == 'Health Status Analysis':
                    health_status = "Healthy"  # Placeholder value
                    analysis_data.append({"ROI": f"ROI {roi_idx}", "Parameter": "Health Status", "Value": health_status})

                elif analysis_type == 'Growth Rate Analysis':
                    growth_rate = "Optimal"  # Placeholder value
                    analysis_data.append({"ROI": f"ROI {roi_idx}", "Parameter": "Growth Rate", "Value": growth_rate})

                elif analysis_type == 'Nutrient Deficiency Detection':
                    nutrient_deficiency = "None Detected"  # Placeholder value
                    analysis_data.append({"ROI": f"ROI {roi_idx}", "Parameter": "Nutrient Deficiency", "Value": nutrient_deficiency})

                elif analysis_type == 'Machine Learning Detection':
                    machine_learning_status = "Analysis Completed"  # Placeholder value
                    analysis_data.append({"ROI": f"ROI {roi_idx}", "Parameter": "ML Status", "Value": machine_learning_status})

                elif analysis_type == 'Plant Morphology Analysis':
                    try:
                        self.process_plant_morphology(roi_idx, mask, analysis_data)
                    except Exception as e:
                        logging.error(f"Error in Plant Morphology Analysis for ROI {roi_idx}: {e}")
                        analysis_data.append({"ROI": f"ROI {roi_idx}", "Parameter": "Error", "Value": str(e)})
        except Exception as e:
            logging.error(f"Error in process_image: {e}")
            # Add a default entry if analysis failed
            analysis_data.append({"ROI": "Error", "Parameter": "Processing Error", "Value": str(e)})
        
        # Convert to DataFrame (even if empty)
        if not analysis_data:
            analysis_data.append({"ROI": "None", "Parameter": "No Data", "Value": "N/A"})
        
        return frame, pd.DataFrame(analysis_data), pd.DataFrame(photo_data), overlay_img

    def process_plant_morphology(self, roi_idx, mask, analysis_data):
        # Convert mask to proper format
        mask_uint8 = mask.astype(np.uint8)
        
        # Calculate morphology metrics
        skeleton = pcv.morphology.skeletonize(mask=mask_uint8)
        pruned_skel, seg_img, edge_objects = pcv.morphology.prune(skel_img=skeleton, size=50, mask=mask_uint8)
        
        # Skip further processing if no segments were found
        if len(edge_objects) == 0:
            analysis_data.append({
                "ROI": f"ROI {roi_idx}",
                "Parameter": "Status",
                "Value": "No plant segments detected"
            })
            return
            
        try:
            leaf_obj, stem_obj = pcv.morphology.segment_sort(skel_img=pruned_skel, objects=edge_objects, mask=mask_uint8)
            stem_count = len(stem_obj)
            leaf_count = len(leaf_obj)
            segment_count = len(edge_objects)
            branch_points = pcv.morphology.find_branch_pts(skel_img=pruned_skel, mask=mask_uint8, label="default")
            tip_points = pcv.morphology.find_tips(skel_img=pruned_skel, mask=None, label="default")

            # Add overall metrics
            analysis_data.append({
                "ROI": f"ROI {roi_idx}",
                "Parameter": "Stem Count",
                "Value": stem_count
            })
            analysis_data.append({
                "ROI": f"ROI {roi_idx}",
                "Parameter": "Leaf Count",
                "Value": leaf_count
            })
            analysis_data.append({
                "ROI": f"ROI {roi_idx}",
                "Parameter": "Segment Count",
                "Value": segment_count
            })
            analysis_data.append({
                "ROI": f"ROI {roi_idx}",
                "Parameter": "Branch Points",
                "Value": len(branch_points)
            })
            analysis_data.append({
                "ROI": f"ROI {roi_idx}",
                "Parameter": "Tip Points",
                "Value": len(tip_points)
            })

            # Process each leaf object
            for idx, obj in enumerate(leaf_obj, start=1):
                length = pcv.morphology.segment_path_length(segmented_img=seg_img, objects=[obj], label="default")
                eu_length = pcv.morphology.segment_euclidean_length(segmented_img=seg_img, objects=[obj], label="default")
                curvature = pcv.morphology.segment_curvature(segmented_img=seg_img, objects=[obj], label="default")
                angle = pcv.morphology.segment_angle(segmented_img=seg_img, objects=[obj], label="default")
                
                # Try insertion angle calculation, but handle if it fails
                try:
                    insertion_angle = pcv.morphology.segment_insertion_angle(
                        skel_img=pruned_skel, 
                        segmented_img=seg_img, 
                        leaf_objects=[obj], 
                        stem_objects=stem_obj, 
                        size=20, 
                        label="default"
                    )
                except Exception as e:
                    logging.warning(f"Could not calculate insertion angle for leaf {idx} in ROI {roi_idx}: {e}")
                    insertion_angle = 0
                
                analysis_data.append({
                    "ROI": f"ROI {roi_idx} Leaf {idx}",
                    "Parameter": "Length",
                    "Value": length
                })
                analysis_data.append({
                    "ROI": f"ROI {roi_idx} Leaf {idx}",
                    "Parameter": "Euclidean Length",
                    "Value": eu_length
                })
                analysis_data.append({
                    "ROI": f"ROI {roi_idx} Leaf {idx}",
                    "Parameter": "Curvature",
                    "Value": curvature
                })
                analysis_data.append({
                    "ROI": f"ROI {roi_idx} Leaf {idx}",
                    "Parameter": "Angle",
                    "Value": angle
                })
                analysis_data.append({
                    "ROI": f"ROI {roi_idx} Leaf {idx}",
                    "Parameter": "Insertion Angle",
                    "Value": insertion_angle
                })
        except Exception as e:
            logging.error(f"Error during plant morphology processing for ROI {roi_idx}: {e}")
            analysis_data.append({
                "ROI": f"ROI {roi_idx}",
                "Parameter": "Error",
                "Value": str(e)
            })

    def display_image(self, frame, overlay_img, grid_idx):
        blended = cv2.addWeighted(frame, 0.6, overlay_img, 0.4, 0)
        blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(blended)
        img = ImageTk.PhotoImage(img)
        display = self.grids[grid_idx]["image_display"]
        display.configure(image=img)
        display.image = img

    def update_table(self, grid_idx, analysis_data, photo_data):
        if analysis_data.empty:
            return
            
        tree = self.grids[grid_idx]["tree"]
        for row in tree.get_children():
            tree.delete(row)

        analysis_type = self.grids[grid_idx]['analysis_dropdown'].get()

        if analysis_type == 'Plant Morphology Analysis':
            # Check if we have the detailed leaf data or just error/status messages
            if 'Length' in analysis_data.columns:
                # Set columns for Plant Morphology Analysis with detailed data
                tree["columns"] = ("ROI", "Length", "Euclidean Length", "Curvature", "Angle", "Insertion Angle")
                for col in tree["columns"]:
                    tree.heading(col, text=col)
                    tree.column(col, anchor="center")
                
                # Insert data rows
                for _, row in analysis_data.iterrows():
                    values = []
                    for col in tree["columns"]:
                        values.append(row.get(col, "N/A"))
                    tree.insert("", "end", values=tuple(values))
            else:
                # Use simpler format with parameter-value pairs
                tree["columns"] = ("ROI", "Parameter", "Value")
                for col in tree["columns"]:
                    tree.heading(col, text=col)
                    tree.column(col, anchor="center")
                
                for _, row in analysis_data.iterrows():
                    tree.insert("", "end", values=(
                        row.get("ROI", "N/A"), 
                        row.get("Parameter", "N/A"), 
                        row.get("Value", "N/A")
                    ))
        else:
            # Set default columns for other analyses
            tree["columns"] = ("ROI", "Parameter", "Value")
            for col in tree["columns"]:
                tree.heading(col, text=col)
                tree.column(col, anchor="center")
            
            for _, row in analysis_data.iterrows():
                tree.insert("", "end", values=(
                    row.get("ROI", "N/A"), 
                    row.get("Parameter", "N/A"), 
                    row.get("Value", "N/A")
                ))

    def update_graph(self, grid_idx, analysis_data, photo_data):
        for widget in self.grids[grid_idx]["graph_frame"].winfo_children():
            widget.destroy()

        if analysis_data.empty:
            return

        analysis_type = self.grids[grid_idx]['analysis_dropdown'].get()
        fig, ax = plt.subplots()

        try:
            # Populate the graph with dynamic data from the analysis
            if 'Parameter' in analysis_data.columns and 'Value' in analysis_data.columns:
                # Group data by ROI
                rois = analysis_data["ROI"].unique()
                
                for roi in rois:
                    roi_data = analysis_data[analysis_data["ROI"] == roi]
                    parameters = roi_data["Parameter"].tolist()
                    
                    # Convert values to numeric, handling different formats
                    values = []
                    for v in roi_data["Value"].tolist():
                        try:
                            if isinstance(v, (int, float)):
                                values.append(float(v))
                            elif isinstance(v, str) and v.replace('.', '', 1).replace('-', '', 1).isdigit():
                                values.append(float(v))
                            else:
                                values.append(0)  # Default value for non-numeric data
                        except (ValueError, TypeError):
                            values.append(0)
                    
                    if len(parameters) == len(values) and len(parameters) > 0:
                        ax.bar(parameters, values, label=roi)
            
            ax.set_title(f"{analysis_type} Metrics")
            ax.set_ylabel("Value")
            if len(ax.get_legend_handles_labels()[0]) > 0:
                ax.legend()
                
            # Adjust layout to make labels readable
            plt.tight_layout()
            fig.subplots_adjust(bottom=0.3)
            plt.xticks(rotation=45, ha='right')
            
            canvas = FigureCanvasTkAgg(fig, master=self.grids[grid_idx]["graph_frame"])
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            logging.error(f"Error updating graph: {e}")
        finally:
            plt.close(fig)

    def create_dataset(self):
        dataset_directory = filedialog.askdirectory(title="Select Directory to Save New Dataset")
        if dataset_directory:
            logging.info(f"Creating new dataset in directory: {dataset_directory}")
            # Placeholder logic for creating dataset
            # Actual logic should collect data from analysis and save it as required (e.g., CSV format)
            with open(os.path.join(dataset_directory, 'dataset.csv'), 'w') as f:
                f.write("ROI,Parameter,Value\n")
                f.write("Example ROI,Example Parameter,Example Value\n")
            logging.info("Dataset created successfully.")

    def train_model(self):
        training_file = filedialog.askopenfilename(title="Select Training File", filetypes=[("CSV Files", "*.csv"), ("Text Files", "*.txt")])
        if training_file:
            logging.info(f"Training model using file: {training_file}")
            # Placeholder logic for training model
            # Actual logic should include reading the dataset and training an ML model
            logging.info("Model training completed successfully.")

    def mouse_wheel_zoom(self, event):
        if event.state & 0x0004:  # If Ctrl key is held down
            if event.delta > 0:
                self.main_frame.zoom(1.1)
            elif event.delta < 0:
                self.main_frame.zoom(0.9)


if __name__ == "__main__":
    root = tk.Tk()
    app = PlantDetectionDashboard(root)
    root.geometry("1200x800")  # Set initial window size
    root.mainloop()
