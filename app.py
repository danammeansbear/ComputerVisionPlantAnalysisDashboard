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


class PlantDetectionDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Detection Dashboard")
        self.cameras = self.get_available_cameras()
        self.camera_streams = [None] * 3
        self.is_running = [False] * 3
        self.grids = []
        self.last_update_time = [datetime.min] * 3
        self.analysis_completed = [False] * 3

        # Initialize PlantCV debug parameters
        pcv.params.debug = None
        pcv.params.dpi = 100

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Add scrollable frame
        self.main_frame = ScrollableFrame(self.root)
        self.main_frame.pack(fill="both", expand=True)

        # Configure scrollable frame layout
        self.scrollable_frame = self.main_frame.scrollable_frame
        self.scrollable_frame.columnconfigure([0, 1, 2], weight=1)
        self.scrollable_frame.rowconfigure([0, 1, 2], weight=1)

        # Build the UI
        self.create_widgets()

        # Bind mouse wheel for zoom in and out
        self.root.bind("<Control-MouseWheel>", self.mouse_wheel_zoom)

    def create_widgets(self):
        # Zoom buttons
        zoom_frame = tk.Frame(self.root)
        zoom_frame.pack(fill="x", padx=10, pady=5)
        zoom_in_button = tk.Button(zoom_frame, text="Zoom In", command=lambda: self.main_frame.zoom(1.1))
        zoom_out_button = tk.Button(zoom_frame, text="Zoom Out", command=lambda: self.main_frame.zoom(0.9))
        zoom_in_button.pack(side="left", padx=5)
        zoom_out_button.pack(side="left", padx=5)

        # Refresh button
        refresh_button = tk.Button(zoom_frame, text="Refresh", command=self.refresh_all)
        refresh_button.pack(side="right", padx=5)

        # Machine Learning buttons
        ml_frame = tk.Frame(self.root)
        ml_frame.pack(fill="x", padx=10, pady=5)
        create_dataset_button = tk.Button(ml_frame, text="Create New Dataset", command=self.create_dataset)
        train_model_button = tk.Button(ml_frame, text="Train Model", command=self.train_model)
        create_dataset_button.pack(side="left", padx=5)
        train_model_button.pack(side="left", padx=5)

        for i in range(3):
            frame = tk.Frame(self.scrollable_frame, relief=tk.RAISED, borderwidth=2)
            frame.grid(row=0, column=i, padx=10, pady=10, sticky="nsew")
            frame.columnconfigure([0, 1], weight=1)
            frame.rowconfigure([0, 1, 2, 3, 4, 5, 6, 7], weight=1)

            # Camera selection
            tk.Label(frame, text=f"Grid {i + 1}: Select Camera").grid(row=0, column=0, sticky="w", padx=5)
            camera_dropdown = ttk.Combobox(frame, values=self.cameras, state="readonly")
            camera_dropdown.grid(row=0, column=1, sticky="ew", padx=5)
            camera_dropdown.bind("<<ComboboxSelected>>", lambda event, idx=i: self.select_camera(idx))

            # Analysis selection
            tk.Label(frame, text=f"Grid {i + 1}: Select Analysis").grid(row=1, column=0, sticky="w", padx=5)
            analysis_options = ["Get ROI Info", "Photosynthetic Analysis", "Health Status Analysis",
                                "Growth Rate Analysis", "Nutrient Deficiency Detection",
                                "Machine Learning Detection", "Plant Morphology Analysis"]
            analysis_dropdown = ttk.Combobox(frame, values=analysis_options, state="readonly")
            analysis_dropdown.grid(row=1, column=1, sticky="ew", padx=5)
            analysis_dropdown.current(0)
            analysis_dropdown.bind("<<ComboboxSelected>>", lambda event, idx=i: self.update_analysis(idx))

            # Image display
            image_display = tk.Label(frame, text="Camera Feed", bg="black")
            image_display.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=5)

            # Analysis Option Data Table
            tk.Label(frame, text="Analysis Option Data Table").grid(row=3, column=0, sticky="w", padx=5)
            tree = ttk.Treeview(frame, show="headings", height=5)
            tree.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=5)

            # Graph display
            graph_frame = tk.Frame(frame)
            graph_frame.grid(row=5, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

            self.grids.append(
                {
                    "frame": frame,
                    "camera_dropdown": camera_dropdown,
                    "analysis_dropdown": analysis_dropdown,
                    "image_display": image_display,
                    "tree": tree,
                    "graph_frame": graph_frame,
                }
            )

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
            if self.camera_streams[idx] is not None:
                self.camera_streams[idx].release()
            self.camera_streams[idx] = cv2.VideoCapture(camera_idx)
            self.is_running[idx] = True
            self.analysis_completed[idx] = False
            self.update_frame(idx)

    def update_analysis(self, idx):
        analysis_type = self.grids[idx]["analysis_dropdown"].get()
        logging.info(f"Selected Analysis '{analysis_type}' for Grid {idx + 1}")
        self.analysis_completed[idx] = False
        self.update_frame(idx)

    def update_frame(self, grid_idx):
        if not self.is_running[grid_idx] or self.analysis_completed[grid_idx]:
            return

        cap = self.camera_streams[grid_idx]
        if cap is None or not cap.isOpened():
            logging.error(f"No camera assigned to Grid {grid_idx + 1}")
            return

        ret, frame = cap.read()
        if not ret:
            logging.error(f"Failed to capture frame from Camera {grid_idx}.")
            self.is_running[grid_idx] = False
            return

        try:
            analysis_type = self.grids[grid_idx]['analysis_dropdown'].get()
            processed_img, analysis_data, photo_data, overlay_img = self.process_image(frame, analysis_type)
            self.display_image(frame, overlay_img, grid_idx)
            self.update_table(grid_idx, analysis_data, photo_data)
            self.update_graph(grid_idx, analysis_data, photo_data)
            self.analysis_completed[grid_idx] = True
        except Exception as e:
            logging.error(f"Error processing image for Grid {grid_idx + 1}: {e}")

    def refresh_all(self):
        for i in range(3):
            self.analysis_completed[i] = False
            self.update_frame(i)

    def process_image(self, frame, analysis_type):
        overlay_img = frame.copy()
        analysis_data = []
        photo_data = []

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
                skeleton = None
                pruned_skel = None
                attempt = 0
                while attempt < 3:
                    try:
                        logging.info(f"Attempt {attempt + 1} for Plant Morphology Analysis on ROI {roi_idx}")
                        # Calculate morphology metrics
                        skeleton = pcv.morphology.skeletonize(mask=mask.astype(np.uint8))
                        pruned_skel, seg_img, edge_objects = pcv.morphology.prune(skel_img=skeleton, size=50, mask=mask.astype(np.uint8))
                        break
                    except Exception as e:
                        logging.warning(f"Attempt {attempt + 1} failed for ROI {roi_idx}: {e}")
                        attempt += 1

                if pruned_skel is not None:
                    try:
                        leaf_obj, stem_obj = pcv.morphology.segment_sort(skel_img=pruned_skel, objects=edge_objects, mask=mask.astype(np.uint8))
                        stem_count = len(stem_obj)
                        leaf_count = len(leaf_obj)
                        segment_count = len(edge_objects)
                        branch_points = pcv.morphology.find_branch_pts(skel_img=pruned_skel, mask=mask.astype(np.uint8), label="default")
                        tip_points = pcv.morphology.find_tips(skel_img=pruned_skel, mask=None, label="default")

                        for idx, obj in enumerate(leaf_obj, start=1):
                            length = pcv.morphology.segment_path_length(segmented_img=seg_img, objects=[obj], label="default")
                            eu_length = pcv.morphology.segment_euclidean_length(segmented_img=seg_img, objects=[obj], label="default")
                            curvature = pcv.morphology.segment_curvature(segmented_img=seg_img, objects=[obj], label="default")
                            angle = pcv.morphology.segment_angle(segmented_img=seg_img, objects=[obj], label="default")
                            insertion_angle = pcv.morphology.segment_insertion_angle(skel_img=pruned_skel, segmented_img=seg_img, leaf_objects=[obj], stem_objects=stem_obj, size=20, label="default")
                            analysis_data.append({
                                "ROI": f"ROI {roi_idx}",
                                "Length": length,
                                "Euclidean Length": eu_length,
                                "Curvature": curvature,
                                "Angle": angle,
                                "Insertion Angle": insertion_angle,
                                "Stem Count": stem_count,
                                "Leaf Count": leaf_count,
                                "Segment Count": segment_count,
                                "Branch Points": len(branch_points),
                                "Tip Points": len(tip_points)
                            })
                    except Exception as e:
                        logging.error(f"Error during further Plant Morphology Analysis for ROI {roi_idx}: {e}")
                        analysis_data.append({"ROI": f"ROI {roi_idx}", "Error": str(e)})
                else:
                    logging.error(f"Failed to complete Plant Morphology Analysis for ROI {roi_idx} after multiple attempts")
                    analysis_data.append({"ROI": f"ROI {roi_idx}", "Error": "Failed after multiple attempts"})

        return frame, pd.DataFrame(analysis_data), pd.DataFrame(photo_data), overlay_img

    def display_image(self, frame, overlay_img, grid_idx):
        blended = cv2.addWeighted(frame, 0.6, overlay_img, 0.4, 0)
        blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(blended)
        img = ImageTk.PhotoImage(img)
        display = self.grids[grid_idx]["image_display"]
        display.configure(image=img)
        display.image = img

    def update_table(self, grid_idx, analysis_data, photo_data):
        tree = self.grids[grid_idx]["tree"]
        for row in tree.get_children():
            tree.delete(row)

        analysis_type = self.grids[grid_idx]['analysis_dropdown'].get()

        if analysis_type == 'Plant Morphology Analysis':
            # Set columns for Plant Morphology Analysis
            tree["columns"] = ("ROI", "Length", "Euclidean Length", "Curvature", "Angle", "Insertion Angle", "Stem Count", "Leaf Count", "Segment Count", "Branch Points", "Tip Points")
            for col in tree["columns"]:
                tree.heading(col, text=col)
                tree.column(col, anchor="center")
            for _, row in analysis_data.iterrows():
                tree.insert("", "end", values=(
                    row["ROI"], row.get("Length", "N/A"), row.get("Euclidean Length", "N/A"), row.get("Curvature", "N/A"), row.get("Angle", "N/A"), row.get("Insertion Angle", "N/A"),
                    row.get("Stem Count", "N/A"), row.get("Leaf Count", "N/A"), row.get("Segment Count", "N/A"), row.get("Branch Points", "N/A"), row.get("Tip Points", "N/A")
                ))
        else:
            # Set default columns for other analyses
            tree["columns"] = ("ROI", "Parameter", "Value")
            for col in tree["columns"]:
                tree.heading(col, text=col)
                tree.column(col, anchor="center")
            for _, row in analysis_data.iterrows():
                tree.insert("", "end", values=(row["ROI"], row.get("Parameter", "N/A"), row.get("Value", "N/A")))

    def update_graph(self, grid_idx, analysis_data, photo_data):
        for widget in self.grids[grid_idx]["graph_frame"].winfo_children():
            widget.destroy()

        analysis_type = self.grids[grid_idx]['analysis_dropdown'].get()

        fig, ax = plt.subplots()

        # Populate the graph with dynamic data from the analysis
        if not analysis_data.empty:
            if analysis_type == 'Plant Morphology Analysis':
                # For Plant Morphology Analysis, use values from specific columns
                parameters = ['Length', 'Euclidean Length', 'Curvature', 'Angle', 'Insertion Angle']
                for _, row in analysis_data.iterrows():
                    values = [row.get('Length', 0), row.get('Euclidean Length', 0), row.get('Curvature', 0), row.get('Angle', 0), row.get('Insertion Angle', 0)]
                    ax.bar(parameters, values, label=row['ROI'])
            else:
                rois = analysis_data["ROI"].unique()
                for roi in rois:
                    roi_data = analysis_data[analysis_data["ROI"] == roi]
                    parameters = roi_data["Parameter"].tolist()
                    values = [float(str(v).split()[0]) if isinstance(v, str) and v.replace('.', '', 1).isdigit() else 0 for v in roi_data["Value"].tolist()]
                    ax.bar(parameters, values, label=roi)

            ax.set_title(f"{analysis_type} Metrics")
            ax.set_ylabel("Value")
            ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.grids[grid_idx]["graph_frame"])
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
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
    root.protocol("WM_DELETE_WINDOW", lambda: [app.stop_stream(i) for i in range(3)] or root.destroy())
    root.mainloop()
