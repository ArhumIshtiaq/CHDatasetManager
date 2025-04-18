import os
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import re
import csv
import datetime
import threading
import queue
import time # For demo/debug purposes if needed
import concurrent.futures # Added for parallel processing

# --- Try importing required libraries and provide guidance if missing ---
try:
    import cv2
    from PIL import Image, ImageTk
    from skimage.metrics import structural_similarity as ssim
    import numpy as np
except ImportError as e:
    messagebox.showerror(
        "Missing Dependencies",
        f"Error: Required library not found: {e.name}\n\n"
        "Please install required libraries using pip:\n"
        "pip install opencv-python Pillow scikit-image numpy"
    )
    # Exit if dependencies are missing, otherwise the app will crash later
    import sys
    sys.exit(1)

# --- Constants ---
THUMBNAIL_WIDTH = 160 # Width for the animation display area
THUMBNAIL_HEIGHT = 90  # Height for the animation display area
LOG_FILE = "verification_log.csv"
# Keyframes for SSIM calculation (Optimized)
SSIM_KEYFRAME_PERCENTAGES = [0.25, 0.33, 0.5, 0.66, 0.75]
NUM_SSIM_KEYFRAMES = len(SSIM_KEYFRAME_PERCENTAGES)
SSIM_RESIZE_WIDTH = 160
SSIM_RESIZE_HEIGHT = 90
# Preview Frames Configuration
NUM_PREVIEW_FRAMES = 5
PREVIEW_START_PERC = 0.25 # Start preview sampling at 25%
PREVIEW_END_PERC = 0.75   # End preview sampling at 75%
PREVIEW_FRAME_PERCENTAGES = np.linspace(PREVIEW_START_PERC, PREVIEW_END_PERC, NUM_PREVIEW_FRAMES)
PREVIEW_ANIMATION_DELAY = 250 # Milliseconds between preview frames (Adjust as needed)

MAX_VIDEOS = 4
PRE_MARKING_SD_FACTOR = 0.5 # Factor for SD-based pre-marking
PRE_MARKING_SCORE_THRESHOLD = 0.85

class VideoPlacerApp:
    """
    GUI application to place up to 4 video takes into a folder structure.
    Base Directory and Interpreter ID are set once per session.
    Includes automatic similarity scoring (Per-Video Multi-Frame SSIM),
    displays an animated preview using sampled frames, pre-marks videos
    close to the max score, allows manual per-video approval, and logs
    initial/final states. Processes only approved videos.
    Renames files as InterpreterID_TakeNumber.ext.
    """
    def __init__(self, master):
        self.master = master
        master.title("Video Placement Helper (Animated Preview)") # Updated title
        master.geometry("600x720") # Adjusted height if needed

        # --- State Variables ---
        self.initial_setup_done = False

        # --- Tkinter Variables ---
        self.base_directory = tk.StringVar()
        self.selected_interpreter_id = tk.StringVar()
        self.selected_col_a = tk.StringVar()
        self.selected_col_b = tk.StringVar()
        self.selected_file_paths_tuple = ()
        self.selected_files_info = tk.StringVar(value="No files selected")
        self.status_message = tk.StringVar(value="Step 1: Select Base Directory")
        self.take_assignment_display = tk.StringVar(value="Takes: -")
        # Analysis/verification variables
        self.per_video_similarity_scores = [None] * MAX_VIDEOS
        self.score_display_vars = [tk.StringVar(value="Score: -") for _ in range(MAX_VIDEOS)]
        self.per_video_confirmed_vars = [tk.BooleanVar(value=False) for _ in range(MAX_VIDEOS)]
        self.initial_confirmation_state = [False] * MAX_VIDEOS
        # Store references to the multiple preview PhotoImage objects (list of lists)
        self.preview_photo_images = [[] for _ in range(MAX_VIDEOS)]
        # Store current animation frame index for each slot
        self.preview_animation_index = [0] * MAX_VIDEOS
        # Store IDs returned by master.after for cancellation
        self.preview_after_ids = [None] * MAX_VIDEOS
        self.analysis_queue = queue.Queue()
        self.is_analysis_running = False

        # --- Style ---
        style = ttk.Style(); style.configure("TLabel", padding=5, font=('Helvetica', 10)); style.configure("TButton", padding=5, font=('Helvetica', 10)); style.configure("TCombobox", padding=5, font=('Helvetica', 10)); style.configure("TEntry", padding=5, font=('Helvetica', 10)); style.configure("Status.TLabel", font=('Helvetica', 9), foreground="grey"); style.configure("TakeAssign.TLabel", font=('Helvetica', 10, 'bold'), foreground="blue"); style.configure("Score.TLabel", font=('Helvetica', 9, 'bold'), foreground="green"); style.configure("Confirm.TCheckbutton", font=('Helvetica', 9))

        # --- GUI Layout ---
        row_basedir = 0; row_id = 1; row_cola = 2; row_colb = 3
        row_fileselect = 4; row_verification_area = 5; row_analysis_info = 6
        row_processbtn = 7; row_status = 8

        # --- Widgets ---
        # (Dropdowns and file selection remain mostly the same)
        ttk.Label(master, text="Base Directory:").grid(row=row_basedir, column=0, sticky=tk.W, padx=10, pady=5)
        self.base_dir_entry = ttk.Entry(master, textvariable=self.base_directory, width=45, state='readonly'); self.base_dir_entry.grid(row=row_basedir, column=1, columnspan=2, padx=5, pady=5)
        self.base_dir_button = ttk.Button(master, text="Browse...", command=self.select_base_dir); self.base_dir_button.grid(row=row_basedir, column=3, padx=10, pady=5)
        ttk.Label(master, text="Interpreter ID:").grid(row=row_id, column=0, sticky=tk.W, padx=10, pady=5)
        interpreter_ids = [f"{i:03d}" for i in range(1, 11)]
        self.interpreter_id_combobox = ttk.Combobox(master, textvariable=self.selected_interpreter_id, values=interpreter_ids, width=43, state='disabled'); self.interpreter_id_combobox.grid(row=row_id, column=1, columnspan=2, padx=5, pady=5); self.interpreter_id_combobox.bind("<<ComboboxSelected>>", self.on_id_select)
        ttk.Label(master, text="Category (Col A):").grid(row=row_cola, column=0, sticky=tk.W, padx=10, pady=5)
        self.col_a_combobox = ttk.Combobox(master, textvariable=self.selected_col_a, width=43, state='disabled'); self.col_a_combobox.grid(row=row_cola, column=1, columnspan=2, padx=5, pady=5); self.col_a_combobox.bind("<<ComboboxSelected>>", self.on_col_a_select)
        ttk.Label(master, text="Session/Item (Col B):").grid(row=row_colb, column=0, sticky=tk.W, padx=10, pady=5)
        self.col_b_combobox = ttk.Combobox(master, textvariable=self.selected_col_b, width=43, state='disabled'); self.col_b_combobox.grid(row=row_colb, column=1, columnspan=2, padx=5, pady=5); self.col_b_combobox.bind("<<ComboboxSelected>>", self.on_col_b_select)
        ttk.Label(master, text="Selected Files:").grid(row=row_fileselect, column=0, sticky=tk.W, padx=10, pady=5)
        self.files_info_entry = ttk.Entry(master, textvariable=self.selected_files_info, width=45, state='readonly'); self.files_info_entry.grid(row=row_fileselect, column=1, columnspan=2, padx=5, pady=5)
        self.select_files_button = ttk.Button(master, text="Select Files...", command=self.select_video_files, state='disabled'); self.select_files_button.grid(row=row_fileselect, column=3, padx=10, pady=5)

        # --- Verification Area (Animated Preview, Scores, Checkboxes) ---
        ttk.Label(master, text="Review & Approve Takes:").grid(row=row_verification_area, column=0, sticky="nw", padx=10, pady=(10, 0))
        self.verification_frame = ttk.Frame(master, borderwidth=1, relief="sunken"); self.verification_frame.grid(row=row_verification_area, column=1, columnspan=3, sticky="nsew", padx=10, pady=(5,5))

        # Use single label per slot for animation
        self.preview_labels = []
        self.score_labels = []
        self.confirm_checkboxes = []

        for i in range(MAX_VIDEOS):
            item_frame = ttk.Frame(self.verification_frame); item_frame.grid(row=0, column=i, padx=5, pady=5, sticky="n"); self.verification_frame.columnconfigure(i, weight=1)
            # Single Label for Animated Preview
            preview_label = ttk.Label(item_frame) # Size determined by image
            preview_label.pack(pady=(0,2))
            self.preview_labels.append(preview_label)
            # Score Label
            score_label = ttk.Label(item_frame, textvariable=self.score_display_vars[i], style="Score.TLabel"); score_label.pack(pady=(0,2)); self.score_labels.append(score_label)
            # Checkbox
            confirm_cb = ttk.Checkbutton(item_frame, text="Approve", variable=self.per_video_confirmed_vars[i], onvalue=True, offvalue=False, command=self.check_button_state, style="Confirm.TCheckbutton", state='disabled'); confirm_cb.pack(pady=(0,5)); self.confirm_checkboxes.append(confirm_cb)

        # Take Assignment Display
        ttk.Label(master, textvariable=self.take_assignment_display, style="TakeAssign.TLabel").grid(row=row_analysis_info, column=1, columnspan=3, sticky="w", padx=10, pady=0)

        # Process Button
        self.process_button = ttk.Button(master, text="Place Approved Files", command=self.process_selected_videos, state='disabled')
        self.process_button.grid(row=row_processbtn, column=1, columnspan=2, pady=15)

        # Status Label
        ttk.Label(master, textvariable=self.status_message, style="Status.TLabel").grid(row=row_status, column=0, columnspan=4, sticky="ew", padx=10, pady=10)

        # --- Configure Grid ---
        master.columnconfigure(1, weight=1); master.columnconfigure(2, weight=1)
        self.master.after(100, self.check_analysis_queue) # Start queue checker
        # Bind closing event
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)


    # --- Helper Functions ---
    def clear_analysis_results(self):
        """Clears previous analysis results from GUI and stops animations."""
        # Stop any running animations
        for i in range(MAX_VIDEOS):
            if self.preview_after_ids[i] is not None:
                try:
                    self.master.after_cancel(self.preview_after_ids[i])
                except tk.TclError:
                    pass # Job might have already run or been cancelled
                self.preview_after_ids[i] = None

        self.per_video_similarity_scores = [None] * MAX_VIDEOS
        for var in self.score_display_vars: var.set("Score: -")
        self.take_assignment_display.set("Takes: -")
        self.initial_confirmation_state = [False] * MAX_VIDEOS
        for i in range(MAX_VIDEOS):
            self.per_video_confirmed_vars[i].set(False)
            if i < len(self.confirm_checkboxes): self.confirm_checkboxes[i].config(state='disabled')
        # Clear preview images and labels
        self.preview_photo_images = [[] for _ in range(MAX_VIDEOS)] # Reset list of lists
        for label in self.preview_labels: # Use the single preview labels now
            label.config(image='')
        self.preview_animation_index = [0] * MAX_VIDEOS # Reset animation index

    # select_base_dir, on_id_select, populate_col_a, on_col_a_select, populate_col_b, on_col_b_select
    # remain the same as the previous "set-once" version
    def select_base_dir(self):
        if self.initial_setup_done: return
        directory = filedialog.askdirectory(title="Select Base Directory (Set Once)")
        if directory:
            self.base_directory.set(directory); self.status_message.set("Step 2: Select your Interpreter ID")
            self.interpreter_id_combobox.config(state='readonly'); self.base_dir_button.config(state='disabled')
            self.col_a_combobox.config(state='disabled'); self.col_b_combobox.config(state='disabled'); self.select_files_button.config(state='disabled')
        else: self.status_message.set("Base directory selection cancelled. Please select Base Directory.")

    def on_id_select(self, event=None):
        if not self.selected_interpreter_id.get(): return
        if self.initial_setup_done: return
        self.initial_setup_done = True; self.interpreter_id_combobox.config(state='disabled')
        self.status_message.set("Step 3: Select Category"); self.populate_col_a()
        self.col_a_combobox.config(state='readonly' if self.col_a_combobox['values'] else 'disabled')
        self.selected_col_a.set(""); self.selected_col_b.set("")
        self.col_b_combobox.set(""); self.col_b_combobox['values'] = []; self.col_b_combobox.config(state='disabled')
        self.selected_file_paths_tuple = (); self.selected_files_info.set("No files selected")
        self.clear_analysis_results(); self.check_button_state()

    def populate_col_a(self):
        base_dir = self.base_directory.get()
        if not base_dir or not os.path.isdir(base_dir): self.col_a_combobox['values'] = []; return
        try:
            opts = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
            self.col_a_combobox['values'] = opts
            if not opts: self.status_message.set("No category folders found.")
        except Exception as e: messagebox.showerror("Error", f"Failed to read base directory:\n{e}"); self.col_a_combobox['values'] = []; self.status_message.set("Error reading base directory.")

    def on_col_a_select(self, event=None):
        if not self.initial_setup_done: return
        self.selected_col_b.set(""); self.col_b_combobox.set("")
        self.selected_file_paths_tuple = (); self.selected_files_info.set("No files selected")
        self.clear_analysis_results(); self.populate_col_b()
        self.col_b_combobox.config(state='readonly' if self.col_b_combobox['values'] else 'disabled')
        if self.col_b_combobox['values']: self.status_message.set("Step 4: Select Word")
        else: self.status_message.set("No Session/Item folders found for this Category.")
        self.select_files_button.config(state='disabled'); self.check_button_state()

    def populate_col_b(self):
        base_dir = self.base_directory.get(); col_a = self.selected_col_a.get()
        if not base_dir or not col_a: self.col_b_combobox['values'] = []; return
        col_a_path = os.path.join(base_dir, col_a)
        if not os.path.isdir(col_a_path): self.col_b_combobox['values'] = []; return
        try:
            opts = sorted([d for d in os.listdir(col_a_path) if os.path.isdir(os.path.join(col_a_path, d))])
            self.col_b_combobox['values'] = opts
        except Exception as e: messagebox.showerror("Error", f"Failed to read category folder '{col_a}':\n{e}"); self.col_b_combobox['values'] = []

    def on_col_b_select(self, event=None):
        if not self.initial_setup_done: return
        self.selected_file_paths_tuple = (); self.selected_files_info.set("No files selected")
        self.clear_analysis_results()
        if self.selected_col_b.get(): self.select_files_button.config(state='normal'); self.status_message.set("Step 5: Select video file(s)")
        else: self.select_files_button.config(state='disabled'); self.status_message.set("Select Word")
        self.check_button_state()


    # --- Video File Selection and Analysis Trigger ---
    # (Modified to call clear_analysis_results which stops animations)
    def select_video_files(self):
        """Opens dialog to select multiple video files and starts analysis thread."""
        if not self.selected_col_b.get(): messagebox.showwarning("Selection Missing", "Please select Word first."); return
        if self.is_analysis_running: messagebox.showwarning("Busy", "Analysis is already in progress."); return

        video_types = [("Video Files", "*.mp4 *.avi *.mov *.wmv *.mkv *.flv"), ("All Files", "*.*")]
        filepaths = filedialog.askopenfilenames(title=f"Select up to {MAX_VIDEOS} Video Files", filetypes=video_types)

        if filepaths:
            if len(filepaths) > MAX_VIDEOS: messagebox.showwarning("Too Many Files", f"Please select a maximum of {MAX_VIDEOS} files."); return
            # --- Stop previous animations before starting new analysis ---
            self.clear_analysis_results() # This now also cancels animations

            self.selected_file_paths_tuple = filepaths; num_selected = len(filepaths)
            self.selected_files_info.set(f"{num_selected} file(s) selected"); self.status_message.set("Analyzing videos...")
            self.check_button_state(); self.is_analysis_running = True
            analysis_thread = threading.Thread(target=self.analyze_videos_thread, args=(filepaths,), daemon=True)
            analysis_thread.start()
        else: self.selected_file_paths_tuple = (); self.selected_files_info.set("No files selected"); self.clear_analysis_results(); self.status_message.set("Video file selection cancelled."); self.check_button_state()


    # --- Helper Functions for Video Analysis ---
    def _extract_frames(self, video_path):
        """Extracts SSIM keyframes and preview frames from a single video file."""
        keyframes_for_ssim = [None] * NUM_SSIM_KEYFRAMES
        preview_pil_images = [None] * NUM_PREVIEW_FRAMES
        error_messages = []
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                error_messages.append(f"Error opening: {os.path.basename(video_path)}")
                return keyframes_for_ssim, preview_pil_images, error_messages

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count < 2:
                error_messages.append(f"Not enough frames: {os.path.basename(video_path)}")
                return keyframes_for_ssim, preview_pil_images, error_messages

            # Extract SSIM Keyframes
            for kf_idx, percentage in enumerate(SSIM_KEYFRAME_PERCENTAGES):
                frame_idx = max(0, min(int(frame_count * percentage), frame_count - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret_key, frame_key = cap.read()
                if ret_key and frame_key is not None:
                    gray_frame = cv2.cvtColor(frame_key, cv2.COLOR_BGR2GRAY)
                    resized_gray_frame = cv2.resize(gray_frame, (SSIM_RESIZE_WIDTH, SSIM_RESIZE_HEIGHT), interpolation=cv2.INTER_AREA)
                    keyframes_for_ssim[kf_idx] = resized_gray_frame
                else:
                    error_messages.append(f"Error reading SSIM keyframe {kf_idx+1}: {os.path.basename(video_path)}")

            # Extract Preview Frames
            for pv_idx, percentage in enumerate(PREVIEW_FRAME_PERCENTAGES):
                frame_idx = max(0, min(int(frame_count * percentage), frame_count - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret_prev, frame_prev = cap.read()
                if ret_prev and frame_prev is not None:
                    preview_frame_resized = cv2.resize(frame_prev, (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), interpolation=cv2.INTER_AREA)
                    preview_pil_images[pv_idx] = Image.fromarray(cv2.cvtColor(preview_frame_resized, cv2.COLOR_BGR2RGB))
                else:
                    error_messages.append(f"Error reading preview frame {pv_idx+1}: {os.path.basename(video_path)}")

        except Exception as e:
            error_messages.append(f"Error processing {os.path.basename(video_path)}: {e}")
        finally:
            if cap is not None and cap.isOpened():
                cap.release()

        return keyframes_for_ssim, preview_pil_images, error_messages

    def _calculate_ssim_scores(self, all_keyframes):
        """Calculates average SSIM scores between videos based on extracted keyframes."""
        num_videos = len(all_keyframes)
        per_video_avg_scores = [None] * num_videos
        error_messages = []

        valid_video_indices = [i for i, kf_list in enumerate(all_keyframes) if any(kf is not None for kf in kf_list)]

        if len(valid_video_indices) > 1:
            video_score_sums = {idx: 0.0 for idx in valid_video_indices}
            video_pair_counts = {idx: 0 for idx in valid_video_indices}
            try:
                for kf_idx in range(NUM_SSIM_KEYFRAMES):
                    # Collect frames available at this keyframe index, keeping track of original video index
                    frames_at_kf_with_orig_idx = [
                        {'frame': all_keyframes[vid_idx][kf_idx], 'orig_idx': vid_idx}
                        for vid_idx in valid_video_indices
                        if all_keyframes[vid_idx][kf_idx] is not None
                    ]

                    if len(frames_at_kf_with_orig_idx) > 1:
                        # Compare all pairs at this keyframe index
                        for i in range(len(frames_at_kf_with_orig_idx)):
                            for j in range(i + 1, len(frames_at_kf_with_orig_idx)):
                                frame_i_data = frames_at_kf_with_orig_idx[i]
                                frame_j_data = frames_at_kf_with_orig_idx[j]
                                frame_i = frame_i_data['frame']
                                frame_j = frame_j_data['frame']
                                orig_idx_i = frame_i_data['orig_idx']
                                orig_idx_j = frame_j_data['orig_idx']

                                # Ensure frames have the same dimensions for SSIM
                                # h1, w1 = frame_i.shape
                                # h2, w2 = frame_j.shape
                                # if h1 != h2 or w1 != w2:
                                #     # Resize frame_j to match frame_i dimensions
                                #     # This is redundant as frames are pre-resized in _extract_frames
                                #     # frame_j = cv2.resize(frame_j, (w1, h1), interpolation=cv2.INTER_AREA)

                                # Calculate SSIM (Frames are guaranteed to be the same size now)
                                score = ssim(frame_i, frame_j, data_range=frame_i.max() - frame_i.min())

                                # Accumulate scores and counts for averaging later
                                video_score_sums[orig_idx_i] += score
                                video_score_sums[orig_idx_j] += score
                                video_pair_counts[orig_idx_i] += 1
                                video_pair_counts[orig_idx_j] += 1

                # Calculate average score for each video
                for vid_idx in valid_video_indices:
                    if video_pair_counts[vid_idx] > 0:
                        per_video_avg_scores[vid_idx] = video_score_sums[vid_idx] / video_pair_counts[vid_idx]
                    else:
                        # If a video had no pairs (e.g., only one valid video, or issues reading its frames),
                        # assign a default score (e.g., 1.0 or handle as needed)
                        per_video_avg_scores[vid_idx] = 1.0 # Default for single valid video or isolated frame issues

            except Exception as e:
                error_messages.append(f"Error calculating multi-frame SSIM: {e}")
                # Optionally invalidate scores if a critical error occurs
                # per_video_avg_scores = [None] * num_videos

        elif len(valid_video_indices) == 1:
            # If only one video has valid frames, its score relative to others is undefined, assign default
            per_video_avg_scores[valid_video_indices[0]] = 1.0

        return per_video_avg_scores, error_messages

    # --- Background Thread for Analysis (Refactored for Parallel Frame Extraction) ---
    def analyze_videos_thread(self, filepaths):
        """
        Worker function: Coordinates frame extraction (in parallel) and SSIM calculation,
        puts results in queue.
        """
        num_videos = len(filepaths)
        all_preview_pil_images = [[] for _ in range(num_videos)]
        all_keyframes_for_ssim = [[] for _ in range(num_videos)]
        all_error_messages = []

        # Step 1: Extract frames for all videos in parallel
        results = [None] * num_videos # Pre-allocate list for results
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks and store futures with original index
            future_to_index = {executor.submit(self._extract_frames, path): i for i, path in enumerate(filepaths)}

            # Process completed futures as they finish
            for future in concurrent.futures.as_completed(future_to_index):
                original_index = future_to_index[future]
                try:
                    keyframes, previews, errors = future.result()
                    results[original_index] = (keyframes, previews, errors)
                except Exception as exc:
                    # Handle exceptions during the task execution itself
                    filename = os.path.basename(filepaths[original_index])
                    all_error_messages.append(f"Error processing {filename} in thread: {exc}")
                    # Store placeholder or handle error state for this video
                    results[original_index] = ([], [], [f"Task failed for {filename}"])

        # Unpack results, maintaining original order
        for idx in range(num_videos):
            if results[idx]: # Check if result was successfully stored
                keyframes, previews, errors = results[idx]
                all_keyframes_for_ssim[idx] = keyframes
                all_preview_pil_images[idx] = previews
                if errors:
                    all_error_messages.extend(errors)
            # else: Error already logged during future processing

        # Step 2: Calculate SSIM scores using the extracted keyframes
        per_video_avg_scores, ssim_errors = self._calculate_ssim_scores(all_keyframes_for_ssim)
        if ssim_errors:
            all_error_messages.extend(ssim_errors)

        # Step 3: Put results into the queue for the main thread
        self.analysis_queue.put({
            'type': 'analysis_complete',
            'scores': per_video_avg_scores,
            'previews': all_preview_pil_images,
            'errors': all_error_messages,
            'filepaths': filepaths # Pass filepaths back for context
        })


    # --- MODIFIED Queue Checking (Runs in Main Thread) ---
    def check_analysis_queue(self):
        """Checks the queue for results, displays scores, starts animations, pre-marks."""
        try:
            result = self.analysis_queue.get_nowait()

            # Check the type of message from the queue
            if result.get('type') == 'analysis_complete':
                self.is_analysis_running = False
                # Retrieve results using the new keys
                list_of_preview_pil_list = result.get("previews", []) # Use .get with default
                self.per_video_similarity_scores = result.get("scores", [None] * MAX_VIDEOS)
                errors = result.get("errors", [])
                filepaths = result.get("filepaths", []) # Get filepaths for context

                # --- Update Score Display --- (Logic remains similar)
                valid_scores = [s for s in self.per_video_similarity_scores if s is not None]
                max_score = 0.0
                if valid_scores: max_score = max(valid_scores)

                for i, score_var in enumerate(self.score_display_vars):
                    # Ensure index is within bounds of scores list
                    if i < len(self.per_video_similarity_scores):
                        current_score = self.per_video_similarity_scores[i]
                        if current_score is not None:
                            score_var.set(f"Score: {current_score:.3f}")
                        else:
                            score_var.set("Score: N/A")
                    else:
                        score_var.set("Score: -") # Handle cases where fewer scores than slots

                # --- Create PhotoImage objects for previews and enable checkboxes --- (Logic remains similar)
                self.preview_photo_images = [[] for _ in range(MAX_VIDEOS)] # Reset list of lists
                num_videos_with_previews = 0
                checkbox_states_after_load = {}

                for video_idx in range(MAX_VIDEOS):
                    photo_images_for_video = [] # Store PhotoImages for this video's animation
                    preview_success = False # Track if at least one preview frame loaded
                    # Ensure index is within bounds of previews list
                    if video_idx < len(list_of_preview_pil_list):
                        pil_images_for_video = list_of_preview_pil_list[video_idx]

                        for frame_idx in range(NUM_PREVIEW_FRAMES):
                            # Ensure frame index is within bounds
                            pil_image = pil_images_for_video[frame_idx] if frame_idx < len(pil_images_for_video) else None
                            if pil_image is not None:
                                try:
                                    photo_img = ImageTk.PhotoImage(pil_image)
                                    photo_images_for_video.append(photo_img) # Store reference
                                    preview_success = True
                                except Exception as e:
                                    print(f"Error creating PhotoImage for preview: {e}")
                                    photo_images_for_video.append(None) # Add placeholder on error
                            else:
                                photo_images_for_video.append(None) # Add placeholder if no PIL image
                    # else: No previews for this slot if index out of bounds

                    self.preview_photo_images[video_idx] = photo_images_for_video # Store list of PhotoImages

                    # Determine checkbox state based on preview success for this video
                    checkbox_state = 'normal' if preview_success else 'disabled'
                    checkbox_states_after_load[video_idx] = checkbox_state
                    if video_idx < len(self.confirm_checkboxes):
                        self.confirm_checkboxes[video_idx].config(state=checkbox_state)
                        if checkbox_state == 'disabled':
                            self.per_video_confirmed_vars[video_idx].set(False)

                    if preview_success:
                        num_videos_with_previews += 1
                        # --- Start Animation for this slot --- (Logic remains the same)
                        self.start_preview_animation(video_idx)

                # --- Pre-mark Checkbox(es) based on Standard Deviation --- (Logic remains similar)
                for i in range(MAX_VIDEOS): self.per_video_confirmed_vars[i].set(False) # Reset first
                if num_videos_with_previews > 1 and valid_scores:
                    std_dev = np.std(valid_scores) if len(valid_scores) >= 2 else 0
                    score_threshold = max_score - PRE_MARKING_SD_FACTOR * std_dev
                    print(f"Pre-marking check: MaxScore={max_score:.3f}, SD={std_dev:.3f}, Threshold={score_threshold:.3f}") # Debug
                    # Ensure index is within bounds of scores list
                    for i in range(len(self.per_video_similarity_scores)):
                        current_score = self.per_video_similarity_scores[i]
                        is_enabled = checkbox_states_after_load.get(i) == 'normal'
                        # Check score and threshold, ensure it's enabled
                        if current_score is not None and (current_score >= score_threshold or current_score >= PRE_MARKING_SCORE_THRESHOLD) and is_enabled:
                            print(f"Pre-marking video index {i} (Score: {current_score:.3f})") # Debug
                            self.per_video_confirmed_vars[i].set(True)

                # --- Store Initial Confirmation State for Logging --- (Use filepaths length)
                num_selected = len(filepaths) # Use length of filepaths from result
                self.initial_confirmation_state = [self.per_video_confirmed_vars[i].get() if i < MAX_VIDEOS else False for i in range(num_selected)]

                # Report errors from thread
                if errors: messagebox.showwarning("Analysis Issues", "Encountered issues during analysis:\n- " + "\n- ".join(errors))

                # Calculate take assignment ONLY if analysis produced results
                if num_videos_with_previews > 0: self.calculate_and_display_take_assignment()
                else: self.take_assignment_display.set("Takes: Error"); self.status_message.set("Analysis failed for all videos.")

                self.check_button_state() # Update button state based on results
            # else: Handle other message types if needed in the future

        except queue.Empty: pass
        finally: self.master.after(100, self.check_analysis_queue)


    # --- NEW Animation Functions ---
    def start_preview_animation(self, video_idx):
        """Starts or restarts the animation loop for a specific video slot."""
        # Cancel any previous loop for this slot
        if self.preview_after_ids[video_idx] is not None:
            try:
                self.master.after_cancel(self.preview_after_ids[video_idx])
            except tk.TclError: pass # Ignore if job doesn't exist
            self.preview_after_ids[video_idx] = None

        # Reset index and start the update cycle
        self.preview_animation_index[video_idx] = 0
        # Check if the preview label exists before trying to configure it
        if video_idx < len(self.preview_labels):
            self.update_preview_animation(video_idx)
        else:
            print(f"Warning: Preview label for index {video_idx} not found.")


    def update_preview_animation(self, video_idx):
        """Updates the preview label with the next frame and schedules the next update."""
        # Check if component lists are initialized and index is valid
        if not hasattr(self, 'preview_photo_images') or \
           video_idx >= len(self.preview_photo_images) or \
           video_idx >= len(self.preview_labels) or \
           video_idx >= len(self.preview_animation_index) or \
           video_idx >= len(self.preview_after_ids):
             print(f"Warning: Animation component missing for index {video_idx}. Stopping animation.")
             return

        photo_list = self.preview_photo_images[video_idx]
        # Filter out None placeholders (frames that failed to load/create PhotoImage)
        valid_photo_list = [img for img in photo_list if img is not None]

        if not valid_photo_list: # No valid images loaded for this slot
            self.preview_labels[video_idx].config(image='') # Ensure label is clear
            self.preview_after_ids[video_idx] = None # Stop loop
            return

        # Cycle through only the valid images
        num_valid_frames = len(valid_photo_list)
        current_frame_idx = self.preview_animation_index[video_idx] % num_valid_frames
        photo_to_display = valid_photo_list[current_frame_idx]

        # Update the corresponding preview label
        try:
            self.preview_labels[video_idx].config(image=photo_to_display)
        except tk.TclError as e:
            print(f"Error updating preview label {video_idx}: {e}")
            self.preview_after_ids[video_idx] = None # Stop loop on error
            return


        # Increment index for the next cycle
        self.preview_animation_index[video_idx] = (self.preview_animation_index[video_idx] + 1)

        # Schedule the next update
        self.preview_after_ids[video_idx] = self.master.after(
            PREVIEW_ANIMATION_DELAY, self.update_preview_animation, video_idx
        )

    # --- Take Calculation ---
    # (Remains the same as previous version)
    def calculate_and_display_take_assignment(self):
        """Calculates the available take range based on existing files and number selected."""
        base_dir = self.base_directory.get(); col_a = self.selected_col_a.get(); col_b = self.selected_col_b.get(); interpreter_id = self.selected_interpreter_id.get()
        num_selected = len(self.selected_file_paths_tuple)
        if num_selected == 0: self.take_assignment_display.set("Takes: -"); return
        if not all([base_dir, col_a, col_b, interpreter_id]): self.take_assignment_display.set("Takes: Error"); self.status_message.set("Error: Prerequisite selection missing."); return
        target_folder_path = os.path.join(base_dir, col_a, col_b, interpreter_id)
        highest_take = 0; start_take = 1
        if os.path.isdir(target_folder_path):
            try:
                pattern = re.compile(re.escape(f"{interpreter_id}_") + r"([1-4])\..+$")
                for filename in os.listdir(target_folder_path):
                     match = pattern.match(filename)
                     if match:
                        try: highest_take = max(highest_take, int(match.group(1)))
                        except ValueError: continue
                start_take = highest_take + 1
            except Exception as e: self.take_assignment_display.set("Takes: Error"); self.status_message.set(f"Error checking existing takes: {e}"); return
        else: start_take = 1
        if start_take > 4: self.take_assignment_display.set("Takes: FULL (4/4)"); self.status_message.set("Error: Maximum 4 takes already exist.")
        else:
            end_take = start_take + num_selected - 1; available_slots = 4 - start_take + 1
            if end_take > 4: self.take_assignment_display.set(f"Takes: Error (Need {num_selected}, Avail: {available_slots})"); self.status_message.set(f"Error: Too many files selected for available takes (Start: {start_take}). Approve carefully.")
            else:
                if num_selected == 1: self.take_assignment_display.set(f"Potential Take: {start_take}")
                else: self.take_assignment_display.set(f"Potential Takes: {start_take}-{end_take}")
                if "Analyzing" not in self.status_message.get() and "Error" not in self.status_message.get(): self.status_message.set(f"Ready for approval. Approve videos below.")


    # --- Button State Check ---
    # (Remains the same as previous version)
    def check_button_state(self):
        """Enables or disables widgets based on the application state (Set-Once Workflow)."""
        # ... (code identical to previous version) ...
        if not self.initial_setup_done:
            self.base_dir_button.config(state='normal' if not self.base_directory.get() else 'disabled')
            self.interpreter_id_combobox.config(state='readonly' if self.base_directory.get() else 'disabled')
            self.col_a_combobox.config(state='disabled'); self.col_b_combobox.config(state='disabled'); self.select_files_button.config(state='disabled'); self.process_button.config(state='disabled')
            for cb in self.confirm_checkboxes: cb.config(state='disabled')
            return
        self.base_dir_button.config(state='disabled'); self.interpreter_id_combobox.config(state='disabled')
        self.col_a_combobox.config(state='readonly' if self.col_a_combobox['values'] else 'disabled')
        self.col_b_combobox.config(state='readonly' if self.selected_col_a.get() and self.col_b_combobox['values'] else 'disabled')
        self.select_files_button.config(state='normal' if self.selected_col_b.get() else 'disabled')
        take_info = self.take_assignment_display.get(); valid_take_assignment = not take_info.startswith("Takes: FULL") and not take_info.startswith("Takes: Error") and take_info != "Takes: -"
        files_selected = len(self.selected_file_paths_tuple) > 0
        at_least_one_confirmed = any(var.get() for i, var in enumerate(self.per_video_confirmed_vars) if i < len(self.selected_file_paths_tuple))
        can_process = not self.is_analysis_running
        if (files_selected and valid_take_assignment and at_least_one_confirmed and can_process): self.process_button.config(state='normal')
        else: self.process_button.config(state='disabled')


    # --- Logging Function ---
    # (Remains the same as previous version)
    def log_verification_data(self, processed_indices, assigned_take_numbers):
        """Logs verification details including pre-marked and final confirmation status."""
        # ... (code identical to previous version logging initial_confirmation_state) ...
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"); num_selected = len(self.selected_file_paths_tuple)
        final_confirmation_states = [self.per_video_confirmed_vars[i].get() for i in range(num_selected)]; final_confirmation_str = "; ".join(map(str, final_confirmation_states))
        initial_confirmation_str = "; ".join(map(str, self.initial_confirmation_state)) # Use stored initial state
        scores_str = "; ".join([f"{s:.4f}" if s is not None else "N/A" for s in self.per_video_similarity_scores[:num_selected]])
        original_filenames_str = "; ".join(os.path.basename(p) for p in self.selected_file_paths_tuple)
        assigned_takes_str = "; ".join(map(str, assigned_take_numbers))
        log_entry = {
            "Timestamp": timestamp, "BaseDirectory": self.base_directory.get(), "Category": self.selected_col_a.get(), "Session": self.selected_col_b.get(),
            "InterpreterID": self.selected_interpreter_id.get(), "NumFilesSelected": num_selected, "OriginalFileNames": original_filenames_str,
            "PerVideoScores": scores_str, "PreMarkedConfirmation": initial_confirmation_str, "FinalConfirmation": final_confirmation_str,
            "ProcessedIndices": "; ".join(map(str, processed_indices)), "AssignedTakeNumbers": assigned_takes_str
        }
        try:
            file_exists = os.path.isfile(LOG_FILE); fieldnames = list(log_entry.keys())
            with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames);
                if not file_exists or os.path.getsize(LOG_FILE) == 0: writer.writeheader()
                writer.writerow(log_entry)
            print(f"Logged verification data to {LOG_FILE}")
        except Exception as e: print(f"Error logging data: {e}"); self.status_message.set("Error logging data (check console).")


    # --- Processing Logic ---
    # (Remains the same as previous version)
    def process_selected_videos(self):
        """Handles moving and renaming ONLY the individually approved video files."""
        # ... (code identical to previous version) ...
        if self.is_analysis_running: messagebox.showwarning("Busy", "Analysis is in progress."); return
        selected_files = self.selected_file_paths_tuple; num_selected = len(selected_files)
        if num_selected == 0: messagebox.showerror("Error", "No video files selected."); return
        confirmed_indices = [i for i, var in enumerate(self.per_video_confirmed_vars) if i < num_selected and var.get()]
        if not confirmed_indices: messagebox.showerror("No Videos Approved", "Please approve at least one video using the 'Approve' checkbox below it."); return
        base_dir = self.base_directory.get(); col_a = self.selected_col_a.get(); col_b = self.selected_col_b.get(); interpreter_id = self.selected_interpreter_id.get()
        target_folder_path = os.path.join(base_dir, col_a, col_b, interpreter_id)
        highest_take = 0; start_take = 1
        if os.path.isdir(target_folder_path):
             try:
                pattern = re.compile(re.escape(f"{interpreter_id}_") + r"([1-4])\..+$")
                for filename in os.listdir(target_folder_path):
                     match = pattern.match(filename)
                     if match:
                        try: highest_take = max(highest_take, int(match.group(1)))
                        except ValueError: continue
                start_take = highest_take + 1
             except Exception as e: messagebox.showerror("Error", f"Failed to re-verify existing takes: {e}"); return
        else: start_take = 1
        num_approved = len(confirmed_indices); final_take_needed = start_take + num_approved - 1
        if final_take_needed > 4: available_slots = 4 - start_take + 1; messagebox.showerror("Error", f"Cannot process. Too many videos approved ({num_approved}) for available slots (Max {available_slots}, starting from take {start_take}). Please uncheck some."); self.check_button_state(); return
        if not os.path.isdir(target_folder_path):
             try: os.makedirs(target_folder_path); print(f"Created missing folder: {target_folder_path}")
             except Exception as e: messagebox.showerror("Error", f"Could not create target folder: {e}"); return
        assigned_take_numbers_log = []; approved_indices_log = []; current_take_assign_num = start_take
        for index in range(num_selected):
             if index in confirmed_indices: approved_indices_log.append(index); assigned_take_numbers_log.append(current_take_assign_num); current_take_assign_num += 1
        self.log_verification_data(approved_indices_log, assigned_take_numbers_log)
        errors = []; success_count = 0; self.status_message.set(f"Processing {num_approved} approved file(s)..."); self.process_button.config(state='disabled'); self.master.update_idletasks()
        take_counter = 0
        for index, source_video_path in enumerate(selected_files):
            if index in confirmed_indices:
                assigned_take_number = start_take + take_counter
                if not os.path.isfile(source_video_path): errors.append(f"File not found: {os.path.basename(source_video_path)} (Index {index})"); continue
                try:
                    _, file_extension = os.path.splitext(source_video_path)
                    new_filename = f"{interpreter_id}_{assigned_take_number}{file_extension}"
                    final_destination_path = os.path.join(target_folder_path, new_filename)
                    print(f"Moving Approved (Index {index}): {source_video_path} To: {final_destination_path} (Take {assigned_take_number})")
                    shutil.move(source_video_path, final_destination_path)
                    success_count += 1; take_counter += 1
                except Exception as e: error_msg = f"Failed to move '{os.path.basename(source_video_path)}' (Index {index}): {e}"; print(f"Error: {error_msg}"); errors.append(error_msg)
        final_message = f"Processed {success_count}/{num_approved} approved files."
        if errors: final_message += f"\nEncountered {len(errors)} error(s):\n" + "\n".join(f"- {e}" for e in errors); messagebox.showwarning("Processing Complete with Errors", final_message)
        else: messagebox.showinfo("Processing Complete", final_message)

        # --- Reset for next word/session ---
        self.selected_file_paths_tuple = ()
        self.selected_files_info.set("No files selected")
        self.clear_analysis_results()
        # Keep BaseDir, ID, ColA selected
        self.selected_col_b.set("") # Clear Col B selection
        self.col_b_combobox.set("") # Clear combobox display
        # Repopulate Col B list based on current Col A
        self.populate_col_b()
        # Enable Col B if options exist, disable File Select
        self.col_b_combobox.config(state='readonly' if self.col_b_combobox['values'] else 'disabled')
        self.select_files_button.config(state='disabled')
        self.status_message.set("Select name of the next word.")
        self.check_button_state() # Update button states

    # --- NEW: Handle window closing ---
    def on_closing(self):
        """Cancel all pending 'after' jobs when the window is closed."""
        print("Closing application, cancelling pending jobs...")
        for after_id in self.preview_after_ids:
            if after_id:
                try:
                    self.master.after_cancel(after_id)
                except tk.TclError:
                    # Job might have already executed or been cancelled
                    pass
        self.master.destroy()


# --- Run the Application ---
if __name__ == "__main__":
    # (Startup dependency check remains the same)
    try:
        import cv2
        from PIL import Image, ImageTk
        from skimage.metrics import structural_similarity
        import numpy
    except ImportError as e:
         print(f"CRITICAL ERROR: Missing dependency {e.name}. Please install requirements.")
         print("pip install opencv-python Pillow scikit-image numpy")
         root_check = tk.Tk(); root_check.withdraw()
         messagebox.showerror("Missing Dependencies", f"Error: {e.name} not found.\nPlease install requirements:\npip install opencv-python Pillow scikit-image numpy")
         root_check.destroy(); sys.exit(1)

    root = tk.Tk()
    app = VideoPlacerApp(root)
    root.mainloop()
