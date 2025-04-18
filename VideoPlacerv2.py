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
import logging # <-- Add logging import
import sys # Added for sys.exit

# --- Setup Logging ---
# Define log file and format BEFORE potential dependency errors
LOG_FILENAME = 'video_placer_app.log'
logging.basicConfig(
    level=logging.DEBUG, # Set initial level to DEBUG to capture everything
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
    filename=LOG_FILENAME,
    filemode='a' # Append to the log file
)
logger = logging.getLogger(__name__)

# Log that logging has been set up
logger.info("Logging configured. Application start.")

# --- Try importing required libraries and provide guidance if missing ---
try:
    logger.debug("Attempting to import dependencies: cv2, PIL, skimage, numpy")
    import cv2
    from PIL import Image, ImageTk
    from skimage.metrics import structural_similarity as ssim
    import numpy as np
    logger.debug("Dependencies imported successfully.")
except ImportError as e:
    logger.critical(f"Missing required library: {e.name}. Please install dependencies.", exc_info=True)
    messagebox.showerror(
        "Missing Dependencies",
        f"Error: Required library not found: {e.name}\n\n"
        "Please install required libraries using pip:\n"
        "pip install opencv-python Pillow scikit-image numpy"
    )
    # Exit if dependencies are missing, otherwise the app will crash later
    sys.exit(1)
except Exception as e:
     logger.critical(f"An unexpected error occurred during dependency import: {e}", exc_info=True)
     messagebox.showerror("Critical Error", f"An unexpected error occurred during startup:\n{e}")
     sys.exit(1)


# --- Constants ---
THUMBNAIL_WIDTH = 160 # Width for the animation display area
THUMBNAIL_HEIGHT = 90  # Height for the animation display area
LOG_FILE = "verification_log.csv" # This is for the separate CSV log
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
        logger.info("Initializing VideoPlacerApp GUI.")
        self.master = master
        master.title("Video Placement Helper (Animated Preview)") # Updated title

        # --- Center the window ---
        master.update_idletasks() # Ensure window dimensions are calculated
        width = 650 # Adjusted width for better layout
        height = 750 # Adjusted height for more content
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        master.geometry(f'{width}x{height}+{x}+{y}')
        master.minsize(width, height) # Set minimum size
        logger.debug(f"Window size set to {width}x{height}, centered at ({x},{y}) on screen {screen_width}x{screen_height}.")


        # --- State Variables ---
        self.initial_setup_done = False
        logger.debug("State variable 'initial_setup_done' initialized to False.")

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
        self.initial_confirmation_state = [False] * MAX_VIDEOS # Stores state *after* analysis pre-marking
        # Store references to the multiple preview PhotoImage objects (list of lists)
        self.preview_photo_images = [[] for _ in range(MAX_VIDEOS)]
        # Store current animation frame index for each slot
        self.preview_animation_index = [0] * MAX_VIDEOS
        # Store IDs returned by master.after for cancellation
        self.preview_after_ids = [None] * MAX_VIDEOS
        self.analysis_queue = queue.Queue()
        self.is_analysis_running = False
        logger.debug("Tkinter variables and state variables initialized.")


        # --- Style ---
        style = ttk.Style()
        style.configure("TLabel", padding=5, font=('Helvetica', 10))
        style.configure("TButton", padding=5, font=('Helvetica', 10))
        style.configure("TCombobox", padding=5, font=('Helvetica', 10))
        style.configure("TEntry", padding=5, font=('Helvetica', 10))
        style.configure("Status.TLabel", font=('Helvetica', 9), foreground="grey")
        style.configure("TakeAssign.TLabel", font=('Helvetica', 10, 'bold'), foreground="blue")
        style.configure("Score.TLabel", font=('Helvetica', 9, 'bold'), foreground="green")
        style.configure("Confirm.TCheckbutton", font=('Helvetica', 9))
        logger.debug("Configured ttk styles.")


        # --- GUI Layout ---
        main_frame = ttk.Frame(master, padding="15 15 15 15") # Increased padding
        main_frame.pack(expand=True, fill=tk.BOTH)
        logger.debug("Main frame created and packed.")

        row_basedir = 0; row_id = 1; row_cola = 2; row_colb = 3
        row_fileselect = 4; row_verification_area = 5; row_analysis_info = 6
        row_processbtn = 7; row_status = 8
        logger.debug(f"Defined grid row assignments: basedir={row_basedir}, id={row_id}, colA={row_cola}, colB={row_colb}, fileselect={row_fileselect}, verification={row_verification_area}, analysis_info={row_analysis_info}, process_btn={row_processbtn}, status={row_status}")

        # --- Widgets ---
        ttk.Label(main_frame, text="Base Directory:").grid(row=row_basedir, column=0, sticky=tk.W, padx=10, pady=(10, 5))
        self.base_dir_entry = ttk.Entry(main_frame, textvariable=self.base_directory, width=45, state='readonly')
        self.base_dir_entry.grid(row=row_basedir, column=1, columnspan=2, padx=5, pady=(10, 5))
        self.base_dir_button = ttk.Button(main_frame, text="Browse...", command=self.select_base_dir)
        self.base_dir_button.grid(row=row_basedir, column=3, padx=10, pady=(10, 5))
        logger.debug("Base directory widgets placed.")

        ttk.Label(main_frame, text="Interpreter ID:").grid(row=row_id, column=0, sticky=tk.W, padx=10, pady=5)
        interpreter_ids = [f"{i:03d}" for i in range(1, 11)] # Example IDs
        self.interpreter_id_combobox = ttk.Combobox(main_frame, textvariable=self.selected_interpreter_id, values=interpreter_ids, width=43, state='disabled')
        self.interpreter_id_combobox.grid(row=row_id, column=1, columnspan=2, padx=5, pady=5)
        self.interpreter_id_combobox.bind("<<ComboboxSelected>>", self.on_id_select)
        logger.debug("Interpreter ID widgets placed.")

        ttk.Label(main_frame, text="Category:").grid(row=row_cola, column=0, sticky=tk.W, padx=10, pady=5)
        self.col_a_combobox = ttk.Combobox(main_frame, textvariable=self.selected_col_a, width=43, state='disabled')
        self.col_a_combobox.grid(row=row_cola, column=1, columnspan=2, padx=5, pady=5)
        self.col_a_combobox.bind("<<ComboboxSelected>>", self.on_col_a_select)
        logger.debug("Category widgets placed.")

        ttk.Label(main_frame, text="Word:").grid(row=row_colb, column=0, sticky=tk.W, padx=10, pady=5)
        self.col_b_combobox = ttk.Combobox(main_frame, textvariable=self.selected_col_b, width=43, state='disabled')
        self.col_b_combobox.grid(row=row_colb, column=1, columnspan=2, padx=5, pady=5)
        self.col_b_combobox.bind("<<ComboboxSelected>>", self.on_col_b_select)
        logger.debug("Word widgets placed.")

        ttk.Label(main_frame, text="Selected Files:").grid(row=row_fileselect, column=0, sticky=tk.W, padx=10, pady=5)
        self.files_info_entry = ttk.Entry(main_frame, textvariable=self.selected_files_info, width=45, state='readonly')
        self.files_info_entry.grid(row=row_fileselect, column=1, columnspan=2, padx=5, pady=5)
        self.select_files_button = ttk.Button(main_frame, text="Select Files...", command=self.select_video_files, state='disabled')
        self.select_files_button.grid(row=row_fileselect, column=3, padx=10, pady=5)
        logger.debug("File selection widgets placed.")


        # --- Verification Area (Animated Preview, Scores, Checkboxes) ---
        ttk.Label(main_frame, text="Review & Approve Takes:").grid(row=row_verification_area, column=0, sticky="nw", padx=10, pady=(15, 0))
        self.verification_frame = ttk.Frame(main_frame, borderwidth=1, relief="sunken")
        self.verification_frame.grid(row=row_verification_area, column=1, columnspan=3, sticky="nsew", padx=10, pady=(10,5))
        logger.debug("Verification area frame placed.")

        # Use single label per slot for animation
        self.preview_labels = []
        self.score_labels = []
        self.confirm_checkboxes = []

        for i in range(MAX_VIDEOS):
            item_frame = ttk.Frame(self.verification_frame)
            item_frame.grid(row=0, column=i, padx=5, pady=5, sticky="n")
            self.verification_frame.columnconfigure(i, weight=1) # Ensure columns expand equally

            # Single Label for Animated Preview
            preview_label = ttk.Label(item_frame) # Size determined by image
            preview_label.pack(pady=(0,2))
            self.preview_labels.append(preview_label)

            # Score Label
            score_label = ttk.Label(item_frame, textvariable=self.score_display_vars[i], style="Score.TLabel")
            score_label.pack(pady=(0,2))
            self.score_labels.append(score_label)

            # Checkbox
            confirm_cb = ttk.Checkbutton(item_frame, text="Approve", variable=self.per_video_confirmed_vars[i], onvalue=True, offvalue=False, command=self.check_button_state, style="Confirm.TCheckbutton", state='disabled')
            confirm_cb.pack(pady=(0,5))
            self.confirm_checkboxes.append(confirm_cb)
        logger.debug(f"Created {MAX_VIDEOS} slots in verification frame (preview labels, score labels, checkboxes).")

        # Take Assignment Display
        ttk.Label(main_frame, textvariable=self.take_assignment_display, style="TakeAssign.TLabel").grid(row=row_analysis_info, column=1, columnspan=3, sticky="w", padx=10, pady=(5, 0))
        logger.debug("Take assignment display label placed.")

        # Process Button
        self.process_button = ttk.Button(main_frame, text="Place Approved Files", command=self.process_selected_videos, state='disabled')
        self.process_button.grid(row=row_processbtn, column=1, columnspan=2, pady=20)
        logger.debug("Process button placed.")

        # Status Label
        ttk.Label(main_frame, textvariable=self.status_message, style="Status.TLabel").grid(row=row_status, column=0, columnspan=4, sticky="ew", padx=10, pady=(10, 15))
        logger.debug("Status label placed.")

        # --- Configure Grid (within main_frame) ---
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(row_verification_area, weight=1)
        logger.debug("Main frame grid configured for expansion.")

        # --- Start Queue Checker & Bind Closing ---
        self.master.after(100, self.check_analysis_queue) # Start queue checker
        logger.debug("Started check_analysis_queue loop via master.after.")
        # Bind closing event
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        logger.debug("Bound on_closing method to window close event.")

        logger.info("VideoPlacerApp GUI initialization complete.")


    # --- Helper Functions ---
    def clear_analysis_results(self):
        """Clears previous analysis results from GUI and stops animations."""
        logger.info("Clearing previous analysis results and stopping animations.")
        # Stop any running animations
        for i in range(MAX_VIDEOS):
            if self.preview_after_ids[i] is not None:
                try:
                    self.master.after_cancel(self.preview_after_ids[i])
                    logger.debug(f"Cancelled animation 'after' job for slot {i}.")
                except tk.TclError:
                    logger.debug(f"Could not cancel animation 'after' job for slot {i} (job may have already run/cancelled).")
                    pass # Job might have already run or been cancelled
                self.preview_after_ids[i] = None

        self.per_video_similarity_scores = [None] * MAX_VIDEOS
        logger.debug("Similarity scores reset.")
        for var in self.score_display_vars:
            var.set("Score: -")
        logger.debug("Score display labels reset.")
        self.take_assignment_display.set("Takes: -")
        logger.debug("Take assignment display reset.")
        self.initial_confirmation_state = [False] * MAX_VIDEOS # Reset initial state tracking
        logger.debug("Initial confirmation state tracking reset.")
        for i in range(MAX_VIDEOS):
            self.per_video_confirmed_vars[i].set(False)
            if i < len(self.confirm_checkboxes):
                 # Keep checkboxes disabled until analysis results are ready
                 self.confirm_checkboxes[i].config(state='disabled')
        logger.debug("Approval checkboxes reset and disabled.")

        # Clear preview images and labels
        self.preview_photo_images = [[] for _ in range(MAX_VIDEOS)] # Reset list of lists
        logger.debug("Preview photo image list reset.")
        for i, label in enumerate(self.preview_labels): # Use the single preview labels now
             try:
                label.config(image='') # Clear the displayed image
                logger.debug(f"Cleared preview image for slot {i}.")
             except Exception as e:
                 logger.warning(f"Failed to clear image for preview label {i}: {e}") # Should not happen
        self.preview_animation_index = [0] * MAX_VIDEOS # Reset animation index
        logger.debug("Preview animation indices reset.")

        logger.debug("Finished clearing analysis results.")

    def select_base_dir(self):
        logger.debug("select_base_dir called.")
        if self.initial_setup_done:
            logger.debug("select_base_dir: Initial setup already done, returning.")
            return

        directory = filedialog.askdirectory(title="Select Base Directory (Set Once)")
        if directory:
            self.base_directory.set(directory)
            self.status_message.set("Step 2: Select your Interpreter ID")
            logger.info(f"Base directory selected: {directory}")
            # Enable next step and disable this one
            self.interpreter_id_combobox.config(state='readonly')
            self.base_dir_button.config(state='disabled')
            # Ensure subsequent steps are disabled
            self.col_a_combobox.config(state='disabled')
            self.col_b_combobox.config(state='disabled')
            self.select_files_button.config(state='disabled')
            self.process_button.config(state='disabled') # Process button should also be disabled
            logger.debug("Enabled Interpreter ID combobox, disabled Base Dir button.")
        else:
            logger.info("Base directory selection cancelled.")
            self.status_message.set("Base directory selection cancelled. Please select Base Directory.")
            logger.debug("Base directory selection cancelled, status message updated.")

    def on_id_select(self, event=None):
        logger.debug(f"on_id_select called. Event: {event}")
        selected_id = self.selected_interpreter_id.get()
        if not selected_id:
            logger.debug("on_id_select: No interpreter ID selected. Returning.")
            return
        if self.initial_setup_done:
            logger.debug("on_id_select: Initial setup already done. Returning.")
            return

        logger.info(f"Interpreter ID selected: {selected_id}. Initial setup now complete.")
        self.initial_setup_done = True
        self.interpreter_id_combobox.config(state='disabled')
        self.status_message.set("Step 3: Select Category")

        logger.debug("Calling populate_col_a.")
        self.populate_col_a()

        # Update state of Category combobox
        col_a_options_exist = bool(self.col_a_combobox['values'])
        self.col_a_combobox.config(state='readonly' if col_a_options_exist else 'disabled')
        logger.debug(f"Category combobox state set to {'readonly' if col_a_options_exist else 'disabled'}.")

        # Clear subsequent selections and states
        self.selected_col_a.set("")
        self.selected_col_b.set("")
        self.col_b_combobox.set("")
        self.col_b_combobox['values'] = [] # Clear old values
        self.col_b_combobox.config(state='disabled')
        logger.debug("Category and Word selections cleared, Word disabled.")

        self.selected_file_paths_tuple = ()
        self.selected_files_info.set("No files selected")
        logger.debug("File selection cleared.")

        self.clear_analysis_results() # Also stops animations and disables checkboxes
        logger.debug("Analysis results cleared.")

        self.check_button_state() # Re-evaluate process button state
        logger.debug("check_button_state called after ID selection.")

    def populate_col_a(self):
        logger.debug("populate_col_a called.")
        base_dir = self.base_directory.get()
        if not base_dir or not os.path.isdir(base_dir):
            self.col_a_combobox['values'] = []
            logger.debug("Base directory not set or not a directory. Category values cleared.")
            return

        try:
            # List directories directly under the base directory
            opts = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
            self.col_a_combobox['values'] = opts
            logger.debug(f"Populated Category combobox with {len(opts)} Category found in {base_dir}.")
            if not opts:
                self.status_message.set("No category folders found.")
                logger.warning(f"No category folders found in {base_dir}")
        except Exception as e:
            logger.error(f"Failed to read base directory '{base_dir}' for Category population: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to read base directory for Category:\n{e}")
            self.col_a_combobox['values'] = []
            self.status_message.set("Error reading base directory.")
            logger.error("Category combobox values cleared due to error.")

    def on_col_a_select(self, event=None):
        logger.debug(f"on_col_a_select called. Event: {event}")
        if not self.initial_setup_done:
            logger.debug("on_col_a_select: Initial setup not done, returning.")
            return

        selected_col_a = self.selected_col_a.get()
        logger.info(f"Category selected: {selected_col_a}")

        # Clear subsequent selections and states
        self.selected_col_b.set("")
        self.col_b_combobox.set("") # Clear combobox display
        self.selected_file_paths_tuple = ()
        self.selected_files_info.set("No files selected")
        logger.debug("Word selection and file selection cleared.")

        self.clear_analysis_results() # Also stops animations and disables checkboxes
        logger.debug("Analysis results cleared.")

        logger.debug("Calling populate_col_b.")
        self.populate_col_b()

        # Update state of Word combobox
        col_b_options_exist = bool(self.col_b_combobox['values'])
        self.col_b_combobox.config(state='readonly' if col_b_options_exist else 'disabled')
        logger.debug(f"Word combobox state set to {'readonly' if col_b_options_exist else 'disabled'}.")

        # Update status message based on whether Word options exist
        if col_b_options_exist:
            self.status_message.set("Step 4: Select Word")
            logger.debug("Status message updated to Step 4.")
        else:
            self.status_message.set("No Word folders found for this Category.")
            logger.warning(f"No Word folders found for category: {selected_col_a}")
        logger.debug("Status message updated.")


        # Select Files button should be disabled until Word is selected
        self.select_files_button.config(state='disabled')
        logger.debug("File select button disabled.")

        self.check_button_state() # Re-evaluate process button state
        logger.debug("check_button_state called after Category selection.")


    def populate_col_b(self):
        logger.debug("populate_col_b called.")
        base_dir = self.base_directory.get()
        col_a = self.selected_col_a.get()

        if not base_dir or not col_a:
            self.col_b_combobox['values'] = []
            logger.debug("Base directory or Category not set. Word values cleared.")
            return

        col_a_path = os.path.join(base_dir, col_a)
        if not os.path.isdir(col_a_path):
            self.col_b_combobox['values'] = []
            logger.warning(f"Category folder '{col_a_path}' not found or not a directory. Word values cleared.")
            return # Should not happen if populate_col_a worked correctly, but good practice

        try:
            # List directories directly under the category directory
            opts = sorted([d for d in os.listdir(col_a_path) if os.path.isdir(os.path.join(col_a_path, d))])
            self.col_b_combobox['values'] = opts
            logger.debug(f"Populated Word combobox with {len(opts)} sessions/items found in {col_a_path}.")
        except Exception as e:
            logger.error(f"Failed to read category folder '{col_a_path}' for Word population: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to read category folder '{col_a}':\n{e}")
            self.col_b_combobox['values'] = []
            logger.error("Word combobox values cleared due to error.")

    def on_col_b_select(self, event=None):
        logger.debug(f"on_col_b_select called. Event: {event}")
        if not self.initial_setup_done:
            logger.debug("on_col_b_select: Initial setup not done, returning.")
            return

        selected_col_b = self.selected_col_b.get()
        logger.info(f"Word selected: {selected_col_b}")

        # Clear previous file selection and analysis results
        self.selected_file_paths_tuple = ()
        self.selected_files_info.set("No files selected")
        logger.debug("File selection cleared.")

        self.clear_analysis_results() # Also stops animations and disables checkboxes
        logger.debug("Analysis results cleared.")

        # Enable File Select button only if Word is selected
        if selected_col_b:
            self.select_files_button.config(state='normal')
            self.status_message.set("Step 5: Select video file(s)")
            logger.debug("File select button enabled. Status updated to Step 5.")
        else:
            self.select_files_button.config(state='disabled')
            self.status_message.set("Select Word")
            logger.debug("File select button disabled. Status updated to Select Word.")

        self.check_button_state() # Re-evaluate process button state
        logger.debug("check_button_state called after Word selection.")


    # --- Video File Selection and Analysis Trigger ---
    def select_video_files(self):
        """Opens dialog to select multiple video files and starts analysis thread."""
        logger.debug("select_video_files called.")
        if not self.selected_col_b.get():
            messagebox.showwarning("Selection Missing", "Please select Word first.")
            logger.warning("Attempted to select files before selecting Word.")
            return

        if self.is_analysis_running:
            messagebox.showwarning("Busy", "Analysis is already in progress.")
            logger.warning("Attempted to select files while analysis is already running.")
            return

        video_types = [("Video Files", "*.mp4 *.avi *.mov *.wmv *.mkv *.flv"), ("All Files", "*.*")]
        logger.debug(f"Opening file dialog to select up to {MAX_VIDEOS} video files.")
        filepaths = filedialog.askopenfilenames(title=f"Select up to {MAX_VIDEOS} Video Files", filetypes=video_types)

        if filepaths:
            num_selected = len(filepaths)
            logger.info(f"User selected {num_selected} file(s).")
            if num_selected > MAX_VIDEOS:
                messagebox.showwarning("Too Many Files", f"Please select a maximum of {MAX_VIDEOS} files.")
                logger.warning(f"User selected {num_selected} files, exceeding the maximum of {MAX_VIDEOS}. Aborting selection.")
                return # Do not proceed if too many files

            # --- Stop previous animations before starting new analysis ---
            self.clear_analysis_results() # This now also cancels animations and disables checkboxes
            logger.debug("Cleared previous analysis results and animations before new selection.")

            self.selected_file_paths_tuple = filepaths
            self.selected_files_info.set(f"{num_selected} file(s) selected")
            self.status_message.set("Analyzing videos...")
            logger.info(f"Selected {num_selected} video files for analysis: {'; '.join(map(os.path.basename, filepaths))}")
            logger.debug(f"File paths tuple updated: {filepaths}")

            self.check_button_state() # Disable process button
            self.is_analysis_running = True
            logger.info("Analysis started. Setting is_analysis_running to True.")

            # Start analysis in a background thread
            logger.info("Starting video analysis thread.")
            analysis_thread = threading.Thread(target=self.analyze_videos_thread, args=(filepaths,), daemon=True)
            analysis_thread.start()
            logger.debug(f"Analysis thread started: {analysis_thread.name}")

        else:
            self.selected_file_paths_tuple = ()
            self.selected_files_info.set("No files selected")
            self.clear_analysis_results() # Ensure display is clean even on cancel
            self.status_message.set("Video file selection cancelled.")
            logger.info("Video file selection cancelled by user.")
            self.check_button_state() # Update button states after cancellation


    # --- Helper Functions for Video Analysis ---
    def _extract_frames(self, video_path):
        """Extracts SSIM keyframes and preview frames from a single video file."""
        video_filename = os.path.basename(video_path)
        logger.debug(f"Thread {threading.current_thread().name}: Attempting to extract frames from: {video_filename}")

        keyframes_for_ssim = [None] * NUM_SSIM_KEYFRAMES
        preview_pil_images = [None] * NUM_PREVIEW_FRAMES
        error_messages = []
        cap = None

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                error_msg = f"Error opening: {video_filename}"
                error_messages.append(error_msg)
                logger.error(f"Thread {threading.current_thread().name}: Failed to open video file: {video_path}")
                return keyframes_for_ssim, preview_pil_images, error_messages

            logger.debug(f"Thread {threading.current_thread().name}: Successfully opened video file: {video_filename}")

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            logger.debug(f"Thread {threading.current_thread().name}: Frame count for {video_filename}: {frame_count}, FPS: {fps:.2f}")

            if frame_count < 2:
                error_msg = f"Not enough frames: {video_filename} ({frame_count} frames)"
                error_messages.append(error_msg)
                logger.warning(f"Thread {threading.current_thread().name}: Video has less than 2 frames: {video_path}")
                return keyframes_for_ssim, preview_pil_images, error_messages

            # Extract SSIM Keyframes
            logger.debug(f"Thread {threading.current_thread().name}: Starting SSIM keyframe extraction for {video_filename}")
            for kf_idx, percentage in enumerate(SSIM_KEYFRAME_PERCENTAGES):
                frame_idx = max(0, min(int(frame_count * percentage), frame_count - 1))
                logger.debug(f"Thread {threading.current_thread().name}: SSIM Keyframe {kf_idx+1}/{NUM_SSIM_KEYFRAMES} ({percentage*100:.1f}%, frame {frame_idx}) from {video_filename}")
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret_key, frame_key = cap.read()

                if ret_key and frame_key is not None:
                    height, width, _ = frame_key.shape
                    # --- Crop to middle horizontal third ---
                    start_col = width // 3
                    end_col = 2 * width // 3
                    cropped_frame_key = frame_key[:, start_col:end_col]
                    logger.debug(f"Thread {threading.current_thread().name}: Cropped SSIM frame {kf_idx+1} from {width}x{height} to {cropped_frame_key.shape[1]}x{cropped_frame_key.shape[0]}.")

                    gray_frame = cv2.cvtColor(cropped_frame_key, cv2.COLOR_BGR2GRAY)
                    resized_gray_frame = cv2.resize(gray_frame, (SSIM_RESIZE_WIDTH, SSIM_RESIZE_HEIGHT), interpolation=cv2.INTER_AREA)
                    keyframes_for_ssim[kf_idx] = resized_gray_frame
                    logger.debug(f"Thread {threading.current_thread().name}: Successfully extracted, cropped, grayed, and resized SSIM keyframe {kf_idx+1} to {resized_gray_frame.shape[1]}x{resized_gray_frame.shape[0]}.")
                else:
                    error_msg = f"Error reading SSIM keyframe {kf_idx+1}: {video_filename} at frame {frame_idx}"
                    error_messages.append(error_msg)
                    logger.error(f"Thread {threading.current_thread().name}: {error_msg}")
            logger.debug(f"Thread {threading.current_thread().name}: Finished SSIM keyframe extraction for {video_filename}.")


            # Extract Preview Frames
            logger.debug(f"Thread {threading.current_thread().name}: Starting preview frame extraction for {video_filename}")
            for pv_idx, percentage in enumerate(PREVIEW_FRAME_PERCENTAGES):
                frame_idx = max(0, min(int(frame_count * percentage), frame_count - 1))
                logger.debug(f"Thread {threading.current_thread().name}: Preview Frame {pv_idx+1}/{NUM_PREVIEW_FRAMES} ({percentage*100:.1f}%, frame {frame_idx}) from {video_filename}")
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret_prev, frame_prev = cap.read()

                if ret_prev and frame_prev is not None:
                    height, width, _ = frame_prev.shape
                    # --- Crop to middle horizontal third ---
                    start_col = width // 3
                    end_col = 2 * width // 3
                    cropped_frame_prev = frame_prev[:, start_col:end_col]
                    logger.debug(f"Thread {threading.current_thread().name}: Cropped preview frame {pv_idx+1} from {width}x{height} to {cropped_frame_prev.shape[1]}x{cropped_frame_prev.shape[0]}.")

                    # --- Calculate aspect ratio preserving resize ---
                    cropped_h, cropped_w, _ = cropped_frame_prev.shape
                    if cropped_h > 0 and cropped_w > 0:
                        aspect_ratio = cropped_w / cropped_h
                        if THUMBNAIL_WIDTH / aspect_ratio <= THUMBNAIL_HEIGHT:
                            final_w = THUMBNAIL_WIDTH
                            final_h = int(final_w / aspect_ratio)
                        else:
                            final_h = THUMBNAIL_HEIGHT
                            final_w = int(final_h * aspect_ratio)
                        # Ensure dimensions are at least 1x1
                        final_w = max(1, final_w)
                        final_h = max(1, final_h)

                        preview_frame_resized = cv2.resize(cropped_frame_prev, (final_w, final_h), interpolation=cv2.INTER_AREA)
                        preview_pil_images[pv_idx] = Image.fromarray(cv2.cvtColor(preview_frame_resized, cv2.COLOR_BGR2RGB))
                        logger.debug(f"Thread {threading.current_thread().name}: Successfully extracted, cropped, and resized preview frame {pv_idx+1} to {final_w}x{final_h}.")
                    else:
                        # Handle cases with invalid dimensions (though unlikely after successful read)
                        error_msg = f"Invalid frame dimensions for preview {pv_idx+1}: {video_filename}"
                        error_messages.append(error_msg)
                        logger.error(f"Thread {threading.current_thread().name}: {error_msg} - Dimensions {cropped_w}x{cropped_h}")
                        preview_pil_images[pv_idx] = None # Add a placeholder
                    # --- End Aspect Ratio Resize ---
                else:
                    error_msg = f"Error reading preview frame {pv_idx+1}: {video_filename} at frame {frame_idx}"
                    error_messages.append(error_msg)
                    logger.error(f"Thread {threading.current_thread().name}: {error_msg}")
            logger.debug(f"Thread {threading.current_thread().name}: Finished preview frame extraction for {video_filename}.")


        except Exception as e:
            error_msg = f"Unexpected error processing {video_filename}: {e}"
            error_messages.append(error_msg)
            logger.error(f"Thread {threading.current_thread().name}: {error_msg}", exc_info=True)
        finally:
            if cap is not None and cap.isOpened():
                cap.release()
                logger.debug(f"Thread {threading.current_thread().name}: Released video capture for: {video_filename}")
            logger.debug(f"Thread {threading.current_thread().name}: Finished frame extraction process for: {video_filename}.")

        return keyframes_for_ssim, preview_pil_images, error_messages


    def _calculate_ssim_scores(self, all_keyframes, filepaths):
        """Calculates average SSIM scores between videos based on extracted keyframes."""
        num_videos = len(all_keyframes)
        logger.debug(f"Calculating SSIM scores for {num_videos} videos.")

        per_video_avg_scores = [None] * num_videos
        error_messages = []

        # Determine which videos have valid keyframes
        valid_video_indices = [i for i, kf_list in enumerate(all_keyframes) if kf_list and all(kf is not None for kf in kf_list)]
        logger.debug(f"Found {len(valid_video_indices)} videos with all required keyframes for SSIM: {valid_video_indices}")

        if len(valid_video_indices) > 1:
            video_score_sums = {idx: 0.0 for idx in valid_video_indices}
            video_pair_counts = {idx: 0 for idx in valid_video_indices}
            pair_scores = {} # Store individual pair scores for potential debugging/future use

            try:
                logger.debug("Starting SSIM pairwise comparison across keyframes for valid videos.")
                # Iterate through all *pairs* of valid videos
                for i in range(len(valid_video_indices)):
                    orig_idx_i = valid_video_indices[i]
                    for j in range(i + 1, len(valid_video_indices)):
                        orig_idx_j = valid_video_indices[j]

                        # Iterate through all keyframes for this pair
                        pair_keyframe_scores = []
                        for kf_idx in range(NUM_SSIM_KEYFRAMES):
                             frame_i = all_keyframes[orig_idx_i][kf_idx]
                             frame_j = all_keyframes[orig_idx_j][kf_idx]

                             # Both frames must be valid (checked by valid_video_indices)
                             try:
                                 score = ssim(frame_i, frame_j, data_range=frame_i.max() - frame_i.min())
                                 pair_keyframe_scores.append(score)
                                 logger.debug(f"  SSIM Keyframe {kf_idx+1}: Pair ({orig_idx_i}, {orig_idx_j}) = {score:.4f}")
                             except ValueError as ve:
                                 # This should ideally not happen if keyframes are valid numpy arrays of same size
                                 error_messages.append(f"SSIM error for pair ({orig_idx_i}, {orig_idx_j}) at keyframe {kf_idx+1}: {ve}")
                                 logger.warning(f"SSIM calculation error for Keyframe {kf_idx+1} (Videos {orig_idx_i} '{os.path.basename(filepaths[orig_idx_i])}' vs {orig_idx_j} '{os.path.basename(filepaths[orig_idx_j])}'): {ve}. Skipping this keyframe for this pair.")
                                 # Do NOT append score if calculation failed for this keyframe

                        if pair_keyframe_scores: # Only average if there were successful keyframe comparisons for this pair
                            avg_pair_score = np.mean(pair_keyframe_scores)
                            pair_scores[(orig_idx_i, orig_idx_j)] = avg_pair_score
                            pair_scores[(orig_idx_j, orig_idx_i)] = avg_pair_score # Store symmetrically
                            logger.debug(f"  Average SSIM for Pair ({orig_idx_i}, {orig_idx_j}) over {len(pair_keyframe_scores)} keyframes = {avg_pair_score:.4f}")

                            # Accumulate average pair score to each video's total
                            video_score_sums[orig_idx_i] += avg_pair_score
                            video_score_sums[orig_idx_j] += avg_pair_score
                            video_pair_counts[orig_idx_i] += 1
                            video_pair_counts[orig_idx_j] += 1
                        else:
                             # Log if a pair had no successful keyframe SSIM comparisons
                             error_messages.append(f"No successful keyframe SSIM comparisons for pair ({orig_idx_i}, {orig_idx_j})")
                             logger.warning(f"No successful keyframe SSIM comparisons for pair ({orig_idx_i} vs {orig_idx_j}).")


                # Calculate the final average score for each video (average of its average pair scores)
                for vid_idx in valid_video_indices:
                    if video_pair_counts[vid_idx] > 0:
                        final_avg_score = video_score_sums[vid_idx] / video_pair_counts[vid_idx]
                        per_video_avg_scores[vid_idx] = final_avg_score
                        logger.info(f"Final Average SSIM for Video {vid_idx} ('{os.path.basename(filepaths[vid_idx])}'): {final_avg_score:.4f} (based on {video_pair_counts[vid_idx]} pairs)")
                    else:
                         # This case should ideally not happen if valid_video_indices has > 1 element
                         # but included for robustness. Means a video had valid frames but no valid pairs.
                         per_video_avg_scores[vid_idx] = None # Indicate no meaningful score could be calculated
                         error_messages.append(f"Could not calculate average SSIM for video {vid_idx} ('{os.path.basename(filepaths[vid_idx])}') as it had no valid pairs.")
                         logger.warning(f"Could not calculate average SSIM for video {vid_idx} ('{os.path.basename(filepaths[vid_idx])}') - no valid pairs.")


            except Exception as e:
                error_msg = f"Unexpected error during multi-frame SSIM calculation: {e}"
                logger.error(f"{error_msg}", exc_info=True)
                error_messages.append(error_msg)
                # Invalidate all scores if a critical error occurs during the main loop
                per_video_avg_scores = [None] * num_videos

        elif len(valid_video_indices) == 1:
            single_valid_idx = valid_video_indices[0]
            logger.info(f"Only one video ({os.path.basename(filepaths[single_valid_idx])}) had all required keyframes for SSIM. Setting its score to 1.0.")
            # If only one video has valid frames, its score relative to others is undefined, assign default 1.0
            per_video_avg_scores[single_valid_idx] = 1.0
        else:
             logger.warning(f"Not enough valid videos ({len(valid_video_indices)}) with all required keyframes for pairwise SSIM comparison. No SSIM scores calculated.")

        logger.debug("Finished SSIM score calculation.")
        return per_video_avg_scores, error_messages


    # --- Background Thread for Analysis (Refactored for Parallel Frame Extraction) ---
    def analyze_videos_thread(self, filepaths):
        """
        Worker function (runs in a separate thread): Coordinates frame extraction (in parallel)
        and SSIM calculation, puts results in queue.
        """
        thread_name = threading.current_thread().name
        logger.info(f"Analysis thread ({thread_name}) started for {len(filepaths)} videos.")

        num_videos = len(filepaths)
        all_preview_pil_images = [[] for _ in range(num_videos)] # List of lists for PIL images
        all_keyframes_for_ssim = [[] for _ in range(num_videos)] # List of lists for numpy arrays
        all_error_messages = []
        analysis_scores = [None] * num_videos # Pre-allocate scores list


        # Step 1: Extract frames for all videos in parallel
        logger.debug(f"Analysis thread ({thread_name}): Starting parallel frame extraction using ThreadPoolExecutor.")
        results = [None] * num_videos # Pre-allocate list for results keyed by original index

        # Using a limited ThreadPoolExecutor might be better for resource management
        # max_workers=min(MAX_VIDEOS, os.cpu_count() or 1) # Or a fixed number like 2 or 4
        # Using default `None` lets executor choose based on CPU count, which is often fine
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks and store futures with original index
            future_to_index = {executor.submit(self._extract_frames, path): i for i, path in enumerate(filepaths)}
            logger.debug(f"Analysis thread ({thread_name}): Submitted {len(future_to_index)} extraction tasks.")

            # Process completed futures as they finish
            for future in concurrent.futures.as_completed(future_to_index):
                original_index = future_to_index[future]
                filename = os.path.basename(filepaths[original_index])
                try:
                    keyframes, previews, errors = future.result()
                    results[original_index] = (keyframes, previews, errors)
                    logger.debug(f"Analysis thread ({thread_name}): Extraction task for '{filename}' (index {original_index}) completed successfully.")
                    if errors:
                         all_error_messages.extend(errors)
                         logger.warning(f"Analysis thread ({thread_name}): Extraction for '{filename}' completed with errors.")
                except Exception as exc:
                    # Handle exceptions during the task execution itself (e.g., unhandled cv2 error)
                    error_msg = f"Unexpected error during frame extraction for '{filename}' (index {original_index}): {exc}"
                    all_error_messages.append(error_msg)
                    logger.error(f"Analysis thread ({thread_name}): {error_msg}", exc_info=True)
                    # Store placeholder or handle error state for this video - results[original_index] is already None

        # Unpack results, maintaining original order
        logger.debug(f"Analysis thread ({thread_name}): Unpacking extraction results.")
        for idx in range(num_videos):
            if results[idx]: # Check if result was successfully stored
                keyframes, previews, errors = results[idx] # errors already added to all_error_messages
                all_keyframes_for_ssim[idx] = keyframes
                all_preview_pil_images[idx] = previews
            else:
                # If results[idx] is None, the task likely failed completely. Add default empty lists.
                all_keyframes_for_ssim[idx] = [None] * NUM_SSIM_KEYFRAMES
                all_preview_pil_images[idx] = [None] * NUM_PREVIEW_FRAMES
                # Error message for this should have been added when catching the exception


        # Step 2: Calculate SSIM scores using the extracted keyframes
        logger.debug(f"Analysis thread ({thread_name}): Starting SSIM score calculation.")
        per_video_avg_scores, ssim_errors = self._calculate_ssim_scores(all_keyframes_for_ssim, filepaths)
        if ssim_errors:
            all_error_messages.extend(ssim_errors)
            logger.warning(f"Analysis thread ({thread_name}): SSIM calculation completed with errors.")
        else:
            logger.debug(f"Analysis thread ({thread_name}): SSIM score calculation completed successfully.")

        analysis_scores = per_video_avg_scores # Use the calculated scores

        # Step 3: Put results into the queue for the main thread
        logger.debug(f"Analysis thread ({thread_name}): Putting results into analysis queue.")
        self.analysis_queue.put({
            'type': 'analysis_complete',
            'scores': analysis_scores,
            'previews': all_preview_pil_images, # Pass the list of lists of PIL images
            'errors': all_error_messages,
            'filepaths': filepaths # Pass filepaths for context in main thread if needed
        })
        logger.info(f"Analysis thread ({thread_name}) finished and results put in queue.")


    # --- MODIFIED Queue Checking (Runs in Main Thread) ---
    def check_analysis_queue(self):
        """Checks the queue for results, displays scores, starts animations, pre-marks."""
        try:
            # logger.debug("Checking analysis queue...") # Too noisy for debug unless needed
            result = self.analysis_queue.get_nowait()
            logger.info("Received message from analysis queue.")

            # Check the type of message from the queue
            if result.get('type') == 'analysis_complete':
                logger.info("Processing 'analysis_complete' message.")
                self.is_analysis_running = False
                logger.debug("is_analysis_running set to False.")

                # Retrieve results using the new keys
                list_of_preview_pil_list = result.get("previews", [])
                self.per_video_similarity_scores = result.get("scores", [None] * MAX_VIDEOS)
                errors = result.get("errors", [])
                filepaths = result.get("filepaths", []) # Get filepaths for context
                num_selected = len(filepaths) # Use the actual number of files selected

                logger.debug(f"Received scores: {self.per_video_similarity_scores}")
                logger.debug(f"Received errors: {errors}")
                logger.debug(f"Received previews list structure: {len(list_of_preview_pil_list)} lists of previews.")


                # --- Update Score Display ---
                logger.debug("Updating score display labels.")
                valid_scores = [s for s in self.per_video_similarity_scores[:num_selected] if s is not None]
                max_score = 0.0
                if valid_scores: max_score = max(valid_scores)
                logger.debug(f"Valid scores for pre-marking: {valid_scores}, Max Score: {max_score:.3f}")


                for i in range(MAX_VIDEOS):
                    if i < num_selected:
                         current_score = self.per_video_similarity_scores[i]
                         if current_score is not None:
                             self.score_display_vars[i].set(f"Score: {current_score:.3f}")
                             logger.debug(f"Set score for slot {i}: {current_score:.3f}")
                         else:
                             self.score_display_vars[i].set("Score: N/A")
                             logger.debug(f"Set score for slot {i}: N/A (score was None)")
                    else:
                        self.score_display_vars[i].set("Score: -")
                        logger.debug(f"Set score for slot {i}: - (index out of bounds for selected files)")


                # --- Create PhotoImage objects for previews and enable/disable checkboxes ---
                logger.debug("Creating PhotoImage objects and configuring checkboxes.")
                self.preview_photo_images = [[] for _ in range(MAX_VIDEOS)] # Reset list of lists
                checkbox_states_after_load = {} # Map index to state ('normal', 'disabled')
                num_videos_with_valid_previews = 0 # Count videos that successfully loaded at least one preview

                for video_idx in range(MAX_VIDEOS):
                    photo_images_for_video = []
                    preview_success = False # Track if at least one preview frame loaded for this video

                    if video_idx < num_selected and video_idx < len(list_of_preview_pil_list):
                        pil_images_for_video = list_of_preview_pil_list[video_idx]
                        logger.debug(f"Processing preview images for slot {video_idx}. Found {len(pil_images_for_video)} potential PIL images.")

                        for frame_idx in range(NUM_PREVIEW_FRAMES):
                            pil_image = pil_images_for_video[frame_idx] if frame_idx < len(pil_images_for_video) else None

                            if pil_image is not None:
                                try:
                                    photo_img = ImageTk.PhotoImage(pil_image)
                                    photo_images_for_video.append(photo_img)
                                    preview_success = True
                                    logger.debug(f"Successfully created PhotoImage for preview frame {frame_idx} in slot {video_idx}.")
                                except Exception as e:
                                    logger.error(f"Error creating PhotoImage for preview frame {frame_idx} in slot {video_idx}: {e}", exc_info=True)
                                    photo_images_for_video.append(None) # Add placeholder on error
                            else:
                                photo_images_for_video.append(None) # Add placeholder if no PIL image was provided
                                logger.debug(f"No PIL image available for preview frame {frame_idx} in slot {video_idx}.")

                    self.preview_photo_images[video_idx] = photo_images_for_video # Store list of PhotoImages

                    # Determine checkbox state based on whether *any* preview frames loaded for this video
                    checkbox_state = 'normal' if preview_success else 'disabled'
                    checkbox_states_after_load[video_idx] = checkbox_state

                    if video_idx < len(self.confirm_checkboxes):
                        self.confirm_checkboxes[video_idx].config(state=checkbox_state)
                        logger.debug(f"Checkbox for slot {video_idx} set to state '{checkbox_state}'.")
                        if checkbox_state == 'disabled':
                            # If disabled, ensure the checkbox is unchecked
                            if self.per_video_confirmed_vars[video_idx].get():
                                self.per_video_confirmed_vars[video_idx].set(False)
                                logger.debug(f"Checkbox for slot {video_idx} was disabled and unchecked.")
                            else:
                                logger.debug(f"Checkbox for slot {video_idx} was disabled.")

                    if preview_success:
                        num_videos_with_valid_previews += 1
                        # --- Start Animation for this slot ---
                        logger.debug(f"Starting preview animation for slot {video_idx}.")
                        self.start_preview_animation(video_idx)
                    else:
                         logger.debug(f"No valid previews loaded for slot {video_idx}. Animation not started.")
                         # Ensure label is empty if no previews loaded
                         if video_idx < len(self.preview_labels):
                            self.preview_labels[video_idx].config(image='')


                # --- Pre-mark Checkbox(es) based on Standard Deviation ---
                logger.debug("Starting pre-marking process.")
                # Reset all checkboxes first (already done in clear_analysis_results, but good to re-iterate intention)
                # for i in range(MAX_VIDEOS): self.per_video_confirmed_vars[i].set(False) # Already reset

                # Pre-marking logic applies only if there's more than one video and at least two have valid scores
                if num_selected > 1 and len(valid_scores) >= 2:
                    std_dev = np.std(valid_scores)
                    score_threshold = max_score - PRE_MARKING_SD_FACTOR * std_dev
                    logger.info(f"Pre-marking logic: {num_selected} videos selected, {len(valid_scores)} valid scores. MaxScore={max_score:.3f}, SD={std_dev:.3f}, Threshold={score_threshold:.3f}")

                    # Iterate through the indices of the *selected* files
                    for i in range(num_selected):
                        current_score = self.per_video_similarity_scores[i]
                        # Check if the checkbox for this slot was successfully enabled
                        is_enabled = checkbox_states_after_load.get(i) == 'normal'

                        # Check score vs threshold AND ensure the checkbox is enabled
                        if is_enabled and current_score is not None and (current_score >= score_threshold or current_score >= PRE_MARKING_SCORE_THRESHOLD):
                            logger.info(f"Pre-marking video index {i} ('{os.path.basename(filepaths[i])}') - Score {current_score:.3f} >= Threshold {score_threshold:.3f} or {PRE_MARKING_SCORE_THRESHOLD}. Checkbox state was '{checkbox_states_after_load.get(i)}'.")
                            self.per_video_confirmed_vars[i].set(True)
                        elif is_enabled:
                             logger.debug(f"Video index {i} ('{os.path.basename(filepaths[i])}') not pre-marked. Score {current_score} (is None={current_score is None}). Threshold check: {current_score >= score_threshold if current_score is not None else 'N/A'} | {current_score >= PRE_MARKING_SCORE_THRESHOLD if current_score is not None else 'N/A'}. Checkbox state was '{checkbox_states_after_load.get(i)}'.")
                        else:
                            logger.debug(f"Video index {i} ('{os.path.basename(filepaths[i])}') not pre-marked because checkbox was disabled (state: '{checkbox_states_after_load.get(i)}'). Score: {current_score}.")


                # --- Store Initial Confirmation State for Logging ---
                # This state reflects the checkboxes AFTER analysis and pre-marking
                self.initial_confirmation_state = [self.per_video_confirmed_vars[i].get() if i < MAX_VIDEOS else False for i in range(num_selected)]
                logger.debug(f"Stored initial confirmation state after pre-marking: {self.initial_confirmation_state[:num_selected]}")


                # Report errors from thread
                if errors:
                    messagebox.showwarning("Analysis Issues", "Encountered issues during analysis:\n- " + "\n- ".join(errors))
                    logger.warning(f"Analysis completed with {len(errors)} reported issues.")

                # Calculate take assignment ONLY if analysis produced results (at least one valid preview)
                if num_videos_with_valid_previews > 0:
                    logger.debug("Valid previews found. Calculating take assignment.")
                    self.calculate_and_display_take_assignment()
                    # Status message updated in calculate_and_display_take_assignment if successful
                else:
                    self.take_assignment_display.set("Takes: Error")
                    self.status_message.set("Analysis failed for all videos. Cannot assign takes.")
                    logger.error("Analysis failed for all selected videos. No valid previews/results to assign takes.")


                self.check_button_state() # Update process button state based on confirmation checkboxes
                logger.debug("check_button_state called after analysis results processed.")
            # else: Handle other message types if needed in the future

        except queue.Empty:
            # logger.debug("Analysis queue empty.") # Too noisy
            pass # No item in the queue, just continue
        except Exception as e:
            logger.error(f"Unexpected error in check_analysis_queue: {e}", exc_info=True)
            # Potentially show a user error messagebox if this is a critical failure in the main loop

        finally:
             # Schedule the next check regardless of whether an item was processed or an error occurred
             self.master.after(100, self.check_analysis_queue)


    # --- Animation Functions ---
    def start_preview_animation(self, video_idx):
        """Starts or restarts the animation loop for a specific video slot."""
        logger.debug(f"Attempting to start/restart animation for slot {video_idx}.")
        # Cancel any previous loop for this slot
        if self.preview_after_ids[video_idx] is not None:
            try:
                self.master.after_cancel(self.preview_after_ids[video_idx])
                logger.debug(f"Cancelled existing animation 'after' job {self.preview_after_ids[video_idx]} for slot {video_idx}.")
            except tk.TclError:
                 logger.debug(f"Existing animation 'after' job for slot {video_idx} was already cancelled or finished.")
                 pass # Ignore if job doesn't exist
            self.preview_after_ids[video_idx] = None
            logger.debug(f"Cleared preview_after_ids[{video_idx}].")


        # Reset index and start the update cycle
        self.preview_animation_index[video_idx] = 0
        logger.debug(f"Reset animation index for slot {video_idx} to 0.")

        # Check if the preview label exists before trying to configure it
        if video_idx < len(self.preview_labels):
            # Check if there are any valid PhotoImages for this slot before starting
            valid_photo_list = [img for img in self.preview_photo_images[video_idx] if img is not None]
            if valid_photo_list:
                 logger.debug(f"Starting animation update cycle for slot {video_idx} with {len(valid_photo_list)} valid frames.")
                 self.update_preview_animation(video_idx)
            else:
                 logger.debug(f"No valid PhotoImages found for slot {video_idx}. Not starting animation.")
                 # Ensure label is clear if no animation starts
                 self.preview_labels[video_idx].config(image='')

        else:
            logger.warning(f"Preview label list size mismatch. Preview label for index {video_idx} not found.")


    def update_preview_animation(self, video_idx):
        """Updates the preview label with the next frame and schedules the next update."""
        # Basic check if core lists exist
        if not all(hasattr(self, attr) for attr in ['preview_photo_images', 'preview_labels', 'preview_animation_index', 'preview_after_ids']):
             logger.error(f"Animation components missing during update_preview_animation for slot {video_idx}. Stopping.")
             return # Cannot proceed

        # Check if video_idx is within expected bounds
        if video_idx >= MAX_VIDEOS or video_idx >= len(self.preview_photo_images) or \
           video_idx >= len(self.preview_labels) or video_idx >= len(self.preview_animation_index) or \
           video_idx >= len(self.preview_after_ids):
             logger.error(f"Invalid video_idx {video_idx} during update_preview_animation. Index out of bounds for component lists. Stopping.")
             self.preview_after_ids[video_idx] = None # Attempt to stop recurrence if possible
             return


        photo_list = self.preview_photo_images[video_idx]
        # Filter out None placeholders
        valid_photo_list = [img for img in photo_list if img is not None]

        if not valid_photo_list: # No valid images loaded for this slot
            logger.debug(f"No valid photo list for slot {video_idx} during update. Stopping animation.")
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
            # logger.debug(f"Updated preview label {video_idx} with frame {current_frame_idx}.") # Too noisy
        except tk.TclError as e:
            logger.error(f"Error updating preview label {video_idx} with image: {e}", exc_info=True)
            self.preview_after_ids[video_idx] = None # Stop loop on error
            return


        # Increment index for the next cycle
        self.preview_animation_index[video_idx] = (self.preview_animation_index[video_idx] + 1)

        # Schedule the next update
        # Store the ID returned by after for later cancellation
        self.preview_after_ids[video_idx] = self.master.after(
            PREVIEW_ANIMATION_DELAY, self.update_preview_animation, video_idx
        )


    # --- Take Calculation ---
    def calculate_and_display_take_assignment(self):
        """Calculates the available take range based on existing files and number selected."""
        logger.debug("calculate_and_display_take_assignment called.")
        base_dir = self.base_directory.get()
        col_a = self.selected_col_a.get()
        col_b = self.selected_col_b.get()
        interpreter_id = self.selected_interpreter_id.get()
        num_selected = len(self.selected_file_paths_tuple)

        if num_selected == 0:
            self.take_assignment_display.set("Takes: -")
            logger.debug("No files selected, take assignment set to '-'.")
            return

        if not all([base_dir, col_a, col_b, interpreter_id]):
            self.take_assignment_display.set("Takes: Error")
            self.status_message.set("Error: Prerequisite selection missing for take calculation.")
            logger.error(f"Missing prerequisites for take calculation: BaseDir='{base_dir}', ColA='{col_a}', ColB='{col_b}', InterpreterID='{interpreter_id}'.")
            return

        target_folder_path = os.path.join(base_dir, col_a, col_b, interpreter_id)
        logger.debug(f"Checking existing takes in target folder: {target_folder_path}")

        highest_take = 0
        start_take = 1

        if os.path.isdir(target_folder_path):
            try:
                # Regex to find files like 001_1.mp4, 001_2.mov etc.
                pattern = re.compile(re.escape(f"{interpreter_id}_") + r"([1-4])\..+$", re.IGNORECASE) # Added IGNORECASE for extensions
                logger.debug(f"Checking files in '{target_folder_path}' for pattern '{pattern.pattern}'.")
                existing_files = os.listdir(target_folder_path)
                logger.debug(f"Found {len(existing_files)} items in target folder.")
                for filename in existing_files:
                     match = pattern.match(filename)
                     if match:
                        try:
                            take_number = int(match.group(1))
                            highest_take = max(highest_take, take_number)
                            logger.debug(f"Found existing file '{filename}' with take number {take_number}. Current highest take: {highest_take}.")
                        except ValueError:
                            logger.warning(f"Found file matching pattern but with non-integer take number group: '{filename}'. Skipping.")
                            continue # Should not happen with ([1-4]) but good safety

                start_take = highest_take + 1
                logger.debug(f"Highest existing take found: {highest_take}. Calculated starting take: {start_take}.")

            except Exception as e:
                self.take_assignment_display.set("Takes: Error")
                self.status_message.set(f"Error checking existing takes in target folder: {e}")
                logger.error(f"Error checking existing takes in '{target_folder_path}': {e}", exc_info=True)
                return
        else:
            start_take = 1
            logger.debug(f"Target folder '{target_folder_path}' does not exist. Starting take number is 1.")


        if start_take > 4:
            self.take_assignment_display.set("Takes: FULL (4/4)")
            self.status_message.set("Error: Maximum 4 takes already exist for this Word/Interpreter.")
            logger.warning(f"Target folder '{target_folder_path}' already contains 4 takes or more. Cannot add new files.")
        else:
            end_take = start_take + num_selected - 1
            available_slots = 4 - start_take + 1

            if end_take > 4:
                self.take_assignment_display.set(f"Takes: Error (Need {num_selected}, Avail: {available_slots})")
                self.status_message.set(f"Error: Too many files selected ({num_selected}) for available takes ({available_slots} starting from {start_take}). Approve carefully.")
                logger.warning(f"Number of selected files ({num_selected}) exceeds available slots ({available_slots}) starting from take {start_take}. End take would be {end_take}.")
            else:
                if num_selected == 1:
                    self.take_assignment_display.set(f"Potential Take: {start_take}")
                    logger.info(f"Calculated potential take: {start_take} for 1 selected file.")
                else:
                    self.take_assignment_display.set(f"Potential Takes: {start_take}-{end_take}")
                    logger.info(f"Calculated potential takes: {start_take}-{end_take} for {num_selected} selected files.")

                # Update status message only if analysis didn't report errors and isn't still running
                current_status = self.status_message.get()
                if "Analyzing" not in current_status and "Error" not in current_status:
                    self.status_message.set(f"Ready for approval. Review videos and approve below.")
                    logger.debug("Status message updated to 'Ready for approval'.")

        logger.debug("Finished take assignment calculation.")


    # --- Button State Check ---
    def check_button_state(self):
        """Enables or disables widgets based on the application state (Set-Once Workflow)."""
        logger.debug("check_button_state called.")

        # Initial Setup Phase
        if not self.initial_setup_done:
            logger.debug("Initial setup phase.")
            # Base directory button enabled only if no base dir is set yet
            self.base_dir_button.config(state='normal' if not self.base_directory.get() else 'disabled')
            # Interpreter ID enabled only after base dir is set
            self.interpreter_id_combobox.config(state='readonly' if self.base_directory.get() else 'disabled')
            # All subsequent steps disabled
            self.col_a_combobox.config(state='disabled')
            self.col_b_combobox.config(state='disabled')
            self.select_files_button.config(state='disabled')
            self.process_button.config(state='disabled')
            # Checkboxes disabled in clear_analysis_results, which is called during initial setup
            # for cb in self.confirm_checkboxes: cb.config(state='disabled') # Redundant if clear_analysis_results is called appropriately
            logger.debug("Widget states updated for initial setup phase.")
            return

        # Post-Initial Setup Phase
        logger.debug("Post-initial setup phase.")
        # Base dir and ID are disabled after initial setup
        self.base_dir_button.config(state='disabled')
        self.interpreter_id_combobox.config(state='disabled')
        logger.debug("Base dir and ID widgets disabled.")

        # Category enabled if options exist
        col_a_options_exist = bool(self.col_a_combobox['values'])
        self.col_a_combobox.config(state='readonly' if col_a_options_exist else 'disabled')
        logger.debug(f"Category combobox state set to {'readonly' if col_a_options_exist else 'disabled'}.")

        # Word enabled if Category is selected AND options exist
        col_a_selected = bool(self.selected_col_a.get())
        col_b_options_exist = bool(self.col_b_combobox['values'])
        self.col_b_combobox.config(state='readonly' if col_a_selected and col_b_options_exist else 'disabled')
        logger.debug(f"Word combobox state set to {'readonly' if col_a_selected and col_b_options_exist else 'disabled'}.")

        # File Select enabled if Word is selected
        col_b_selected = bool(self.selected_col_b.get())
        self.select_files_button.config(state='normal' if col_b_selected else 'disabled')
        logger.debug(f"File select button state set to {'normal' if col_b_selected else 'disabled'}.")

        # Process button requires specific conditions:
        # 1. Files must be selected.
        files_selected = len(self.selected_file_paths_tuple) > 0
        # 2. Take assignment must be valid (not FULL, not Error, not '-').
        take_info = self.take_assignment_display.get()
        valid_take_assignment = not take_info.startswith("Takes: FULL") and not take_info.startswith("Takes: Error") and take_info != "Takes: -"
        # 3. At least one file must be confirmed via checkbox.
        # Only check checkboxes for files that were actually selected
        num_selected_actual = len(self.selected_file_paths_tuple)
        at_least_one_confirmed = any(var.get() for i, var in enumerate(self.per_video_confirmed_vars) if i < num_selected_actual)
        # 4. Analysis must NOT be running.
        can_process = not self.is_analysis_running

        # Enable process button if all conditions met
        enable_process = (files_selected and valid_take_assignment and at_least_one_confirmed and can_process)
        self.process_button.config(state='normal' if enable_process else 'disabled')
        logger.debug(f"Process button state set to {'normal' if enable_process else 'disabled'}. Conditions: FilesSelected={files_selected}, ValidTakes={valid_take_assignment}, Approved={at_least_one_confirmed}, AnalysisRunning={self.is_analysis_running}.")

        # Checkboxes states are handled during analysis completion in check_analysis_queue
        # or cleared in clear_analysis_results. No need to manage their state here based on workflow steps.

        logger.debug("Finished check_button_state.")


    # --- Logging Function (for CSV) ---
    def log_verification_data(self, processed_indices, assigned_take_numbers):
        """Logs verification details including pre-marked and final confirmation status to CSV."""
        logger.info(f"Logging verification data to CSV: {LOG_FILE}")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        num_selected = len(self.selected_file_paths_tuple)

        # Get final confirmation states for the selected files
        final_confirmation_states = [self.per_video_confirmed_vars[i].get() for i in range(num_selected)]
        final_confirmation_str = "; ".join(map(str, final_confirmation_states))
        logger.debug(f"Final confirmation states: {final_confirmation_states}")

        # Get initial confirmation states (after pre-marking) for the selected files
        # self.initial_confirmation_state was stored after check_analysis_queue processed results
        initial_confirmation_str = "; ".join(map(str, self.initial_confirmation_state[:num_selected])) # Ensure we only log for selected files
        logger.debug(f"Initial (pre-marked) confirmation states: {self.initial_confirmation_state[:num_selected]}")


        # Get scores for the selected files
        scores_str = "; ".join([f"{s:.4f}" if s is not None else "N/A" for s in self.per_video_similarity_scores[:num_selected]])
        logger.debug(f"Scores logged: {scores_str}")

        original_filenames_str = "; ".join(os.path.basename(p) for p in self.selected_file_paths_tuple)
        logger.debug(f"Original filenames logged: {original_filenames_str}")

        assigned_takes_str = "; ".join(map(str, assigned_take_numbers))
        logger.debug(f"Assigned take numbers logged: {assigned_takes_str}")

        log_entry = {
            "Timestamp": timestamp,
            "BaseDirectory": self.base_directory.get(),
            "Category": self.selected_col_a.get(),
            "Session": self.selected_col_b.get(),
            "InterpreterID": self.selected_interpreter_id.get(),
            "NumFilesSelected": num_selected,
            "OriginalFileNames": original_filenames_str,
            "PerVideoScores": scores_str,
            "PreMarkedConfirmation": initial_confirmation_str,
            "FinalConfirmation": final_confirmation_str,
            "ProcessedIndices": "; ".join(map(str, processed_indices)),
            "AssignedTakeNumbers": assigned_takes_str
        }
        logger.debug(f"CSV log entry prepared: {log_entry}")

        try:
            file_exists = os.path.isfile(LOG_FILE)
            fieldnames = list(log_entry.keys())

            with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                # Write header only if file doesn't exist or is empty
                if not file_exists or os.path.getsize(LOG_FILE) == 0:
                    writer.writeheader()
                    logger.info(f"CSV log file '{LOG_FILE}' did not exist or was empty. Wrote header.")

                writer.writerow(log_entry)
            logger.info(f"Successfully logged verification data to {LOG_FILE}")

        except Exception as e:
            logger.error(f"Error logging data to CSV file '{LOG_FILE}': {e}", exc_info=True)
            # Update GUI status message for user visibility
            self.status_message.set("Error logging data (check log file and console).")
            messagebox.showerror("Logging Error", f"Failed to write verification data to log file:\n{e}\nCheck '{LOG_FILE}' and application log for details.")


    # --- Processing Logic ---
    def process_selected_videos(self):
        """Handles moving and renaming ONLY the individually approved video files."""
        logger.info("Process button clicked. Starting video placement.")

        if self.is_analysis_running:
            messagebox.showwarning("Busy", "Analysis is in progress. Please wait.");
            logger.warning("Attempted to process files while analysis is running.")
            return

        selected_files = self.selected_file_paths_tuple
        num_selected = len(selected_files)
        if num_selected == 0:
            messagebox.showerror("Error", "No video files selected.");
            logger.error("Attempted to process with no files selected.")
            return

        # Identify which indices among the selected files are approved
        confirmed_indices = [i for i, var in enumerate(self.per_video_confirmed_vars) if i < num_selected and var.get()]
        logger.debug(f"Selected file indices: {list(range(num_selected))}")
        logger.debug(f"Confirmed indices based on checkboxes: {confirmed_indices}")


        if not confirmed_indices:
            messagebox.showerror("No Videos Approved", "Please approve at least one video using the 'Approve' checkbox below it.");
            logger.warning("Process button clicked but no videos were approved.")
            return

        # Re-verify target path and existing takes immediately before processing
        base_dir = self.base_directory.get()
        col_a = self.selected_col_a.get()
        col_b = self.selected_col_b.get()
        interpreter_id = self.selected_interpreter_id.get()

        if not all([base_dir, col_a, col_b, interpreter_id]):
             messagebox.showerror("Configuration Error", "Base directory, Category, Word, or Interpreter ID is missing.");
             logger.error("Processing attempted with missing configuration details.")
             return

        target_folder_path = os.path.join(base_dir, col_a, col_b, interpreter_id)
        logger.debug(f"Processing target folder: {target_folder_path}")

        highest_take = 0
        start_take = 1
        # Re-check existing files to determine the *actual* starting take number just before moving
        if os.path.isdir(target_folder_path):
             try:
                pattern = re.compile(re.escape(f"{interpreter_id}_") + r"([1-4])\..+$", re.IGNORECASE)
                existing_files = os.listdir(target_folder_path)
                logger.debug(f"Re-verifying existing files in {target_folder_path} before processing.")
                for filename in existing_files:
                     match = pattern.match(filename)
                     if match:
                        try:
                            take_number = int(match.group(1))
                            highest_take = max(highest_take, take_number)
                            logger.debug(f"Found existing file '{filename}' (take {take_number}). Current highest: {highest_take}")
                        except ValueError:
                            logger.warning(f"Skipping file '{filename}' in re-verification due to non-integer take number.")
                            continue
                start_take = highest_take + 1
                logger.info(f"Re-verified highest existing take as {highest_take}. Starting take for new files will be {start_take}.")
             except Exception as e:
                 messagebox.showerror("Error", f"Failed to re-verify existing takes before processing: {e}");
                 logger.error(f"Failed to re-verify existing takes in '{target_folder_path}': {e}", exc_info=True)
                 return
        else:
            start_take = 1
            logger.info(f"Target folder '{target_folder_path}' does not exist. Starting take number is 1.")


        num_approved = len(confirmed_indices)
        final_take_needed = start_take + num_approved - 1

        if final_take_needed > 4:
            available_slots = 4 - start_take + 1
            messagebox.showerror("Error", f"Cannot process. Too many videos approved ({num_approved}) for available slots (Max {available_slots}, starting from take {start_take}). Please uncheck some.");
            logger.warning(f"Processing aborted: {num_approved} videos approved, but only {available_slots} slots available starting from {start_take}. Max take needed: {final_take_needed}.")
            self.check_button_state(); # Update button state
            return


        # Create target folder if it doesn't exist
        if not os.path.isdir(target_folder_path):
             logger.info(f"Target folder does not exist. Creating directory: {target_folder_path}")
             try:
                 os.makedirs(target_folder_path)
                 logger.info(f"Successfully created target folder: {target_folder_path}")
             except Exception as e:
                 messagebox.showerror("Error", f"Could not create target folder '{target_folder_path}': {e}");
                 logger.error(f"Failed to create target folder '{target_folder_path}': {e}", exc_info=True)
                 return


        # Prepare data for CSV logging *before* processing files
        assigned_take_numbers_log = []
        approved_indices_log = []
        current_take_assign_num = start_take
        # Iterate through original selected indices to match approved state
        for index in range(num_selected):
             if index in confirmed_indices:
                approved_indices_log.append(index)
                assigned_take_numbers_log.append(current_take_assign_num)
                current_take_assign_num += 1

        # Log the initial and final states *before* file operations
        self.log_verification_data(approved_indices_log, assigned_take_numbers_log)


        # --- Perform File Movement and Renaming ---
        errors = []
        success_count = 0
        self.status_message.set(f"Processing {num_approved} approved file(s)...")
        self.process_button.config(state='disabled') # Disable button during processing
        self.master.update_idletasks() # Update GUI immediately

        take_counter = 0 # Counter for assigning takes to APPROVED files only
        logger.info(f"Starting file movement for {num_approved} approved files.")

        for index, source_video_path in enumerate(selected_files):
            # Only process files whose index is in the confirmed_indices list
            if index in confirmed_indices:
                assigned_take_number = start_take + take_counter # Assign next available take number
                original_filename = os.path.basename(source_video_path)

                if not os.path.isfile(source_video_path):
                    error_msg = f"Source file not found: {original_filename} (Index {index})"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    continue # Skip this file

                try:
                    _, file_extension = os.path.splitext(source_video_path)
                    new_filename = f"{interpreter_id}_{assigned_take_number}{file_extension}"
                    final_destination_path = os.path.join(target_folder_path, new_filename)

                    logger.info(f"Moving approved file (Index {index}): '{original_filename}' -> '{final_destination_path}' (Assigned Take {assigned_take_number})")

                    # Check if destination file already exists (shouldn't happen with correct take assignment, but safety)
                    if os.path.exists(final_destination_path):
                         error_msg = f"Destination file already exists, skipping move for '{original_filename}': {final_destination_path}"
                         errors.append(error_msg)
                         logger.error(error_msg)
                         continue # Skip this file

                    shutil.move(source_video_path, final_destination_path)
                    logger.info(f"Successfully moved and renamed '{original_filename}' to '{new_filename}'.")
                    success_count += 1
                    take_counter += 1 # Increment take counter only for successful moves

                except Exception as e:
                    error_msg = f"Failed to move '{original_filename}' (Index {index}): {e}"
                    logger.error(f"Error during shutil.move for '{original_filename}': {e}", exc_info=True)
                    errors.append(error_msg)

        logger.info(f"Finished file movement. Successfully processed {success_count} files with {len(errors)} errors.")

        # --- Show Completion Message ---
        final_message = f"Processed {success_count}/{num_approved} approved files."
        if errors:
            final_message += f"\n\nEncountered {len(errors)} error(s):\n" + "\n".join(f"- {e}" for e in errors)
            messagebox.showwarning("Processing Complete with Errors", final_message)
            self.status_message.set("Processing finished with errors.")
            logger.warning("Processing completed with errors.")
        else:
            messagebox.showinfo("Processing Complete", final_message)
            self.status_message.set("Processing complete.")
            logger.info("Processing completed successfully.")

        # --- Reset UI for next word/session ---
        logger.debug("Resetting UI state after processing.")
        self.selected_file_paths_tuple = ()
        self.selected_files_info.set("No files selected")
        logger.debug("File selection state reset.")

        # Clear analysis results and stop animations
        self.clear_analysis_results()
        logger.debug("Analysis results cleared after processing.")

        # Keep BaseDir, ID, ColA selected
        # Clear Word selection to prompt user for next item
        self.selected_col_b.set("")
        self.col_b_combobox.set("") # Clear combobox display
        logger.debug("Word selection cleared.")

        # Repopulate Word list based on current Category (needed if user selects a new word in the same category)
        self.populate_col_b()
        logger.debug("Repopulated Word list.")

        # Enable Word if options exist, disable File Select until Word is chosen again
        col_b_options_exist_after_repop = bool(self.col_b_combobox['values'])
        self.col_b_combobox.config(state='readonly' if col_b_options_exist_after_repop else 'disabled')
        self.select_files_button.config(state='disabled')
        logger.debug(f"Word state after repopulation: {'readonly' if col_b_options_exist_after_repop else 'disabled'}, File Select disabled.")


        self.status_message.set("Select name of the next Word.")
        logger.info("UI reset for next Word selection.")

        self.check_button_state() # Ensure button states are correct after reset
        logger.debug("check_button_state called after processing and UI reset.")


    # --- NEW: Handle window closing ---
    def on_closing(self):
        """Cancel all pending 'after' jobs and destroy the window when the window is closed."""
        logger.info("Application closing requested. Stopping animations and destroying window.")
        for i in range(MAX_VIDEOS):
            if self.preview_after_ids[i] is not None:
                try:
                    self.master.after_cancel(self.preview_after_ids[i])
                    logger.debug(f"Cancelled 'after' job {self.preview_after_ids[i]} during closing.")
                except tk.TclError:
                    # Job might have already executed or been cancelled
                    logger.debug(f"Could not cancel 'after' job for slot {i} during closing (job already finished/cancelled).")
                    pass
                self.preview_after_ids[i] = None
        logger.debug("All known animation 'after' jobs cancelled.")
        self.master.destroy()
        logger.info("Window destroyed. Application exiting.")


# --- Run the Application ---
if __name__ == "__main__":
    # Dependency check is now done near the top after logging setup
    # The check here is redundant but kept for clarity based on the original structure
    # We can simplify this block significantly
    logger.debug("Entering __main__ block.")

    # The primary dependency check is handled at the top.
    # If we reached here, dependencies should be installed.
    # We can add a final check here to be extremely safe, but it's unlikely to fail now.
    try:
        # Simple check if the imported modules are available in the namespace
        # This is mostly for defense-in-depth if the above logic changes
        if 'cv2' not in sys.modules or 'PIL' not in sys.modules or \
           'skimage' not in sys.modules or 'numpy' not in sys.modules:
             raise ImportError("One or more critical dependencies failed to load despite initial check.")
        logger.debug("Dependencies confirmed available in __main__ namespace.")

    except ImportError as e:
         # This block should ideally only be hit if something went wrong *after* the first check
         logger.critical(f"CRITICAL ERROR in __main__: Missing dependency {e}. Application cannot proceed.", exc_info=True)
         print(f"CRITICAL ERROR: Missing dependency {e}. Please install requirements.")
         print("pip install opencv-python Pillow scikit-image numpy")
         root_check = tk.Tk(); root_check.withdraw() # Create a temporary root for the messagebox
         messagebox.showerror("Missing Dependencies", f"Error: {e.name} not found.\nPlease install requirements:\npip install opencv-python Pillow scikit-image numpy")
         root_check.destroy()
         sys.exit(1)
    except Exception as e:
         logger.critical(f"CRITICAL ERROR: An unexpected error occurred in __main__ startup: {e}", exc_info=True)
         print(f"CRITICAL ERROR: An unexpected error occurred during startup: {e}")
         root_check = tk.Tk(); root_check.withdraw()
         messagebox.showerror("Critical Error", f"An unexpected critical error occurred during startup:\n{e}")
         root_check.destroy()
         sys.exit(1)


    root = tk.Tk()
    # The App initialization logs start here
    app = VideoPlacerApp(root)
    logger.info("Starting Tkinter main loop.")
    root.mainloop()
    logger.info("Tkinter main loop finished.")
    # on_closing should handle final cleanup before process exit