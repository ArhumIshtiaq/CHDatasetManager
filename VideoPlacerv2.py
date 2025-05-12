# c:\Users\XC\Desktop\Projects\ConnectHear\CHDatasetManager\VideoPlacerv2.py
import os
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import re
import csv
import datetime
import threading
import queue
import concurrent.futures # Added for parallel processing
import sys # Added for sys.exit
import logging # Added for the decorator's default level
import functools # Added for functools.wraps in the decorator
# import os # Duplicate import, already imported above

# --- Adjust sys.path to allow running script directly within a package structure ---
# This allows imports like 'from CHDatasetManager.module import ...'
# Get the directory of the current script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the script's directory (this should be the directory containing the CHDatasetManager package)
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# Add the project root to sys.path so that 'CHDatasetManager' can be found as a package
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# --- Local Project Imports ---
# Assuming 'CHDatasetManager' is the name of the package directory
from CHDatasetManager.logger_config import logger
from CHDatasetManager.constants import *
from CHDatasetManager.video_processing_operations import VideoProcessor
from CHDatasetManager.file_system_operations import (
    get_directory_structure, # determine_next_take_number, # No longer directly used by VideoPlacerApp
    calculate_take_assignment_details, execute_video_processing_fs, # New imports
    move_and_rename_video, log_verification_to_csv
)

# --- Try importing required libraries and provide guidance if missing ---
try:
    logger.debug("Attempting to import dependencies: cv2, PIL, skimage, numpy, tkinterdnd2")
    import cv2
    from PIL import Image, ImageTk
    from skimage.metrics import structural_similarity as ssim
    import numpy as np
    logger.debug("Core dependencies (cv2, PIL, skimage, numpy) imported successfully.")
    try:
        from tkinterdnd2 import TkinterDnD, DND_FILES
        tkinter_dnd_available = True
        logger.debug("TkinterDnD2 imported successfully for drag-and-drop.")
    except ImportError:
        tkinter_dnd_available = False
        logger.warning("TkinterDnD2 library not found. Drag and drop functionality will be unavailable. Install with 'pip install tkinterdnd2'.")

except ImportError as e:
    logger.critical(f"Missing required library: {e.name}. Please install dependencies.", exc_info=True)
    messagebox.showerror(
        "Missing Dependencies",
        f"Error: Required library not found: {e.name}\n\n"
        "Please install required libraries using pip:\n"
        "pip install opencv-python Pillow scikit-image numpy\n"
        "(Optional for drag-and-drop: pip install tkinterdnd2)"
    )
    # Exit if dependencies are missing, otherwise the app will crash later
    sys.exit(1)
except Exception as e:
     logger.critical(f"An unexpected error occurred during dependency import: {e}", exc_info=True)
     messagebox.showerror("Critical Error", f"An unexpected error occurred during startup:\n{e}")
     sys.exit(1)

# --- Logging Decorator ---
def log_method_call(_func=None, *, level=logging.DEBUG, log_args=True, log_return=True, log_exception=True):
    """
    A decorator to log method calls, arguments, return values, and exceptions.
    Uses the global 'logger' instance.

    Can be used as @log_method_call or with arguments, e.g.,
    @log_method_call(level=logging.INFO, log_args=False).

    Args:
        _func: The function to decorate (implicitly passed if used as @log_method_call).
        level: The logging level for entry and exit messages.
        log_args: Boolean, whether to log function arguments.
        log_return: Boolean, whether to log the return value.
        log_exception: Boolean, whether to log exceptions (always at ERROR level).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs): # Assumes it's decorating an instance method
            func_qualname = func.__qualname__ # Gets Class.method_name

            entry_message = f"CALLING: {func_qualname}"
            if log_args:
                args_repr = [repr(a) for a in args]
                kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
                signature = ", ".join(args_repr + kwargs_repr)
                entry_message += f"({signature})"
            
            logger.log(level, entry_message)

            try:
                result = func(self, *args, **kwargs)
                if log_return:
                    result_repr = repr(result)
                    # Truncate long return values to keep logs manageable
                    if len(result_repr) > 250:
                        result_repr = result_repr[:247] + "..."
                    logger.log(level, f"RETURNED: {func_qualname} -> {result_repr}")
                return result
            except Exception as e:
                if log_exception:
                    logger.error(f"EXCEPTION in {func_qualname}: {e!r}", exc_info=True)
                raise # Re-raise the exception
        return wrapper

    if _func is None:
        # Called with arguments: @log_method_call(level=logging.INFO)
        return decorator
    else:
        # Called without arguments: @log_method_call
        return decorator(_func)

# --- Global Constants for the GUI ---
VIDEO_TYPES_FILTER = [("Video Files", "*.mp4 *.avi *.mov *.wmv *.mkv *.flv"), ("All Files", "*.*")]

# --- Modular UI Frame Classes ---

class SetupWidgetsFrame(ttk.Frame):
    """Frame for Base Directory and Interpreter ID selection widgets."""
    def __init__(self, master, app_instance, logger_instance):
        super().__init__(master)
        self.app = app_instance  # To access tk.StringVars and callbacks
        self.logger = logger_instance
        self._create_widgets()

    @log_method_call
    def _create_widgets(self):
        self.logger.debug(f"{self.__class__.__name__}: Creating widgets.")
        ttk.Label(self, text="Base Directory:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=(10, 5))
        self.base_dir_entry = ttk.Entry(self, textvariable=self.app.base_directory, width=45, state='readonly')
        self.base_dir_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=(10, 5))
        self.base_dir_button = ttk.Button(self, text="Browse...", command=self.app.select_base_dir)
        self.base_dir_button.grid(row=0, column=3, padx=10, pady=(10, 5))

        ttk.Label(self, text="Interpreter ID:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        interpreter_ids = [f"{i:03d}" for i in INTERPRETER_ID_RANGE]
        self.interpreter_id_combobox = ttk.Combobox(self, textvariable=self.app.selected_interpreter_id, values=interpreter_ids, width=40, state='disabled')
        self.interpreter_id_combobox.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        self.interpreter_id_combobox.bind("<<ComboboxSelected>>", self.app.on_id_select)

        self.columnconfigure(1, weight=1) # Allow entry/combobox to expand

    @log_method_call
    def update_widget_states(self, initial_setup_done, base_dir_set):
        self.base_dir_button.config(state='normal' if not base_dir_set and not initial_setup_done else 'disabled')
        self.interpreter_id_combobox.config(state='readonly' if base_dir_set and not initial_setup_done else 'disabled')
        if initial_setup_done:
            self.base_dir_button.config(state='disabled')
            self.interpreter_id_combobox.config(state='disabled')

class CategoryWordTreeFrame(ttk.Frame):
    """Frame for the Category/Word TreeView."""
    def __init__(self, master, app_instance, logger_instance):
        super().__init__(master)
        self.app = app_instance
        self.logger = logger_instance
        self._create_widgets()

    @log_method_call
    def _create_widgets(self):
        self.logger.debug(f"{self.__class__.__name__}: Creating widgets.")
        ttk.Label(self, text="Category / Word:").grid(row=0, column=0, sticky=tk.NW, padx=10, pady=(10,5))
        self.category_word_tree = ttk.Treeview(self, selectmode="browse", height=7, show="tree headings")
        self.category_word_tree.heading("#0", text="Select a Word")
        self.category_word_tree.column("#0", width=250)
        self.tree_scroll = ttk.Scrollbar(self, orient="vertical", command=self.category_word_tree.yview)
        self.category_word_tree.configure(yscrollcommand=self.tree_scroll.set)
        self.category_word_tree.grid(row=0, column=1, sticky="nsew", padx=(5,0), pady=5)
        self.tree_scroll.grid(row=0, column=2, sticky="nsw", padx=(0,10), pady=5)
        # Binding is done by the main app as it controls the handler logic

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

    @log_method_call
    def populate(self, dir_structure):
        for i in self.category_word_tree.get_children():
            self.category_word_tree.delete(i)
        if not dir_structure:
            self.category_word_tree.insert("", "end", text="No categories found or error.", open=False)
            return False
        for category_name, words in dir_structure.items():
            category_id = self.category_word_tree.insert("", "end", text=category_name, open=False, tags=('category',))
            if words:
                for word_name in words:
                    self.category_word_tree.insert(category_id, "end", text=word_name, tags=('word',))
            else:
                self.category_word_tree.insert(category_id, "end", text=" (No words)", tags=('empty_category_info',))
        return True

    def bind_select_event(self, handler): self.category_word_tree.bind("<<TreeviewSelect>>", handler)
    def unbind_select_event(self): self.category_word_tree.unbind("<<TreeviewSelect>>")
    def focus_on_tree(self): self.category_word_tree.focus_set()
    def get_focused_item_id(self): return self.category_word_tree.focus()
    def get_item_details(self, item_id): return self.category_word_tree.item(item_id)
    def get_parent_id(self, item_id): return self.category_word_tree.parent(item_id)
    def clear_selection(self): self.category_word_tree.selection_set(())

class FileSelectionWidgetsFrame(ttk.Frame):
    """Frame for Selected Files display and selection button."""
    def __init__(self, master, app_instance, logger_instance):
        super().__init__(master)
        self.app = app_instance
        self.logger = logger_instance
        self._create_widgets()

    @log_method_call
    def _create_widgets(self):
        self.logger.debug(f"{self.__class__.__name__}: Creating widgets.")
        ttk.Label(self, text="Selected Files:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.files_info_entry = ttk.Entry(self, textvariable=self.app.selected_files_info, width=40, state='readonly')
        self.files_info_entry.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        self.select_files_button = ttk.Button(self, text="Select Files...", command=self.app.select_video_files, state='disabled')
        self.select_files_button.grid(row=0, column=3, padx=10, pady=5)
        self.columnconfigure(1, weight=1)

    @log_method_call
    def update_button_state(self, word_selected):
        self.select_files_button.config(state='normal' if word_selected else 'disabled')

class VerificationPanelFrame(ttk.Frame):
    """Frame for video previews, scores, and approval checkboxes."""
    def __init__(self, master, app_instance, logger_instance):
        super().__init__(master, borderwidth=1, relief="sunken")
        self.app = app_instance
        self.logger = logger_instance
        self.master_tk = master.winfo_toplevel() # To use 'after'

        # UI Elements
        self.preview_labels = []
        self.score_display_vars = []
        self.score_labels = []
        self.per_video_confirmed_vars = []
        self.confirm_checkboxes = []

        # Animation State
        self.preview_photo_images = [[] for _ in range(MAX_VIDEOS)]
        self.preview_animation_index = [0] * MAX_VIDEOS
        self.preview_after_ids = [None] * MAX_VIDEOS

        self._preview_click_callback = None
        self._checkbox_change_callback = None

        self._create_widgets()

    @log_method_call
    def _create_widgets(self):
        self.logger.debug(f"{self.__class__.__name__}: Creating verification slot widgets.")
        for i in range(MAX_VIDEOS):
            item_frame = ttk.Frame(self)
            item_frame.grid(row=0, column=i, padx=5, pady=5, sticky="n")
            self.columnconfigure(i, weight=1)

            preview_label = ttk.Label(item_frame)
            preview_label.pack(pady=(0, 2))
            preview_label.bind("<Button-1>", lambda event, idx=i: self._handle_preview_click(idx))
            self.preview_labels.append(preview_label)

            score_var = tk.StringVar(value="Score: -")
            self.score_display_vars.append(score_var)
            score_label = ttk.Label(item_frame, textvariable=score_var, style="Score.TLabel")
            score_label.pack(pady=(0, 2))
            self.score_labels.append(score_label)

            confirmed_var = tk.BooleanVar(value=False)
            self.per_video_confirmed_vars.append(confirmed_var)
            confirm_cb = ttk.Checkbutton(item_frame, text="Approve", variable=confirmed_var,
                                         onvalue=True, offvalue=False, style="Confirm.TCheckbutton",
                                         state='disabled', command=lambda idx=i: self._handle_checkbox_change(idx))
            confirm_cb.pack(pady=(0, 5))
            self.confirm_checkboxes.append(confirm_cb)

    def _handle_preview_click(self, index):
        if self._preview_click_callback:
            self._preview_click_callback(index)

    def _handle_checkbox_change(self, index):
        if self._checkbox_change_callback:
            self._checkbox_change_callback(index)

    def bind_preview_click_callback(self, callback): self._preview_click_callback = callback
    def bind_checkbox_change_callback(self, callback): self._checkbox_change_callback = callback

    @log_method_call
    def clear_all_slots(self):
        self.logger.debug(f"{self.__class__.__name__}: Clearing all verification slots.")
        self._stop_all_animations()
        self.preview_photo_images = [[] for _ in range(MAX_VIDEOS)]
        self.preview_animation_index = [0] * MAX_VIDEOS
        for i in range(MAX_VIDEOS):
            self.preview_labels[i].config(image='')
            self.score_display_vars[i].set("Score: -")
            self.per_video_confirmed_vars[i].set(False)
            self.confirm_checkboxes[i].config(state='disabled')

    @log_method_call
    def update_slot_preview(self, index, photo_image_list):
        if not (0 <= index < MAX_VIDEOS): return
        self.preview_photo_images[index] = [img for img in photo_image_list if img is not None] # Store only valid PhotoImages
        if self.preview_photo_images[index]:
            self._start_preview_animation(index)
            self.enable_disable_slot_controls(index, True)
        else:
            self.preview_labels[index].config(image='') # Clear if no valid images
            self.enable_disable_slot_controls(index, False) # Disable if no preview

    @log_method_call
    def set_slot_score(self, index, score_value, score_text_prefix="Score: "):
        if not (0 <= index < MAX_VIDEOS): return
        self.score_display_vars[index].set(f"{score_text_prefix}{score_value:.3f}" if score_value is not None else f"{score_text_prefix}N/A")

    @log_method_call
    def set_slot_approved(self, index, is_approved, is_enabled=True):
        if not (0 <= index < MAX_VIDEOS): return
        self.per_video_confirmed_vars[index].set(is_approved)
        self.confirm_checkboxes[index].config(state='normal' if is_enabled else 'disabled')

    def get_slot_approved_state(self, index):
        return self.per_video_confirmed_vars[index].get() if 0 <= index < MAX_VIDEOS else False

    @log_method_call
    def enable_disable_slot_controls(self, index, is_enabled):
        if not (0 <= index < MAX_VIDEOS): return
        self.confirm_checkboxes[index].config(state='normal' if is_enabled else 'disabled')
        # Preview label click is always active, but callback can check enabled state if needed

    @log_method_call
    def _start_preview_animation(self, video_idx):
        if self.preview_after_ids[video_idx] is not None:
            try: self.master_tk.after_cancel(self.preview_after_ids[video_idx])
            except tk.TclError: pass
            self.preview_after_ids[video_idx] = None
        self.preview_animation_index[video_idx] = 0
        if self.preview_photo_images[video_idx]: # Check if there are images to animate
            self._update_preview_animation(video_idx)
        else: # Ensure label is clear if no images
            if video_idx < len(self.preview_labels):
                 self.preview_labels[video_idx].config(image='')

    @log_method_call
    def _update_preview_animation(self, video_idx):
        valid_photo_list = self.preview_photo_images[video_idx]
        if not valid_photo_list:
            if video_idx < len(self.preview_labels): self.preview_labels[video_idx].config(image='')
            self.preview_after_ids[video_idx] = None
            return

        num_valid_frames = len(valid_photo_list)
        current_frame_idx = self.preview_animation_index[video_idx] % num_valid_frames
        photo_to_display = valid_photo_list[current_frame_idx]

        try:
            if video_idx < len(self.preview_labels): self.preview_labels[video_idx].config(image=photo_to_display)
        except tk.TclError as e:
            self.logger.error(f"Error updating preview label {video_idx}: {e}", exc_info=True)
            self.preview_after_ids[video_idx] = None
            return

        self.preview_animation_index[video_idx] = (self.preview_animation_index[video_idx] + 1)
        self.preview_after_ids[video_idx] = self.master_tk.after(
            PREVIEW_ANIMATION_DELAY, self._update_preview_animation, video_idx
        )

    @log_method_call
    def _stop_all_animations(self):
        self.logger.debug(f"{self.__class__.__name__}: Stopping all animations.")
        for i in range(MAX_VIDEOS):
            if self.preview_after_ids[i] is not None:
                try: self.master_tk.after_cancel(self.preview_after_ids[i])
                except tk.TclError: pass
                self.preview_after_ids[i] = None

    @log_method_call
    def on_app_closing(self):
        self._stop_all_animations()

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
        self.selected_category = tk.StringVar() # Replaces selected_col_a
        self.selected_word = tk.StringVar()     # Replaces selected_col_b
        self.selected_file_paths_tuple = ()
        self.selected_files_info = tk.StringVar(value="No files selected")
        self.status_message = tk.StringVar(value="Step 1: Select Base Directory")
        self.take_assignment_display = tk.StringVar(value="Takes: -")
        # Analysis/verification variables
        self.per_video_similarity_scores = [None] * MAX_VIDEOS # Actual score data
        self.video_approval_states = [False] * MAX_VIDEOS      # Actual approval data
        self.initial_confirmation_state = [False] * MAX_VIDEOS # Stores state *after* analysis pre-marking
        self.analysis_queue = queue.Queue()
        self.is_analysis_running = False
        self.globally_processed_video_paths = set() # Tracks all successfully processed original video paths in this session
        logger.debug("Tkinter variables and state variables initialized.")

        # --- Initialize Helper Classes ---
        self.video_processor = VideoProcessor(self.analysis_queue.put)


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

        row_basedir = 0; row_id = 1; row_treeview = 2 # Category/Word Tree
        row_fileselect = 3; row_verification_area = 4; row_analysis_info = 5
        row_processbtn = 6; row_status = 7
        logger.debug(f"Defined grid row assignments: basedir={row_basedir}, id={row_id}, treeview={row_treeview}, fileselect={row_fileselect}, verification={row_verification_area}, analysis_info={row_analysis_info}, process_btn={row_processbtn}, status={row_status}")

        # --- Widgets ---
        # Instantiate new frame classes
        self.setup_widgets_frame = SetupWidgetsFrame(main_frame, self, logger)
        self.setup_widgets_frame.grid(row=row_basedir, column=0, columnspan=4, sticky=tk.EW, pady=(10,0))
        # Direct access to child widgets for now, can be refactored later if needed
        self.base_dir_button = self.setup_widgets_frame.base_dir_button
        self.interpreter_id_combobox = self.setup_widgets_frame.interpreter_id_combobox
        logger.debug("SetupWidgetsFrame placed.")

        # --- Category/Word TreeView ---
        self.category_word_tree_frame = CategoryWordTreeFrame(main_frame, self, logger)
        self.category_word_tree_frame.grid(row=row_treeview, column=0, columnspan=4, sticky="nsew", pady=5)
        # Direct access to child tree widget
        self.category_word_tree = self.category_word_tree_frame.category_word_tree
        self.category_word_tree_frame.bind_select_event(self.on_tree_item_select)
        logger.debug("CategoryWordTreeFrame placed.")

        # --- File Selection Widgets ---
        self.file_selection_widgets_frame = FileSelectionWidgetsFrame(main_frame, self, logger)
        self.file_selection_widgets_frame.grid(row=row_fileselect, column=0, columnspan=4, sticky=tk.EW, pady=5)
        # Direct access to child button
        self.select_files_button = self.file_selection_widgets_frame.select_files_button
        logger.debug("FileSelectionWidgetsFrame placed.")


        # --- Verification Area (Animated Preview, Scores, Checkboxes) ---
        ttk.Label(main_frame, text="Review & Approve Takes:").grid(row=row_verification_area, column=0, sticky="nw", padx=10, pady=(15, 0))
        self.verification_panel_frame = VerificationPanelFrame(main_frame, self, logger) # This is the new container
        self.verification_panel_frame.bind_preview_click_callback(self._handle_preview_click_app_level)
        self.verification_panel_frame.bind_checkbox_change_callback(self._handle_checkbox_change_app_level)
        self.verification_panel_frame.grid(row=row_verification_area, column=1, columnspan=3, sticky="nsew", padx=5, pady=(10,5))
        logger.debug("Verification area frame placed.")

        # Widgets inside VerificationPanelFrame are now created by the frame itself.

        # Take Assignment Display
        ttk.Label(main_frame, textvariable=self.take_assignment_display, style="TakeAssign.TLabel").grid(row=row_analysis_info, column=1, columnspan=2, sticky="w", padx=5, pady=(5, 0)) # Adjusted columnspan and padx
        logger.debug("Take assignment display label placed.")

        # Process Button
        self.process_button = ttk.Button(main_frame, text="Place Approved Files", command=self.process_selected_videos, state='disabled')
        self.process_button.grid(row=row_processbtn, column=1, sticky=tk.EW, pady=20, padx=5) # Adjusted columnspan and sticky
        logger.debug("Process button placed.")

        # Status Label
        ttk.Label(main_frame, textvariable=self.status_message, style="Status.TLabel").grid(row=row_status, column=0, columnspan=4, sticky="ew", padx=5, pady=(10, 15)) # Adjusted padx
        logger.debug("Status label placed.")

        # --- Configure Grid (within main_frame) ---
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=0) # Column for scrollbar / some buttons
        main_frame.columnconfigure(3, weight=0) # Column for some buttons
        main_frame.rowconfigure(row_verification_area, weight=1)
        logger.debug("Main frame grid configured for expansion.")

        # --- Start Queue Checker & Bind Closing ---
        self.master.after(100, self.check_analysis_queue) # Start queue checker
        logger.debug("Started check_analysis_queue loop via master.after.")
        # Bind closing event
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        logger.debug("Bound on_closing method to window close event.")

        # --- Drag and Drop Setup ---
        self.dnd_initialized_successfully = False
        if tkinter_dnd_available: # Check if TkinterDnD was imported successfully at the script level
            try:
                if isinstance(self.master, TkinterDnD.Tk): # Check if master is DND enabled
                    self.verification_panel_frame.drop_target_register(DND_FILES) # Use the new panel
                    self.verification_panel_frame.dnd_bind('<<Drop>>', self.handle_drop_event)
                    logger.info("Drag and drop enabled for the verification frame.")
                    self.dnd_initialized_successfully = True
                else:
                    logger.warning("Drag and drop setup skipped: master window is not a TkinterDnD.Tk instance (should be if tkinter_dnd_available is True).")
            except NameError: # Handles if DND_FILES or TkinterDnD is not defined due to failed import within this scope
                logger.warning("TkinterDnD2 components not available for DND setup in __init__.")
            except Exception as e:
                logger.error(f"Error setting up drag and drop in __init__: {e}", exc_info=True)
        else:
            logger.info("Drag and drop disabled as TkinterDnD2 library is not available.")
        logger.info("VideoPlacerApp GUI initialization complete.")


    # --- Helper Functions ---
    @log_method_call
    def clear_analysis_results(self):
        """Clears previous analysis results from GUI and stops animations."""
        self.verification_panel_frame.clear_all_slots()
        
        self.per_video_similarity_scores = [None] * MAX_VIDEOS
        logger.debug("Similarity scores reset.")

        self.video_approval_states = [False] * MAX_VIDEOS
        logger.debug("Video approval states reset.")

        self.take_assignment_display.set("Takes: -")
        logger.debug("Take assignment display reset.")
        self.initial_confirmation_state = [False] * MAX_VIDEOS # Reset initial state tracking
        logger.debug("Initial confirmation state tracking reset.")

        logger.debug("Finished clearing analysis results.")

    @log_method_call
    def select_base_dir(self):
        if self.initial_setup_done:
            logger.debug(f"{self.__class__.__name__}.select_base_dir: Initial setup already done, returning.") # Specific log still useful
            return

        directory = filedialog.askdirectory(title="Select Base Directory (Set Once)")
        if directory:
            self.base_directory.set(directory)
            self.status_message.set("Step 2: Select your Interpreter ID")
            logger.info(f"Base directory selected: {directory}")
            # Enable next step and disable this one
            self.setup_widgets_frame.interpreter_id_combobox.config(state='readonly') # Use frame's widget
            self.setup_widgets_frame.interpreter_id_combobox.focus()
            self.setup_widgets_frame.base_dir_button.config(state='disabled')
            self.category_word_tree_frame.unbind_select_event()
            logger.debug("Treeview unbind <<TreeviewSelect>> in select_base_dir")
            self.select_files_button.config(state='disabled')
            self.process_button.config(state='disabled') # Process button should also be disabled
            logger.debug("Enabled Interpreter ID combobox, disabled Base Dir button.")
        else:
            logger.info("Base directory selection cancelled.")
            self.status_message.set("Base directory selection cancelled. Please select Base Directory.")
            logger.debug("Base directory selection cancelled, status message updated.")

    @log_method_call
    def on_id_select(self, event=None):
        selected_id = self.selected_interpreter_id.get()
        if not selected_id:
            return
        if self.initial_setup_done:
            return

        logger.info(f"Interpreter ID selected: {selected_id}. Initial setup now complete.")
        self.initial_setup_done = True
        self.setup_widgets_frame.interpreter_id_combobox.config(state='disabled') # Use frame's widget
        self.status_message.set("Step 3: Select a Word from the tree below.")
        logger.debug("Calling populate_category_word_tree.")
        self.populate_category_word_tree()
        # populate_category_word_tree itself will set state to 'normal' if successful
        # or 'disabled' if not.
        self.category_word_tree_frame.focus_on_tree() # Set focus to the tree

        # Clear subsequent selections and states
        self.selected_category.set("")
        self.selected_word.set("")
        self.category_word_tree_frame.clear_selection() # Clear tree selection
        logger.debug("Category and Word selections (StringVar) cleared. Tree selection cleared.")
        
        self.selected_file_paths_tuple = ()
        self.selected_files_info.set("No files selected")
        logger.debug("Word selection and file selection cleared.")

        self.clear_analysis_results()
        logger.debug("Analysis results cleared.")

        # Status message is handled by populate_category_word_tree or on_tree_item_select

        # Select Files button should be disabled until Word is selected
        self.select_files_button.config(state='disabled')
        logger.debug("File select button disabled.")

        self.check_button_state() # Re-evaluate process button state
        logger.debug("check_button_state called after Category selection.")
        logger.debug("on_id_select finished processing.")

    @log_method_call
    def populate_category_word_tree(self):
        """Populates the TreeView with categories and words from the base directory."""
        base_dir = self.base_directory.get()
        if not base_dir or not os.path.isdir(base_dir):
            logger.warning(f"Base directory '{base_dir}' not set or not a directory. Tree not populated.")
            self.category_word_tree_frame.populate(None) # Pass None to indicate error/empty
            self.category_word_tree_frame.unbind_select_event()
            logger.debug("Treeview unbind <<TreeviewSelect>> in populate_category_word_tree (base_dir invalid)")
            self.status_message.set("Error: Base directory invalid. Cannot load categories.")
            return

        dir_structure = get_directory_structure(base_dir)

        if not self.category_word_tree_frame.populate(dir_structure):
            logger.info(f"No category subdirectories found or error scanning {base_dir}.")
            self.category_word_tree_frame.unbind_select_event()
            self.status_message.set("No categories found or error. Check base directory.")
            return
            
        self.category_word_tree_frame.bind_select_event(self.on_tree_item_select)
        logger.debug("Treeview bind <<TreeviewSelect>> in populate_category_word_tree (success)")
        self.status_message.set("Select a Category, then a Word from the tree.")
        logger.info(f"Populated category/word tree with {len(dir_structure)} categories from {base_dir}.")

    @log_method_call
    def on_tree_item_select(self, event=None):
        """Handles selection changes in the category/word TreeView."""
        selected_item_id = self.category_word_tree_frame.get_focused_item_id()

        if not selected_item_id: # No item selected (e.g., selection cleared)
            self.selected_category.set("")
            self.selected_word.set("")
            logger.debug("Tree selection cleared. Category and Word reset.")
        else:
            item = self.category_word_tree_frame.get_item_details(selected_item_id)
            item_text = item['text']
            item_tags = item['tags']

            if 'word' in item_tags:
                parent_id = self.category_word_tree_frame.get_parent_id(selected_item_id)
                self.selected_category.set(self.category_word_tree_frame.get_item_details(parent_id)["text"])
                self.selected_word.set(item_text)
                logger.info(f"Word selected: '{item_text}' under Category: '{self.selected_category.get()}'")
            elif 'category' in item_tags:
                self.selected_category.set(item_text)
                self.selected_word.set("") # Clear word if only category is selected
                logger.info(f"Category selected: '{item_text}'. Word cleared.")
            else: # Should not happen if tags are set correctly
                self.selected_category.set("")
                self.selected_word.set("")
                logger.warning(f"Selected tree item '{item_text}' has no 'category' or 'word' tag.")

        # Common actions for any selection change
        self.selected_file_paths_tuple = ()
        self.selected_files_info.set("No files selected")
        self.clear_analysis_results() # Clears scores, previews, and disables checkboxes

        if self.selected_word.get():
            self.status_message.set(f"Step 4: Select video files for Word '{self.selected_word.get()}'")
            self.calculate_and_display_take_assignment() # Update take info if a word is selected
        elif self.selected_category.get():
            self.status_message.set(f"Select a Word under Category '{self.selected_category.get()}'")
            self.take_assignment_display.set("Takes: -") # Clear take info if only category
        else:
            self.status_message.set("Select a Category, then a Word.")
            self.take_assignment_display.set("Takes: -")

        self.check_button_state()
        logger.debug("on_tree_item_select finished processing.")

    @log_method_call
    def _process_filepaths_for_analysis(self, filepaths):
        """Common logic to handle a list of filepaths for analysis."""
        if not filepaths:
            logger.debug("_process_filepaths_for_analysis called with no filepaths.")
            return

        num_selected = len(filepaths)
        logger.info(f"Processing {num_selected} file(s) for analysis via _process_filepaths_for_analysis.")

        if num_selected > MAX_VIDEOS:
            messagebox.showwarning("Too Many Files", f"Please select/drop a maximum of {MAX_VIDEOS} files. {num_selected} were provided.")
            logger.warning(f"User provided {num_selected} files, exceeding the maximum of {MAX_VIDEOS}.")
            return

        # Validate file extensions (basic check)
        # And check against globally processed files
        skipped_due_to_already_processed_basenames = []
        actually_validated_for_analysis = []

        for fp in filepaths:
            if fp in self.globally_processed_video_paths:
                skipped_due_to_already_processed_basenames.append(os.path.basename(fp))
                logger.info(f"Skipping globally already processed video: {fp}")
                continue # Skip this file for further validation and analysis

            if os.path.splitext(fp)[1].lower() in VALID_VIDEO_EXTENSIONS:
                actually_validated_for_analysis.append(fp) # Passed extension check and not globally processed
            else:
                logger.warning(f"Skipping non-video file (based on extension): {fp}")
        
        if skipped_due_to_already_processed_basenames:
            messagebox.showinfo("Files Skipped",
                                "The following files were skipped as they have already been processed in this session:\n- " +
                                "\n- ".join(skipped_due_to_already_processed_basenames))

        if not actually_validated_for_analysis:
            if not skipped_due_to_already_processed_basenames: # Only show if no files were skipped for being processed
                 messagebox.showwarning("No Valid New Video Files", "No new, valid video files were found to process among the provided files.")
            logger.warning("No valid new video files after all checks from provided list.")
            # Clear selections if all provided files were invalid
            self.selected_file_paths_tuple = ()
            self.selected_files_info.set("No files selected")
            self.clear_analysis_results() # Clean up UI
            self.status_message.set("No new, valid video files found to process.")
            self.check_button_state()
            return

        filepaths_to_process = tuple(actually_validated_for_analysis)
        num_to_process = len(filepaths_to_process)

        self.clear_analysis_results() # Clears previous results and animations
        logger.debug("Cleared previous analysis results and animations before new analysis.")

        self.selected_file_paths_tuple = filepaths_to_process
        self.selected_files_info.set(f"{num_to_process} file(s) selected")
        self.status_message.set("Analyzing videos...")
        logger.info(f"Selected {num_to_process} video files for analysis: {'; '.join(map(os.path.basename, filepaths_to_process))}")

        self.check_button_state() # Update button states (e.g., disable process button)
        self.is_analysis_running = True
        logger.info("Analysis started. Setting is_analysis_running to True.")

        # Use the VideoProcessor instance to run analysis
        analysis_thread = threading.Thread(
            target=self.video_processor.run_analysis_in_thread,
            args=(filepaths_to_process,),
            daemon=True)
        analysis_thread.start()
        logger.debug(f"Analysis thread started: {analysis_thread.name}")

    @log_method_call
    def select_video_files(self):
        """Opens dialog to select multiple video files and starts analysis thread."""
        # logger.debug("select_video_files called.") # Covered by decorator
        if not self.selected_word.get():
            messagebox.showwarning("Selection Missing", "Please select Word first.")
            logger.warning("Attempted to select files before selecting Word.")
            return

        if self.is_analysis_running:
            messagebox.showwarning("Busy", "Analysis is already in progress.")
            logger.warning("Attempted to select files while analysis is already running.")
            return

        logger.debug(f"Opening file dialog to select up to {MAX_VIDEOS} video files.")
        filepaths_from_dialog = filedialog.askopenfilenames(title=f"Select up to {MAX_VIDEOS} Video Files", filetypes=VIDEO_TYPES_FILTER)

        if filepaths_from_dialog:
            self._process_filepaths_for_analysis(filepaths_from_dialog)
        else:
            self.selected_file_paths_tuple = ()
            self.selected_files_info.set("No files selected")
            self.clear_analysis_results() # Ensure display is clean on cancel
            self.status_message.set("Video file selection cancelled.")
            logger.info("Video file selection cancelled by user.")
            self.check_button_state() # Update button states after cancellation

    @log_method_call
    def handle_drop_event(self, event):
        """Handles files dropped onto the designated drop target."""
        # logger.debug(f"handle_drop_event called. Event data raw: '{event.data}'") # Covered by decorator, args logged

        if not self.selected_word.get():
            messagebox.showwarning("Selection Missing", "Please select Category and Word first before dropping files.")
            logger.warning("Attempted to drop files before selecting Word.")
            return

        if self.is_analysis_running:
            messagebox.showwarning("Busy", "Analysis is already in progress. Please wait before dropping more files.")
            logger.warning("Attempted to drop files while analysis is already running.")
            return

        try:
            # Parse the file paths from the event data. TkinterDnD2 uses Tcl-list format.
            filepaths = self.master.tk.splitlist(event.data)
            if not filepaths:
                logger.warning("Drop event occurred but no filepaths were parsed from event.data.")
                return
            
            logger.info(f"Files dropped and parsed: {filepaths}")
            self._process_filepaths_for_analysis(filepaths)

        except Exception as e:
            logger.error(f"Error handling dropped files: {e}", exc_info=True)
            messagebox.showerror("Drop Error", f"An error occurred while processing dropped files:\n{e}")
            self.selected_file_paths_tuple = () # Reset state
            self.selected_files_info.set("No files selected")
            self.clear_analysis_results()
            self.status_message.set("Error processing dropped files.")
            self.check_button_state()

    @log_method_call
    def _update_scores_display_from_analysis(self, scores, num_selected):
        # logger.debug("Updating score display labels.") # Covered by decorator
        for i in range(MAX_VIDEOS):
            current_score = scores[i] if i < num_selected else None
            score_text = "Score: -"
            if current_score is not None: score_text = f"Score: {current_score:.3f}"
            elif i < num_selected : score_text = "Score: N/A" # Only show N/A if it was a selected file
            self.verification_panel_frame.set_slot_score(i, current_score, score_text_prefix="Score: " if current_score is not None else "") # Pass prefix only if score exists
            logger.debug(f"Set score for slot {i}: - (index out of bounds for selected files)")
        
        valid_scores = [s for s in scores[:num_selected] if s is not None]
        max_score = 0.0
        if valid_scores: max_score = max(valid_scores)
        return valid_scores, max_score

    @log_method_call
    def _create_previews_and_init_checkboxes(self, list_of_preview_pil_list, num_selected):
        # logger.debug("Creating PhotoImage objects and configuring checkboxes.") # Covered by decorator
        # self.preview_photo_images = [[] for _ in range(MAX_VIDEOS)] # Now managed by VerificationPanelFrame
        num_videos_with_valid_previews = 0

        for video_idx in range(MAX_VIDEOS):
            photo_images_for_video = []
            preview_success = False 

            if video_idx < num_selected and video_idx < len(list_of_preview_pil_list):
                pil_images_for_video = list_of_preview_pil_list[video_idx]
                logger.debug(f"Processing preview images for slot {video_idx}. Found {len(pil_images_for_video)} PIL images.")

                for img_list_idx, pil_image in enumerate(pil_images_for_video):
                    if pil_image is not None:
                        try:
                            photo_img = ImageTk.PhotoImage(pil_image)
                            photo_images_for_video.append(photo_img)
                            preview_success = True
                            logger.debug(f"Successfully created PhotoImage for frame {img_list_idx +1} in slot {video_idx}.")
                        except Exception as e:
                            logger.error(f"Error creating PhotoImage for image at list index {img_list_idx} in slot {video_idx}: {e}", exc_info=True)
                            photo_images_for_video.append(None) 
                    else:
                        photo_images_for_video.append(None)
                        logger.warning(f"PIL image was None for image at list index {img_list_idx} in slot {video_idx}.")
            
            # Pass the list of PhotoImage objects to the frame
            self.verification_panel_frame.update_slot_preview(video_idx, photo_images_for_video)

            if preview_success:
                num_videos_with_valid_previews += 1
                # Animation is started by update_slot_preview if photo_images_for_video is not empty
                self.verification_panel_frame.enable_disable_slot_controls(video_idx, True)
                logger.debug(f"Slot {video_idx} has valid previews. Controls enabled.")
            else:
                self.verification_panel_frame.enable_disable_slot_controls(video_idx, False)
                # If it was previously approved, unapprove it as preview failed
                if self.video_approval_states[video_idx]:
                    self.video_approval_states[video_idx] = False
                    self.verification_panel_frame.set_slot_approved(video_idx, False, is_enabled=False)
                logger.debug(f"No valid previews for slot {video_idx}. Animation not started, label cleared.")
        
        return num_videos_with_valid_previews

    @log_method_call
    def _apply_pre_marking_logic(self, scores, num_selected, filepaths, valid_scores_list, max_score_val):
        # logger.debug("Starting pre-marking process.") # Covered by decorator
        if num_selected <= 1 or len(valid_scores_list) < 2:
            logger.info("Pre-marking skipped: not enough videos or valid scores.")
            self.initial_confirmation_state = list(self.video_approval_states[:num_selected]) # Copy current (likely all False)
            logger.debug(f"Stored initial confirmation state (pre-marking skipped): {self.initial_confirmation_state[:num_selected]}")
            return

        std_dev = np.std(valid_scores_list)
        score_threshold_sd = max_score_val - PRE_MARKING_SD_FACTOR * std_dev
        logger.info(f"Pre-marking logic: {num_selected} videos, {len(valid_scores_list)} valid. MaxScore={max_score_val:.3f}, SD={std_dev:.3f}, SD_Threshold={score_threshold_sd:.3f}, Fixed_Threshold={PRE_MARKING_SCORE_THRESHOLD}")

        for i in range(num_selected):
            current_score = scores[i] # Use the full scores list passed
            # Check if the slot's controls are enabled (meaning it has a valid preview)
            is_enabled = self.verification_panel_frame.confirm_checkboxes[i].cget('state') == 'normal'

            if is_enabled and current_score is not None and \
               (current_score >= score_threshold_sd or current_score >= PRE_MARKING_SCORE_THRESHOLD):
                logger.info(f"Pre-marking video index {i} ('{os.path.basename(filepaths[i])}') - Score {current_score:.3f}. Checkbox enabled: {is_enabled}.")
                self.video_approval_states[i] = True
                self.verification_panel_frame.set_slot_approved(i, True, is_enabled=True)
            elif is_enabled:
                logger.debug(f"Video index {i} ('{os.path.basename(filepaths[i])}') not pre-marked. Score {current_score}. Checkbox enabled: {is_enabled}.")
            else:
                logger.debug(f"Video index {i} ('{os.path.basename(filepaths[i])}') not pre-marked (checkbox disabled). Score: {current_score}.")
        
        self.initial_confirmation_state = list(self.video_approval_states[:num_selected]) # Store the state after pre-marking
        logger.debug(f"Stored initial confirmation state after pre-marking: {self.initial_confirmation_state[:num_selected]}")
        
    @log_method_call
    def _report_analysis_issues(self, errors):
        if errors:
            messagebox.showwarning("Analysis Issues", "Encountered issues during analysis:\n- " + "\n- ".join(errors))
            logger.warning(f"Analysis completed with {len(errors)} reported issues.")

    @log_method_call
    def _finalize_analysis_ui_updates(self, num_videos_with_valid_previews):
        if num_videos_with_valid_previews > 0:
            logger.debug("Valid previews found. Calculating take assignment.")
            self.calculate_and_display_take_assignment()
        else:
            self.take_assignment_display.set("Takes: Error")
            self.status_message.set("Analysis failed for all videos. Cannot assign takes.")
            logger.error("Analysis failed for all selected videos. No valid previews/results to assign takes.")
        
        self.check_button_state()
        logger.debug("check_button_state called after analysis results processed.")

    @log_method_call
    def _handle_analysis_completion(self, result):
        # logger.info("Processing 'analysis_complete' message.") # Covered by decorator
        self.is_analysis_running = False
        logger.debug("is_analysis_running set to False.")

        list_of_preview_pil_list = result.get("previews", [])
        self.per_video_similarity_scores = result.get("scores", [None] * MAX_VIDEOS)
        errors = result.get("errors", [])
        filepaths = result.get("filepaths", [])
        num_selected = len(filepaths)

        logger.debug(f"Received scores: {self.per_video_similarity_scores}")
        logger.debug(f"Received errors: {errors}")
        logger.debug(f"Received previews list structure: {len(list_of_preview_pil_list)} lists of previews.")

        valid_scores_list, max_score_val = self._update_scores_display_from_analysis(self.per_video_similarity_scores, num_selected)
        
        num_valid_previews = self._create_previews_and_init_checkboxes(list_of_preview_pil_list, num_selected)
        
        self._apply_pre_marking_logic(self.per_video_similarity_scores, num_selected, filepaths, valid_scores_list, max_score_val)
        
        self._report_analysis_issues(errors)
        self._finalize_analysis_ui_updates(num_valid_previews)

    @log_method_call
    def check_analysis_queue(self):
        """Checks the queue for results, displays scores, starts animations, pre-marks."""
        try:
            result = self.analysis_queue.get_nowait() # Args will be logged by decorator if it's a complex object
            # logger.info("Received message from analysis queue.") # Covered by decorator if result is complex enough

            if result.get('type') == 'analysis_complete':
                self._handle_analysis_completion(result)
            # else: Handle other message types if needed

        except queue.Empty:
            pass # No item in the queue, just continue
        except Exception as e:
            logger.error(f"Unexpected error in check_analysis_queue: {e}", exc_info=True)
            # Potentially show a user error messagebox if this is a critical failure in the main loop

        finally:
             # Schedule the next check regardless of whether an item was processed or an error occurred
             self.master.after(100, self.check_analysis_queue)

    @log_method_call
    def calculate_and_display_take_assignment(self):
        """Calculates the available take range based on existing files and number selected."""
        base_dir = self.base_directory.get() 
        category = self.selected_category.get() 
        word = self.selected_word.get()         
        interpreter_id = self.selected_interpreter_id.get()
        num_selected = len(self.selected_file_paths_tuple)

        if not all([base_dir, category, word, interpreter_id]): 
            self.take_assignment_display.set("Takes: Error")
            self.status_message.set("Error: Prerequisite selection missing for take calculation.")
            logger.error(f"Missing prerequisites for take calculation: BaseDir='{base_dir}', Category='{category}', Word='{word}', InterpreterID='{interpreter_id}'.")
            return

        target_folder_path = os.path.join(base_dir, category, word, interpreter_id)
        logger.debug(f"Checking existing takes in target folder: {target_folder_path}")

        # Call the new helper function from file_system_operations
        assignment_info = calculate_take_assignment_details(
            target_folder_path,
            interpreter_id,
            num_selected # Pass current number of selected files
        )

        self.take_assignment_display.set(assignment_info['message_short'])
        logger.info(f"Take assignment display set to: {assignment_info['message_short']}")

        # Update status message, but be careful not to overwrite "Analyzing..." or other critical errors
        current_status = self.status_message.get()
        if "Analyzing" not in current_status: # Only update if not in middle of analysis
            self.status_message.set(assignment_info['message_long'])
            logger.info(f"Status message updated to: {assignment_info['message_long']}")
        elif assignment_info['error_condition']: # If analysis is running but take calc has an error, still show take error
            self.status_message.set(assignment_info['message_long']) # Prioritize take calculation error message
            logger.info(f"Status message (during analysis) updated due to take calc error: {assignment_info['message_long']}")

        logger.debug("Finished take assignment calculation.")

    @log_method_call
    def check_button_state(self):
        """Enables or disables widgets based on the application state (Set-Once Workflow)."""
        # logger.debug("check_button_state called.") # Covered by decorator

        # Initial Setup Phase
        if not self.initial_setup_done:
            self.setup_widgets_frame.update_widget_states(
                initial_setup_done=False,
                base_dir_set=bool(self.base_directory.get())
            )
            # self.category_word_tree.config(state='disabled') # Tree is managed by unbinding select
            self.category_word_tree_frame.unbind_select_event()
            self.file_selection_widgets_frame.update_button_state(word_selected=False)
            self.process_button.config(state='disabled')
            logger.debug("Widget states updated for initial setup phase.")
            return

        # Post-Initial Setup Phase
        logger.debug("Post-initial setup phase.")
        self.setup_widgets_frame.update_widget_states(initial_setup_done=True, base_dir_set=True)

        if self.base_directory.get(): # Should always be true if initial_setup_done
            self.category_word_tree_frame.bind_select_event(self.on_tree_item_select)
            logger.debug("Treeview bind <<TreeviewSelect>> in check_button_state (enabled)")
        else:
            self.category_word_tree_frame.unbind_select_event()
            logger.debug("Treeview unbind <<TreeviewSelect>> in check_button_state (disabled)")
        logger.debug(f"Category/Word TreeView event binding updated.")

        word_selected = bool(self.selected_word.get())
        self.file_selection_widgets_frame.update_button_state(word_selected=word_selected)
        logger.debug(f"File select button state set to {'normal' if word_selected else 'disabled'}.")

        files_selected = len(self.selected_file_paths_tuple) > 0
        take_info = self.take_assignment_display.get()
        valid_take_assignment = not take_info.startswith("Takes: FULL") and not take_info.startswith("Takes: Error") and take_info != "Takes: -"
        num_selected_actual = len(self.selected_file_paths_tuple)
        at_least_one_confirmed = any(self.video_approval_states[i] for i in range(num_selected_actual))
        can_process = not self.is_analysis_running
        
        enable_process = (files_selected and valid_take_assignment and at_least_one_confirmed and can_process)
        self.process_button.config(state='normal' if enable_process else 'disabled')
        logger.debug(f"Process button state set to {'normal' if enable_process else 'disabled'}. Conditions: FilesSelected={files_selected}, ValidTakes={valid_take_assignment}, Approved={at_least_one_confirmed}, AnalysisRunning={self.is_analysis_running}.")

        logger.debug("Finished check_button_state.")

    @log_method_call
    def _trigger_csv_logging(self, processed_indices, assigned_take_numbers):
        """Logs verification details including pre-marked and final confirmation status to CSV."""
        # logger.info(f"Preparing data for CSV logging to: {VERIFICATION_LOG_FILE}") # Covered by decorator
        num_selected = len(self.selected_file_paths_tuple)

        final_confirmation_states = self.video_approval_states[:num_selected]
        final_confirmation_str = "; ".join(map(str, final_confirmation_states))
        logger.debug(f"Final confirmation states: {final_confirmation_states}")

        initial_confirmation_str = "; ".join(map(str, self.initial_confirmation_state[:num_selected])) 
        logger.debug(f"Initial (pre-marked) confirmation states: {self.initial_confirmation_state[:num_selected]}")

        scores_str = "; ".join([f"{s:.4f}" if s is not None else "N/A" for s in self.per_video_similarity_scores[:num_selected]])
        logger.debug(f"Scores logged: {scores_str}")

        original_filenames_str = "; ".join(os.path.basename(p) for p in self.selected_file_paths_tuple)
        logger.debug(f"Original filenames logged: {original_filenames_str}")

        assigned_takes_str = "; ".join(map(str, assigned_take_numbers))
        logger.debug(f"Assigned take numbers logged: {assigned_takes_str}")

        log_entry = {
            "BaseDirectory": self.base_directory.get(),
            "Category": self.selected_category.get(), 
            "Word": self.selected_word.get(),         
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

        if not log_verification_to_csv(log_entry):
            self.status_message.set("Error logging data (check log file and console).")
            messagebox.showerror("Logging Error", f"Failed to write verification data to log file.\nCheck '{VERIFICATION_LOG_FILE}' and application log for details.")

    @log_method_call
    def _validate_processing_prerequisites(self):
        """Checks if all conditions are met before starting the file processing."""
        if self.is_analysis_running:
            messagebox.showwarning("Busy", "Analysis is in progress. Please wait.")
            logger.warning("Attempted to process files while analysis is running.")
            return False

        if not self.selected_file_paths_tuple:
            messagebox.showerror("Error", "No video files selected.")
            logger.error("Attempted to process with no files selected.")
            return False
        
        num_selected = len(self.selected_file_paths_tuple)
        self.confirmed_indices_for_processing_ = [i for i in range(num_selected) if self.video_approval_states[i]]

        if not self.confirmed_indices_for_processing_:
            messagebox.showerror("No Videos Approved", "Please approve at least one video using the 'Approve' checkbox below it.")
            logger.warning("Process button clicked but no videos were approved.")
            return False

        base_dir = self.base_directory.get()
        category = self.selected_category.get()
        word = self.selected_word.get()
        interpreter_id = self.selected_interpreter_id.get()

        if not all([base_dir, category, word, interpreter_id]):
             messagebox.showerror("Configuration Error", "Base directory, Category, Word, or Interpreter ID is missing.")
             logger.error("Processing attempted with missing configuration details.")
             return False
        
        self.target_folder_path_for_processing_ = os.path.join(base_dir, category, word, interpreter_id)
        self.interpreter_id_for_processing_ = interpreter_id
        return True

    @log_method_call
    def _verify_and_calculate_takes_for_processing(self):
        """Re-verifies existing takes and checks if approved videos fit."""
        num_approved = len(self.confirmed_indices_for_processing_)

        # Use the new helper, passing the number of *approved* videos
        assignment_info = calculate_take_assignment_details(
            self.target_folder_path_for_processing_,
            self.interpreter_id_for_processing_,
            num_approved # Crucially, use the count of videos we intend to process
        )

        if assignment_info['error_condition']:
            # The message_long from calculate_take_assignment_details should be suitable for messagebox
            messagebox.showerror("Take Assignment Error", 
                                 f"Cannot process due to take assignment issues:\n{assignment_info['message_long']}\n\nPlease adjust approved videos or check existing files.")
            logger.warning(f"Processing aborted due to take assignment error: {assignment_info['message_long']}")
            return None
        
        if assignment_info['start_take'] is None: # Should be caught by error_condition, but as a safeguard
            messagebox.showerror("Error", "Failed to determine a valid starting take number.")
            logger.error("start_take was None in assignment_info despite no error_condition.")
            return None
            
        self.start_take_for_processing_ = assignment_info['start_take']
        logger.info(f"Re-verified. Starting take for {num_approved} approved files will be {self.start_take_for_processing_}.")
        return self.start_take_for_processing_

    @log_method_call
    def _execute_file_operations(self) -> tuple[list[str], list[str], int]:
        """Moves and renames the approved video files. Updates globally processed set."""
        # errors = [] # Replaced by errors_list
        # success_count = 0 # Replaced by successfully_moved_original_paths
        num_approved_for_processing = len(self.confirmed_indices_for_processing_)
        
        self.status_message.set(f"Processing {num_approved_for_processing} approved file(s)...")
        self.process_button.config(state='disabled')
        self.master.update_idletasks()
        logger.info(f"Starting file movement for {num_approved_for_processing} approved files.")

        # Call the new file system operation
        # execute_video_processing_fs returns: successfully_processed_source_paths, errors_list
        successfully_moved_original_paths, errors_list = execute_video_processing_fs(
            self.selected_file_paths_tuple,
            self.confirmed_indices_for_processing_,
            self.target_folder_path_for_processing_,
            self.interpreter_id_for_processing_,
            self.start_take_for_processing_
        )
        for moved_path in successfully_moved_original_paths:
            self.globally_processed_video_paths.add(moved_path)
        logger.info(f"Added {len(successfully_moved_original_paths)} paths to globally_processed_video_paths. Total: {len(self.globally_processed_video_paths)}")

        logger.info(f"Finished file movement. Successfully processed {len(successfully_moved_original_paths)} files with {len(errors_list)} errors.")
        return successfully_moved_original_paths, errors_list, num_approved_for_processing

    @log_method_call
    def _finalize_processing_and_reset_ui(self, success_count, errors, num_approved_initially):
        """Shows completion message and resets the UI for the next operation."""
        final_message = f"Processed {success_count}/{num_approved_initially} approved files."
        if errors:
            final_message += f"\n\nEncountered {len(errors)} error(s):\n" + "\n".join(f"- {e}" for e in errors)
            messagebox.showwarning("Processing Complete with Errors", final_message)
            self.status_message.set("Processing finished with errors.")
            logger.warning("Processing completed with errors.")
        else:
            messagebox.showinfo("Processing Complete", final_message)
            self.status_message.set("Processing complete.")
            logger.info("Processing completed successfully.")

        logger.debug("Resetting UI state after processing.")
        self.selected_file_paths_tuple = ()
        self.selected_files_info.set("No files selected")
        self.clear_analysis_results()
        self.selected_word.set("") # This will trigger on_tree_item_select if it was bound, which calls check_button_state
        self.category_word_tree_frame.clear_selection() # Clear tree selection to reset state
        self.take_assignment_display.set("Takes: -")
        self.status_message.set("Select a Word from the tree.")
        logger.info("UI reset for next Word selection.")
        self.check_button_state()

    @log_method_call
    def process_selected_videos(self):
        """Handles moving and renaming ONLY the individually approved video files."""
        # logger.info("Process button clicked. Starting video placement.") # Covered by decorator

        if not self._validate_processing_prerequisites():
            self.check_button_state() 
            return

        start_take = self._verify_and_calculate_takes_for_processing()
        if start_take is None: 
            self.check_button_state()
            return
        
        assigned_take_numbers_for_log = list(range(self.start_take_for_processing_, self.start_take_for_processing_ + len(self.confirmed_indices_for_processing_)))
        self._trigger_csv_logging(
            self.confirmed_indices_for_processing_,
            assigned_take_numbers_for_log
        )

        successfully_moved_paths, errors, num_initially_approved_for_op = self._execute_file_operations()
        
        actual_success_count = len(successfully_moved_paths)
        self._finalize_processing_and_reset_ui(actual_success_count, errors, num_initially_approved_for_op)
        
        del self.confirmed_indices_for_processing_
        del self.target_folder_path_for_processing_
        del self.interpreter_id_for_processing_
        del self.start_take_for_processing_

    @log_method_call
    def _handle_preview_click_app_level(self, slot_index):
        """
        Toggles the approval state for a given video slot if its checkbox is enabled.
        Called when the preview label for a slot is clicked.
        """
        if not (0 <= slot_index < MAX_VIDEOS):
            logger.warning(f"Invalid slot_index {slot_index} in _handle_preview_click_app_level.")
            return

        # Check if the checkbox for this slot is enabled in the VerificationPanelFrame
        current_checkbox_state = self.verification_panel_frame.confirm_checkboxes[slot_index].cget('state')
        logger.debug(f"Checkbox for slot {slot_index} current state: '{current_checkbox_state}'")

        if current_checkbox_state == 'normal': 
            current_app_approval_state = self.video_approval_states[slot_index]
            new_app_approval_state = not current_app_approval_state
            self.video_approval_states[slot_index] = new_app_approval_state
            
            # Tell the frame to update its visual state
            self.verification_panel_frame.set_slot_approved(slot_index, new_app_approval_state, is_enabled=True)
            logger.info(f"Approval for slot {slot_index} toggled by preview click. App state was {current_app_approval_state}, now {new_app_approval_state}. Checkbox was '{current_checkbox_state}'.")
            self.check_button_state() 
        else:
            logger.debug(f"Approval checkbox for slot {slot_index} is disabled. Click on preview ignored.")

    @log_method_call
    def _handle_checkbox_change_app_level(self, slot_index):
        """Called when a checkbox in VerificationPanelFrame changes state."""
        if not (0 <= slot_index < MAX_VIDEOS): return
        self.video_approval_states[slot_index] = self.verification_panel_frame.get_slot_approved_state(slot_index)
        logger.info(f"Checkbox for slot {slot_index} changed. App approval state updated to: {self.video_approval_states[slot_index]}")
        self.check_button_state()

    @log_method_call
    def on_closing(self):
        """Cancel all pending 'after' jobs and destroy the window when the window is closed."""
        self.verification_panel_frame.on_app_closing()
        self.master.destroy()
        logger.info("Window destroyed. Application exiting.")


# --- Run the Application ---
if __name__ == "__main__":
    logger.debug("Entering __main__ block.")

    try:
        if 'cv2' not in sys.modules or 'PIL' not in sys.modules or \
           'skimage' not in sys.modules or 'numpy' not in sys.modules:
             raise ImportError("One or more critical dependencies failed to load despite initial check.")
        logger.debug("Dependencies confirmed available in __main__ namespace.")

    except ImportError as e:
         logger.critical(f"CRITICAL ERROR in __main__: Missing dependency {e}. Application cannot proceed.", exc_info=True)
         print(f"CRITICAL ERROR: Missing dependency {e}. Please install requirements.")
         print("pip install opencv-python Pillow scikit-image numpy tkinterdnd2")
         root_check = tk.Tk(); root_check.withdraw() 
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

    if tkinter_dnd_available:
        try:
            root = TkinterDnD.Tk()
            logger.info("Root window created with TkinterDnD for drag-and-drop support.")
        except Exception as e: 
            logger.error(f"Failed to initialize TkinterDnD.Tk(): {e}. Falling back to standard tk.Tk(). Drag-and-drop will be disabled.", exc_info=True)
            root = tk.Tk()
    else:
        root = tk.Tk()
        logger.info("Root window created with standard tk.Tk(). Drag-and-drop is disabled.")

    app = VideoPlacerApp(root)
    logger.info("Starting Tkinter main loop.")
    root.mainloop()
    logger.info("Tkinter main loop finished.")
