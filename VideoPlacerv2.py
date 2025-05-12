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
    get_directory_structure, determine_next_take_number,
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
        ttk.Label(main_frame, text="Base Directory:").grid(row=row_basedir, column=0, sticky=tk.W, padx=10, pady=(10, 5))
        self.base_dir_entry = ttk.Entry(main_frame, textvariable=self.base_directory, width=45, state='readonly')
        self.base_dir_entry.grid(row=row_basedir, column=1, columnspan=2, padx=5, pady=(10, 5))
        self.base_dir_button = ttk.Button(main_frame, text="Browse...", command=self.select_base_dir)
        self.base_dir_button.grid(row=row_basedir, column=3, padx=10, pady=(10, 5))
        logger.debug("Base directory widgets placed.")

        ttk.Label(main_frame, text="Interpreter ID:").grid(row=row_id, column=0, sticky=tk.W, padx=10, pady=5)
        interpreter_ids = [f"{i:03d}" for i in INTERPRETER_ID_RANGE]
        self.interpreter_id_combobox = ttk.Combobox(main_frame, textvariable=self.selected_interpreter_id, values=interpreter_ids, width=40, state='disabled') # Adjusted width
        self.interpreter_id_combobox.grid(row=row_id, column=1, sticky=tk.EW, padx=5, pady=5)
        self.interpreter_id_combobox.bind("<<ComboboxSelected>>", self.on_id_select)
        logger.debug("Interpreter ID widgets placed.")

        # --- Category/Word TreeView ---
        ttk.Label(main_frame, text="Category / Word:").grid(row=row_treeview, column=0, sticky=tk.NW, padx=10, pady=(10,5))
        self.category_word_tree = ttk.Treeview(main_frame, selectmode="browse", height=7, show="tree headings") # height is number of rows
        self.category_word_tree.heading("#0", text="Select a Word")
        self.category_word_tree.column("#0", width=250) # Adjust width as needed
        self.tree_scroll = ttk.Scrollbar(main_frame, orient="vertical", command=self.category_word_tree.yview)
        self.category_word_tree.configure(yscrollcommand=self.tree_scroll.set)
        self.category_word_tree.grid(row=row_treeview, column=1, sticky="nsew", padx=(5,0), pady=5)
        self.tree_scroll.grid(row=row_treeview, column=2, sticky="nsw", padx=(0,10), pady=5) # Adjusted padx
        self.category_word_tree.bind("<<TreeviewSelect>>", self.on_tree_item_select)
        logger.debug("Category/Word TreeView placed.")

        ttk.Label(main_frame, text="Selected Files:").grid(row=row_fileselect, column=0, sticky=tk.W, padx=10, pady=5)
        self.files_info_entry = ttk.Entry(main_frame, textvariable=self.selected_files_info, width=40, state='readonly') # Adjusted width
        self.files_info_entry.grid(row=row_fileselect, column=1, sticky=tk.EW, padx=5, pady=5)
        self.select_files_button = ttk.Button(main_frame, text="Select Files...", command=self.select_video_files, state='disabled')
        self.select_files_button.grid(row=row_fileselect, column=3, padx=10, pady=5)
        logger.debug("File selection widgets placed.")


        # --- Verification Area (Animated Preview, Scores, Checkboxes) ---
        ttk.Label(main_frame, text="Review & Approve Takes:").grid(row=row_verification_area, column=0, sticky="nw", padx=10, pady=(15, 0))
        self.verification_frame = ttk.Frame(main_frame, borderwidth=1, relief="sunken")
        self.verification_frame.grid(row=row_verification_area, column=1, columnspan=3, sticky="nsew", padx=5, pady=(10,5)) # Adjusted padx
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
            preview_label.bind("<Button-1>", lambda event, idx=i: self._toggle_approval_for_slot(idx))

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
                    self.verification_frame.drop_target_register(DND_FILES)
                    self.verification_frame.dnd_bind('<<Drop>>', self.handle_drop_event)
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
        # logger.info("Clearing previous analysis results and stopping animations.") # Covered by decorator
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
            self.interpreter_id_combobox.config(state='readonly')
            self.interpreter_id_combobox.focus()
            self.base_dir_button.config(state='disabled')
            self.category_word_tree.unbind("<<TreeviewSelect>>")
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
        self.interpreter_id_combobox.config(state='disabled')
        self.status_message.set("Step 3: Select a Word from the tree below.")
        logger.debug("Calling populate_category_word_tree.")
        self.populate_category_word_tree()
        # populate_category_word_tree itself will set state to 'normal' if successful
        # or 'disabled' if not.
        self.category_word_tree.focus_set() # Set focus to the tree

        # Clear subsequent selections and states
        self.selected_category.set("")
        self.selected_word.set("")
        self.category_word_tree.selection_set(()) # Clear tree selection
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
        # logger.debug("populate_category_word_tree called.") # Covered by decorator
        for i in self.category_word_tree.get_children():
            self.category_word_tree.delete(i)
        logger.debug("Cleared existing items from category_word_tree.")

        base_dir = self.base_directory.get()
        if not base_dir or not os.path.isdir(base_dir):
            logger.warning(f"Base directory '{base_dir}' not set or not a directory. Tree not populated.")
            self.category_word_tree.insert("", "end", text="Base directory not set or invalid.", open=False)
            self.category_word_tree.unbind("<<TreeviewSelect>>")
            logger.debug("Treeview unbind <<TreeviewSelect>> in populate_category_word_tree (base_dir invalid)")
            self.status_message.set("Error: Base directory invalid. Cannot load categories.")
            return

        dir_structure = get_directory_structure(base_dir)

        if not dir_structure:
            logger.info(f"No category subdirectories found or error scanning {base_dir}.")
            self.category_word_tree.insert("", "end", text="No categories found or error.", open=False)
            self.category_word_tree.unbind("<<TreeviewSelect>>")
            self.status_message.set("No categories found or error. Check base directory.")
            return

        for category_name, words in dir_structure.items():
                category_id = self.category_word_tree.insert("", "end", text=category_name, open=False, tags=('category',))
                if words:
                    for word_name in words:
                        self.category_word_tree.insert(category_id, "end", text=word_name, tags=('word',))
                else:
                    self.category_word_tree.insert(category_id, "end", text=" (No words)", tags=('empty_category_info',))
            
        self.category_word_tree.bind("<<TreeviewSelect>>", self.on_tree_item_select)
        logger.debug("Treeview bind <<TreeviewSelect>> in populate_category_word_tree (success)")
        self.status_message.set("Select a Category, then a Word from the tree.")
        logger.info(f"Populated category/word tree with {len(dir_structure)} categories from {base_dir}.")

    @log_method_call
    def on_tree_item_select(self, event=None):
        """Handles selection changes in the category/word TreeView."""
        # logger.debug("on_tree_item_select called.") # Covered by decorator
        selected_item_id = self.category_word_tree.focus() # Get the ID of the focused/selected item

        if not selected_item_id: # No item selected (e.g., selection cleared)
            self.selected_category.set("")
            self.selected_word.set("")
            logger.debug("Tree selection cleared. Category and Word reset.")
        else:
            item = self.category_word_tree.item(selected_item_id)
            item_text = item['text']
            item_tags = item['tags']

            if 'word' in item_tags:
                parent_id = self.category_word_tree.parent(selected_item_id)
                self.selected_category.set(self.category_word_tree.item(parent_id, "text"))
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
        validated_filepaths = []
        for fp in filepaths:
            if os.path.splitext(fp)[1].lower() in VALID_VIDEO_EXTENSIONS:
                validated_filepaths.append(fp)
            else:
                logger.warning(f"Skipping non-video file (based on extension): {fp}")
        
        if not validated_filepaths:
            messagebox.showwarning("No Valid Video Files", "No files with recognized video extensions were found among the provided files.")
            logger.warning("No valid video files after extension check from provided list.")
            # Clear selections if all provided files were invalid
            self.selected_file_paths_tuple = ()
            self.selected_files_info.set("No files selected")
            self.clear_analysis_results() # Clean up UI
            self.status_message.set("No valid video files found.")
            self.check_button_state()
            return

        filepaths_to_process = tuple(validated_filepaths)
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
            if i < num_selected:
                current_score = scores[i]
                if current_score is not None:
                    self.score_display_vars[i].set(f"Score: {current_score:.3f}")
                    logger.debug(f"Set score for slot {i}: {current_score:.3f}")
                else:
                    self.score_display_vars[i].set("Score: N/A")
                    logger.debug(f"Set score for slot {i}: N/A (score was None)")
            else:
                self.score_display_vars[i].set("Score: -")
                logger.debug(f"Set score for slot {i}: - (index out of bounds for selected files)")
        
        valid_scores = [s for s in scores[:num_selected] if s is not None]
        max_score = 0.0
        if valid_scores: max_score = max(valid_scores)
        return valid_scores, max_score

    @log_method_call
    def _create_previews_and_init_checkboxes(self, list_of_preview_pil_list, num_selected):
        # logger.debug("Creating PhotoImage objects and configuring checkboxes.") # Covered by decorator
        self.preview_photo_images = [[] for _ in range(MAX_VIDEOS)] 
        checkbox_states_after_load = {} 
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
            
            self.preview_photo_images[video_idx] = photo_images_for_video

            checkbox_state = 'normal' if preview_success else 'disabled'
            checkbox_states_after_load[video_idx] = checkbox_state

            if video_idx < len(self.confirm_checkboxes):
                self.confirm_checkboxes[video_idx].config(state=checkbox_state)
                logger.debug(f"Checkbox for slot {video_idx} set to state '{checkbox_state}'.")
                if checkbox_state == 'disabled' and self.per_video_confirmed_vars[video_idx].get():
                    self.per_video_confirmed_vars[video_idx].set(False)
                    logger.debug(f"Checkbox for slot {video_idx} was disabled and unchecked.")
            
            if preview_success:
                num_videos_with_valid_previews += 1
                logger.debug(f"Starting preview animation for slot {video_idx}.")
                self.start_preview_animation(video_idx)
            elif video_idx < len(self.preview_labels): # Ensure label is empty if no previews
                self.preview_labels[video_idx].config(image='')
                logger.debug(f"No valid previews for slot {video_idx}. Animation not started, label cleared.")
        
        return num_videos_with_valid_previews, checkbox_states_after_load

    @log_method_call
    def _apply_pre_marking_logic(self, scores, num_selected, filepaths, checkbox_states_after_load, valid_scores_list, max_score_val):
        # logger.debug("Starting pre-marking process.") # Covered by decorator
        if num_selected <= 1 or len(valid_scores_list) < 2:
            logger.info("Pre-marking skipped: not enough videos or valid scores.")
            self.initial_confirmation_state = [self.per_video_confirmed_vars[i].get() if i < num_selected else False for i in range(num_selected)]
            logger.debug(f"Stored initial confirmation state (pre-marking skipped): {self.initial_confirmation_state[:num_selected]}")
            return

        std_dev = np.std(valid_scores_list)
        score_threshold_sd = max_score_val - PRE_MARKING_SD_FACTOR * std_dev
        logger.info(f"Pre-marking logic: {num_selected} videos, {len(valid_scores_list)} valid. MaxScore={max_score_val:.3f}, SD={std_dev:.3f}, SD_Threshold={score_threshold_sd:.3f}, Fixed_Threshold={PRE_MARKING_SCORE_THRESHOLD}")

        for i in range(num_selected):
            current_score = scores[i] # Use the full scores list passed
            is_enabled = checkbox_states_after_load.get(i) == 'normal'

            if is_enabled and current_score is not None and \
               (current_score >= score_threshold_sd or current_score >= PRE_MARKING_SCORE_THRESHOLD):
                logger.info(f"Pre-marking video index {i} ('{os.path.basename(filepaths[i])}') - Score {current_score:.3f}. Checkbox state '{checkbox_states_after_load.get(i)}'.")
                self.per_video_confirmed_vars[i].set(True)
            elif is_enabled:
                logger.debug(f"Video index {i} ('{os.path.basename(filepaths[i])}') not pre-marked. Score {current_score}. Checkbox state '{checkbox_states_after_load.get(i)}'.")
            else:
                logger.debug(f"Video index {i} ('{os.path.basename(filepaths[i])}') not pre-marked (checkbox disabled). Score: {current_score}.")
        
        self.initial_confirmation_state = [self.per_video_confirmed_vars[i].get() if i < num_selected else False for i in range(num_selected)]
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
        
        num_valid_previews, cb_states = self._create_previews_and_init_checkboxes(list_of_preview_pil_list, num_selected)
        
        self._apply_pre_marking_logic(self.per_video_similarity_scores, num_selected, filepaths, cb_states, valid_scores_list, max_score_val)
        
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
    def start_preview_animation(self, video_idx):
        """Starts or restarts the animation loop for a specific video slot."""
        # logger.debug(f"Attempting to start/restart animation for slot {video_idx}.") # Covered by decorator
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

    @log_method_call
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
             if video_idx < MAX_VIDEOS and video_idx < len(self.preview_after_ids): # Check bounds before trying to stop
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

    @log_method_call
    def calculate_and_display_take_assignment(self):
        """Calculates the available take range based on existing files and number selected."""
        # logger.debug("calculate_and_display_take_assignment called.") # Covered by decorator
        base_dir = self.base_directory.get() 
        category = self.selected_category.get() 
        word = self.selected_word.get()         
        interpreter_id = self.selected_interpreter_id.get()
        num_selected = len(self.selected_file_paths_tuple)

        if num_selected == 0:
            self.take_assignment_display.set("Takes: -")
            logger.debug("No files selected, take assignment set to '-'.")
            return

        if not all([base_dir, category, word, interpreter_id]): 
            self.take_assignment_display.set("Takes: Error")
            self.status_message.set("Error: Prerequisite selection missing for take calculation.")
            logger.error(f"Missing prerequisites for take calculation: BaseDir='{base_dir}', Category='{category}', Word='{word}', InterpreterID='{interpreter_id}'.")
            return

        target_folder_path = os.path.join(base_dir, category, word, interpreter_id) 
        logger.debug(f"Checking existing takes in target folder: {target_folder_path}")

        start_take = determine_next_take_number(target_folder_path, interpreter_id)
        if start_take == -1: # Error occurred
            self.take_assignment_display.set("Takes: Error")
            self.status_message.set(f"Error checking existing takes in target folder.")
            return

        logger.debug(f"Calculated starting take: {start_take}.")


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

    @log_method_call
    def check_button_state(self):
        """Enables or disables widgets based on the application state (Set-Once Workflow)."""
        # logger.debug("check_button_state called.") # Covered by decorator

        # Initial Setup Phase
        if not self.initial_setup_done:
            logger.debug("Initial setup phase.")
            # Base directory button enabled only if no base dir is set yet
            self.base_dir_button.config(state='normal' if not self.base_directory.get() else 'disabled')
            # Interpreter ID enabled only after base dir is set
            self.interpreter_id_combobox.config(state='readonly' if self.base_directory.get() and not self.initial_setup_done else 'disabled')
            self.category_word_tree.config(state='disabled')
            self.select_files_button.config(state='disabled')
            self.process_button.config(state='disabled')
            logger.debug("Widget states updated for initial setup phase.")
            return

        # Post-Initial Setup Phase
        logger.debug("Post-initial setup phase.")
        # Base dir and ID are disabled after initial setup
        self.base_dir_button.config(state='disabled')
        self.interpreter_id_combobox.config(state='disabled')
        logger.debug("Base dir and ID widgets disabled.")

        if self.initial_setup_done and self.base_directory.get():
            self.category_word_tree.bind("<<TreeviewSelect>>", self.on_tree_item_select)
            logger.debug("Treeview bind <<TreeviewSelect>> in check_button_state (enabled)")
            tree_state_log = "bound"
        else:
            self.category_word_tree.unbind("<<TreeviewSelect>>")
            logger.debug("Treeview unbind <<TreeviewSelect>> in check_button_state (disabled)")
            tree_state_log = "unbound"
        logger.debug(f"Category/Word TreeView event <<TreeviewSelect>> is {tree_state_log}")

        word_selected = bool(self.selected_word.get())
        self.select_files_button.config(state='normal' if word_selected else 'disabled')
        logger.debug(f"File select button state set to {'normal' if word_selected else 'disabled'}.")

        files_selected = len(self.selected_file_paths_tuple) > 0
        take_info = self.take_assignment_display.get()
        valid_take_assignment = not take_info.startswith("Takes: FULL") and not take_info.startswith("Takes: Error") and take_info != "Takes: -"
        num_selected_actual = len(self.selected_file_paths_tuple)
        at_least_one_confirmed = any(var.get() for i, var in enumerate(self.per_video_confirmed_vars) if i < num_selected_actual)
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

        final_confirmation_states = [self.per_video_confirmed_vars[i].get() for i in range(num_selected)]
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
        self.confirmed_indices_for_processing_ = [i for i, var in enumerate(self.per_video_confirmed_vars) if i < num_selected and var.get()]

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
        start_take = determine_next_take_number(self.target_folder_path_for_processing_, self.interpreter_id_for_processing_)
        if start_take == -1:
            messagebox.showerror("Error", "Failed to re-verify existing takes before processing.")
            logger.error(f"Error determining next take in {self.target_folder_path_for_processing_}")
            return None 

        logger.info(f"Re-verified. Starting take for new files will be {start_take}.")
        
        num_approved = len(self.confirmed_indices_for_processing_)
        final_take_needed = start_take + num_approved - 1

        if final_take_needed > 4: 
            available_slots = 4 - start_take + 1
            messagebox.showerror("Error", f"Cannot process. Too many videos approved ({num_approved}) for available slots (Max {available_slots}, starting from take {start_take}). Please uncheck some.")
            logger.warning(f"Processing aborted: {num_approved} videos approved, but only {available_slots} slots available starting from {start_take}. Max take needed: {final_take_needed}.")
            return None
        
        self.start_take_for_processing_ = start_take 
        return start_take

    @log_method_call
    def _execute_file_operations(self):
        """Moves and renames the approved video files."""
        errors = []
        success_count = 0
        num_approved = len(self.confirmed_indices_for_processing_)
        
        self.status_message.set(f"Processing {num_approved} approved file(s)...")
        self.process_button.config(state='disabled')
        self.master.update_idletasks()

        take_counter = 0 
        logger.info(f"Starting file movement for {num_approved} approved files.")

        for index, source_video_path in enumerate(self.selected_file_paths_tuple):
            if index in self.confirmed_indices_for_processing_:
                assigned_take_number = self.start_take_for_processing_ + take_counter
                _, file_extension = os.path.splitext(source_video_path)
                new_filename = f"{self.interpreter_id_for_processing_}_{assigned_take_number}{file_extension}"

                success, error_msg = move_and_rename_video(
                    source_video_path,
                    self.target_folder_path_for_processing_,
                    new_filename
                )
                if success:
                    success_count += 1
                    take_counter += 1
                else:
                    errors.append(error_msg)
        
        logger.info(f"Finished file movement. Successfully processed {success_count} files with {len(errors)} errors.")
        return success_count, errors, num_approved

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
        self.selected_word.set("") 
        self.select_files_button.config(state='disabled')
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

        success_count, errors, num_approved = self._execute_file_operations()
        
        self._finalize_processing_and_reset_ui(success_count, errors, num_approved)
        
        del self.confirmed_indices_for_processing_
        del self.target_folder_path_for_processing_
        del self.interpreter_id_for_processing_
        del self.start_take_for_processing_

    @log_method_call
    def _toggle_approval_for_slot(self, slot_index):
        """
        Toggles the approval state for a given video slot if its checkbox is enabled.
        Called when the preview label for a slot is clicked.
        """
        # logger.debug(f"Preview clicked for slot {slot_index}. Attempting to toggle approval.") # Covered by decorator
        if not (0 <= slot_index < MAX_VIDEOS):
            logger.warning(f"Invalid slot_index {slot_index} in _toggle_approval_for_slot.")
            return

        checkbox = self.confirm_checkboxes[slot_index]
        var = self.per_video_confirmed_vars[slot_index]

        current_checkbox_state = checkbox.cget('state')
        logger.debug(f"Checkbox for slot {slot_index} current state: '{current_checkbox_state}'")

        if current_checkbox_state == 'normal': 
            current_state = var.get()
            var.set(not current_state)
            logger.info(f"Approval for slot {slot_index} toggled (var was {current_state}, now {var.get()}) by preview click. Checkbox state was '{current_checkbox_state}'.")
            self.check_button_state() 
        else:
            logger.debug(f"Approval checkbox for slot {slot_index} is disabled. Click on preview ignored.")

    @log_method_call
    def on_closing(self):
        """Cancel all pending 'after' jobs and destroy the window when the window is closed."""
        # logger.info("Application closing requested. Stopping animations and destroying window.") # Covered by decorator
        for i in range(MAX_VIDEOS):
            if self.preview_after_ids[i] is not None:
                try:
                    self.master.after_cancel(self.preview_after_ids[i])
                    logger.debug(f"Cancelled 'after' job {self.preview_after_ids[i]} during closing.")
                except tk.TclError:
                    logger.debug(f"Could not cancel 'after' job for slot {i} during closing (job already finished/cancelled).")
                    pass
                self.preview_after_ids[i] = None
        logger.debug("All known animation 'after' jobs cancelled.")
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
