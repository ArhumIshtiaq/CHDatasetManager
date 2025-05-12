import sys
import os

# Determine the project root directory.
# This assumes your test file is located at <project_root>/CHDatasetManager/tests/unit/
# Adjust the number of '..' if your project structure is different.
project_root_for_imports = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root_for_imports not in sys.path:
    sys.path.insert(0, project_root_for_imports)

import pytest
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from unittest import mock
import queue # Moved after sys.path modification
import logging # Needed for logging levels used in decorator tests
import functools # Needed for functools.wraps in mock decorator

# Mock external libraries and internal modules before importing the App class
# This is necessary for modules imported at the top level of VideoPlacerV2.py
mock_cv2 = mock.MagicMock()
mock_PIL_Image = mock.MagicMock()
mock_PIL_ImageTk = mock.MagicMock()
mock_skimage_metrics_ssim = mock.MagicMock()
mock_numpy = mock.MagicMock()
mock_tkinterdnd2 = mock.MagicMock()

sys.modules['cv2'] = mock_cv2
sys.modules['PIL'] = mock.MagicMock() # Mock the PIL package itself
sys.modules['PIL.Image'] = mock_PIL_Image
sys.modules['PIL.ImageTk'] = mock_PIL_ImageTk
sys.modules['skimage'] = mock.MagicMock() # Mock the skimage package
sys.modules['skimage.metrics'] = mock.MagicMock() # Mock the metrics submodule
sys.modules['skimage.metrics.structural_similarity'] = mock_skimage_metrics_ssim
sys.modules['numpy'] = mock_numpy
sys.modules['tkinterdnd2'] = mock_tkinterdnd2

# Before mocking CHDatasetManager.logger_config, import the actual decorator if it's tested here.
# This assumes log_method_call is defined in CHDatasetManager.logger_config
# If it's defined elsewhere (e.g., VideoPlacerV2.py), this import needs adjustment,
# or the decorator tests might need to import it from VideoPlacerV2.
try:
    from CHDatasetManager.logger_config import log_method_call as actual_log_method_call
except ImportError:
    # Fallback if logger_config or log_method_call doesn't exist or is in VideoPlacerV2
    # If log_method_call is in VideoPlacerV2, it will be imported later.
    actual_log_method_call = None

# Mock internal modules that the App class imports
mock_constants = mock.MagicMock()
mock_constants.MAX_VIDEOS = 4
mock_constants.PRE_MARKING_SD_FACTOR = 0.5
mock_constants.PRE_MARKING_SCORE_THRESHOLD = 0.85
mock_constants.VALID_VIDEO_EXTENSIONS = ('.mp4', '.mov')
# Ensure VIDEO_TYPES_FILTER is mocked if constants.VIDEO_TYPES_FILTER is used
mock_constants.VIDEO_TYPES_FILTER = [("Video files", "*.mp4 *.mov"), ("All files", "*.*")]

mock_constants.VERIFICATION_LOG_FILE = "verification_log.csv"
mock_constants.NUM_SSIM_KEYFRAMES = 5
mock_constants.NUM_PREVIEW_FRAMES = 5
mock_constants.THUMBNAIL_WIDTH = 160
mock_constants.THUMBNAIL_HEIGHT = 90
mock_constants.PREVIEW_ANIMATION_DELAY = 250
mock_constants.INTERPRETER_ID_RANGE = range(1, 11)
mock_constants.MAX_TAKES_PER_WORD_INTERPRETER = 4

mock_logger_config = mock.MagicMock()
mock_logger = mock.MagicMock()
mock_logger_config.logger = mock_logger # Mock the logger instance

# Provide a pass-through mock for log_method_call if VideoPlacerApp uses it
def pass_through_decorator_mock(func_to_decorate):
    @functools.wraps(func_to_decorate)
    def wrapper(*args, **kwargs):
        return func_to_decorate(*args, **kwargs)
    return wrapper

mock_logger_config.log_method_call = mock.MagicMock(side_effect=pass_through_decorator_mock)


mock_video_processor_ops = mock.MagicMock()
mock_VideoProcessor_class = mock.MagicMock()
mock_video_processor_ops.VideoProcessor = mock_VideoProcessor_class

mock_file_system_ops = mock.MagicMock()
mock_file_system_ops.get_directory_structure = mock.Mock()
mock_file_system_ops.calculate_take_assignment_details = mock.Mock()
mock_file_system_ops.execute_video_processing_fs = mock.Mock()
mock_file_system_ops.move_and_rename_video = mock.Mock() # Although execute_video_processing_fs uses it, good to mock if needed directly
mock_file_system_ops.log_verification_to_csv = mock.Mock()
mock_file_system_ops.get_app_base_path = mock.Mock(return_value="/fake/app/base") # Mock this helper too

# Patch sys.modules to inject our mocks
sys.modules['CHDatasetManager.constants'] = mock_constants
sys.modules['CHDatasetManager.logger_config'] = mock_logger_config
sys.modules['CHDatasetManager.video_processing_operations'] = mock_video_processor_ops
sys.modules['CHDatasetManager.file_system_operations'] = mock_file_system_ops

# Mock TkinterDnD availability at the script level
tkinter_dnd_available = False # Assume not available for most tests unless specifically patched

# Now import the class under test
from CHDatasetManager.VideoPlacerV2 import VideoPlacerApp # lgtm [py/unused-import]
# If log_method_call is defined in VideoPlacerV2.py and needs to be tested:
# from CHDatasetManager.VideoPlacerV2 import VideoPlacerApp, log_method_call as video_placer_log_method_call
# And then use video_placer_log_method_call in decorator tests if actual_log_method_call is None.
# For now, we assume actual_log_method_call from logger_config is the one to test.

# --- Fixtures ---

@pytest.fixture
def mock_master():
    """Provides a mock Tkinter root window."""
    # Use a real Tk root but withdraw it to avoid showing a window
    root = tk.Tk()
    root.withdraw()
    # Add necessary mock methods if the App expects them on the master
    root.update_idletasks = mock.Mock()
    root.winfo_screenwidth = mock.Mock(return_value=1920)
    root.winfo_screenheight = mock.Mock(return_value=1080)
    root.geometry = mock.Mock()
    root.minsize = mock.Mock()
    root.after = mock.Mock() # Mock the after method used for the queue checker
    root.protocol = mock.Mock() # Mock protocol for WM_DELETE_WINDOW

    yield root

    # Clean up the real Tk root
    root.destroy()

@pytest.fixture
def mock_frames():
    """Provides mocks for the modular UI frames."""
    mock_setup_frame = mock.MagicMock(spec=ttk.Frame)
    mock_setup_frame.base_dir_button = mock.MagicMock(spec=ttk.Button)
    mock_setup_frame.interpreter_id_combobox = mock.MagicMock(spec=ttk.Combobox)
    mock_setup_frame.update_widget_states = mock.Mock()

    mock_tree_frame = mock.MagicMock(spec=ttk.Frame)
    mock_tree_frame.category_word_tree = mock.MagicMock(spec=ttk.Treeview)
    mock_tree_frame.populate = mock.Mock()
    mock_tree_frame.bind_select_event = mock.Mock()
    mock_tree_frame.unbind_select_event = mock.Mock()
    mock_tree_frame.focus_on_tree = mock.Mock()
    mock_tree_frame.get_focused_item_id = mock.Mock()
    mock_tree_frame.get_item_details = mock.Mock()
    mock_tree_frame.get_parent_id = mock.Mock()
    mock_tree_frame.clear_selection = mock.Mock()

    mock_fileselect_frame = mock.MagicMock(spec=ttk.Frame)
    mock_fileselect_frame.select_files_button = mock.MagicMock(spec=ttk.Button)
    mock_fileselect_frame.update_button_state = mock.Mock()

    mock_verification_frame = mock.MagicMock(spec=ttk.Frame)
    mock_verification_frame.preview_labels = [mock.MagicMock(spec=ttk.Label) for _ in range(mock_constants.MAX_VIDEOS)]
    mock_verification_frame.score_display_vars = [mock.MagicMock(spec=tk.StringVar) for _ in range(mock_constants.MAX_VIDEOS)]
    mock_verification_frame.score_labels = [mock.MagicMock(spec=ttk.Label) for _ in range(mock_constants.MAX_VIDEOS)]
    mock_verification_frame.per_video_confirmed_vars = [mock.MagicMock(spec=tk.BooleanVar) for _ in range(mock_constants.MAX_VIDEOS)]
    mock_verification_frame.confirm_checkboxes = [mock.MagicMock(spec=ttk.Checkbutton) for _ in range(mock_constants.MAX_VIDEOS)]
    # Mock cget('state') for checkboxes
    for cb in mock_verification_frame.confirm_checkboxes:
        cb.cget.return_value = 'normal' # Default to enabled

    mock_verification_frame.clear_all_slots = mock.Mock()
    mock_verification_frame.update_slot_preview = mock.Mock()
    mock_verification_frame.set_slot_score = mock.Mock()
    mock_verification_frame.set_slot_approved = mock.Mock()
    mock_verification_frame.get_slot_approved_state = mock.Mock()
    mock_verification_frame.enable_disable_slot_controls = mock.Mock()
    mock_verification_frame.bind_preview_click_callback = mock.Mock()
    mock_verification_frame.bind_checkbox_change_callback = mock.Mock()
    mock_verification_frame.on_app_closing = mock.Mock()

    return (mock_setup_frame, mock_tree_frame, mock_fileselect_frame, mock_verification_frame)

@pytest.fixture
def app_instance(mock_master, mock_frames):
    """Provides a VideoPlacerApp instance with mocked dependencies."""
    mock_setup_frame, mock_tree_frame, mock_fileselect_frame, mock_verification_frame = mock_frames

    # Patch the frame classes during App instantiation
    with mock.patch('CHDatasetManager.VideoPlacerV2.SetupWidgetsFrame', return_value=mock_setup_frame), \
         mock.patch('CHDatasetManager.VideoPlacerV2.CategoryWordTreeFrame', return_value=mock_tree_frame), \
         mock.patch('CHDatasetManager.VideoPlacerV2.FileSelectionWidgetsFrame', return_value=mock_fileselect_frame), \
         mock.patch('CHDatasetManager.VideoPlacerV2.VerificationPanelFrame', return_value=mock_verification_frame), \
         mock.patch('CHDatasetManager.VideoPlacerV2.tkinter_dnd_available', False): # Assume DND is off by default

        app = VideoPlacerApp(mock_master)

    # Manually attach the mock frames to the app instance if needed for direct access in tests
    app.setup_widgets_frame = mock_setup_frame
    app.category_word_tree_frame = mock_tree_frame
    app.file_selection_widgets_frame = mock_fileselect_frame
    app.verification_panel_frame = mock_verification_frame

    # Mock the VideoProcessor instance created in __init__
    app.video_processor = mock.MagicMock(spec=mock_video_processor_ops.VideoProcessor)

    # Reset mocks after init if their calls during init are not part of the test
    mock_setup_frame.reset_mock()
    mock_tree_frame.reset_mock()
    mock_fileselect_frame.reset_mock()
    mock_verification_frame.reset_mock()
    mock_VideoProcessor_class.reset_mock()
    mock_master.after.reset_mock() # Reset the initial after call
    mock_master.protocol.reset_mock() # Reset the initial protocol call

    # Set some default state for easier testing
    app.base_directory.set("/fake/base/dir")
    app.selected_interpreter_id.set("001")
    app.initial_setup_done = True # Assume initial setup is done unless testing that flow
    app.selected_category.set("CategoryA")
    app.selected_word.set("Word1")
    app.selected_file_paths_tuple = () # Start with no files selected
    app.video_approval_states = [False] * mock_constants.MAX_VIDEOS
    app.per_video_similarity_scores = [None] * mock_constants.MAX_VIDEOS
    app.initial_confirmation_state = [False] * mock_constants.MAX_VIDEOS
    app.is_analysis_running = False
    app.globally_processed_video_paths = set()

    # Mock check_button_state as it's called frequently
    app.check_button_state = mock.Mock()

    yield app

# --- Tests for core workflow methods ---

@mock.patch('CHDatasetManager.VideoPlacerV2.filedialog.askdirectory', return_value="/selected/base/dir")
def test_select_base_dir_success(mock_askdirectory, app_instance, mock_frames):
    """Test successful base directory selection."""
    app_instance.initial_setup_done = False # Ensure initial setup is not done
    app_instance.base_directory.set("") # Clear initial state
    app_instance.check_button_state.reset_mock() # Reset mock after fixture setup

    app_instance.select_base_dir()

    mock_askdirectory.assert_called_once_with(title="Select Base Directory (Set Once)")
    assert app_instance.base_directory.get() == "/selected/base/dir"
    assert app_instance.status_message.get() == "Step 2: Select your Interpreter ID"
    # Verify widget state updates via the frame mock
    mock_frames[0].update_widget_states.assert_called_once_with(initial_setup_done=False, base_dir_set=True)
    mock_frames[0].interpreter_id_combobox.focus.assert_called_once()
    mock_frames[1].unbind_select_event.assert_called_once()
    mock_frames[2].update_button_state.assert_called_once_with(word_selected=False)
    app_instance.process_button.config.assert_called_once_with(state='disabled')
    app_instance.check_button_state.assert_not_called() # Should not be called if initial setup is not done

@mock.patch('CHDatasetManager.VideoPlacerV2.filedialog.askdirectory', return_value="") # User cancels
def test_select_base_dir_cancelled(mock_askdirectory, app_instance, mock_frames):
    """Test cancelling base directory selection."""
    app_instance.initial_setup_done = False
    app_instance.base_directory.set("")
    app_instance.check_button_state.reset_mock()

    app_instance.select_base_dir()

    mock_askdirectory.assert_called_once_with(title="Select Base Directory (Set Once)")
    assert app_instance.base_directory.get() == ""
    assert "Base directory selection cancelled" in app_instance.status_message.get()
    # Widget states should remain as they were before the call (likely initial state)
    # We can assert they were NOT called to change state based on selection success
    mock_frames[0].update_widget_states.assert_not_called()
    mock_frames[0].interpreter_id_combobox.focus.assert_not_called()
    mock_frames[1].unbind_select_event.assert_not_called()
    mock_frames[2].update_button_state.assert_not_called()
    app_instance.process_button.config.assert_not_called()
    app_instance.check_button_state.assert_not_called()

def test_on_id_select_success(app_instance, mock_frames):
    """Test successful interpreter ID selection."""
    app_instance.initial_setup_done = False # Ensure initial setup is not done
    app_instance.selected_interpreter_id.set("005")
    app_instance.base_directory.set("/fake/base") # Ensure base dir is set
    app_instance.check_button_state.reset_mock()
    app_instance.populate_category_word_tree = mock.Mock() # Mock this dependency
    app_instance.clear_analysis_results = mock.Mock() # Mock this dependency

    app_instance.on_id_select()

    assert app_instance.initial_setup_done is True
    assert app_instance.selected_interpreter_id.get() == "005"
    mock_frames[0].update_widget_states.assert_called_once_with(initial_setup_done=True, base_dir_set=True)
    assert app_instance.status_message.get() == "Step 3: Select a Word from the tree below."
    app_instance.populate_category_word_tree.assert_called_once()
    mock_frames[1].focus_on_tree.assert_called_once()
    assert app_instance.selected_category.get() == ""
    assert app_instance.selected_word.get() == ""
    mock_frames[1].clear_selection.assert_called_once()
    assert app_instance.selected_file_paths_tuple == ()
    assert app_instance.selected_files_info.get() == "No files selected"
    app_instance.clear_analysis_results.assert_called_once()
    mock_frames[2].update_button_state.assert_called_once_with(word_selected=False)
    app_instance.check_button_state.assert_called_once()

def test_on_id_select_already_done(app_instance):
    """Test selecting ID when initial setup is already done."""
    app_instance.initial_setup_done = True
    app_instance.selected_interpreter_id.set("001") # Already set
    app_instance.check_button_state.reset_mock()
    app_instance.populate_category_word_tree = mock.Mock()
    app_instance.clear_analysis_results = mock.Mock()

    app_instance.on_id_select() # Call again

    assert app_instance.initial_setup_done is True # Remains true
    # No state changes or dependent calls should happen
    app_instance.populate_category_word_tree.assert_not_called()
    app_instance.clear_analysis_results.assert_not_called()
    app_instance.check_button_state.assert_not_called()

@mock.patch('CHDatasetManager.VideoPlacerV2.get_directory_structure', return_value={"CatA": ["Word1", "Word2"], "CatB": []})
def test_populate_category_word_tree_success(mock_get_structure, app_instance, mock_frames):
    """Test populating the treeview successfully."""
    app_instance.base_directory.set("/fake/base")
    app_instance.status_message.set("Initial status")

    app_instance.populate_category_word_tree()

    mock_get_structure.assert_called_once_with("/fake/base")
    mock_frames[1].populate.assert_called_once_with({"CatA": ["Word1", "Word2"], "CatB": []})
    mock_frames[1].bind_select_event.assert_called_once_with(app_instance.on_tree_item_select)
    assert app_instance.status_message.get() == "Select a Category, then a Word from the tree."

@mock.patch('CHDatasetManager.VideoPlacerV2.get_directory_structure', return_value={})
def test_populate_category_word_tree_empty(mock_get_structure, app_instance, mock_frames):
    """Test populating the treeview when no categories are found."""
    app_instance.base_directory.set("/fake/base")
    app_instance.status_message.set("Initial status")
    mock_frames[1].populate.return_value = False # Simulate populate returning False for empty

    app_instance.populate_category_word_tree()

    mock_get_structure.assert_called_once_with("/fake/base")
    mock_frames[1].populate.assert_called_once_with({})
    mock_frames[1].unbind_select_event.assert_called_once()
    assert app_instance.status_message.get() == "No categories found or error. Check base directory."

def test_populate_category_word_tree_invalid_base_dir(app_instance, mock_frames):
    """Test populating the treeview with an invalid base directory."""
    app_instance.base_directory.set("") # Invalid base dir
    app_instance.status_message.set("Initial status")
    mock_frames[1].populate.return_value = False # Simulate populate returning False

    app_instance.populate_category_word_tree()

    # get_directory_structure should not be called if base_dir is invalid
    mock_file_system_ops.get_directory_structure.assert_not_called()
    mock_frames[1].populate.assert_called_once_with(None) # Should pass None
    mock_frames[1].unbind_select_event.assert_called_once()
    assert app_instance.status_message.get() == "Error: Base directory invalid. Cannot load categories."

def test_on_tree_item_select_word(app_instance, mock_frames):
    """Test selecting a word in the treeview."""
    mock_frames[1].get_focused_item_id.return_value = "word_item_id"
    mock_frames[1].get_item_details.side_effect = [
        {'text': 'Word1', 'tags': ('word',)}, # Details for the selected word
        {'text': 'CategoryA', 'tags': ('category',)} # Details for the parent category
    ]
    mock_frames[1].get_parent_id.return_value = "category_item_id"
    app_instance.check_button_state.reset_mock()
    app_instance.clear_analysis_results = mock.Mock()
    app_instance.calculate_and_display_take_assignment = mock.Mock()

    app_instance.on_tree_item_select()

    assert app_instance.selected_category.get() == "CategoryA"
    assert app_instance.selected_word.get() == "Word1"
    assert app_instance.selected_file_paths_tuple == ()
    assert app_instance.selected_files_info.get() == "No files selected"
    app_instance.clear_analysis_results.assert_called_once()
    assert app_instance.status_message.get() == "Step 4: Select video files for Word 'Word1'"
    app_instance.calculate_and_display_take_assignment.assert_called_once()
    app_instance.check_button_state.assert_called_once()
    mock_frames[1].clear_selection.assert_called_once() # Should clear selection after processing

def test_on_tree_item_select_category(app_instance, mock_frames):
    """Test selecting only a category in the treeview."""
    mock_frames[1].get_focused_item_id.return_value = "category_item_id"
    mock_frames[1].get_item_details.return_value = {'text': 'CategoryB', 'tags': ('category',)}
    mock_frames[1].get_parent_id.return_value = "" # No parent for category
    app_instance.check_button_state.reset_mock()
    app_instance.clear_analysis_results = mock.Mock()
    app_instance.calculate_and_display_take_assignment = mock.Mock()
    app_instance.take_assignment_display.set("Existing Takes") # Simulate existing display

    app_instance.on_tree_item_select()

    assert app_instance.selected_category.get() == "CategoryB"
    assert app_instance.selected_word.get() == "" # Word should be cleared
    assert app_instance.selected_file_paths_tuple == ()
    assert app_instance.selected_files_info.get() == "No files selected"
    app_instance.clear_analysis_results.assert_called_once()
    assert app_instance.status_message.get() == "Select a Word under Category 'CategoryB'"
    assert app_instance.take_assignment_display.get() == "Takes: -" # Take info should be cleared
    app_instance.calculate_and_display_take_assignment.assert_not_called() # Not called for category only
    app_instance.check_button_state.assert_called_once()
    mock_frames[1].clear_selection.assert_called_once()

def test_on_tree_item_select_cleared(app_instance, mock_frames):
    """Test when the tree selection is cleared."""
    mock_frames[1].get_focused_item_id.return_value = "" # No item focused
    app_instance.selected_category.set("OldCat")
    app_instance.selected_word.set("OldWord")
    app_instance.check_button_state.reset_mock()
    app_instance.clear_analysis_results = mock.Mock()
    app_instance.calculate_and_display_take_assignment = mock.Mock()
    app_instance.take_assignment_display.set("Existing Takes")

    app_instance.on_tree_item_select()

    assert app_instance.selected_category.get() == "" # Should be cleared
    assert app_instance.selected_word.get() == "" # Should be cleared
    assert app_instance.selected_file_paths_tuple == ()
    assert app_instance.selected_files_info.get() == "No files selected"
    app_instance.clear_analysis_results.assert_called_once()
    assert app_instance.status_message.get() == "Select a Category, then a Word."
    assert app_instance.take_assignment_display.get() == "Takes: -"
    app_instance.calculate_and_display_take_assignment.assert_not_called()
    app_instance.check_button_state.assert_called_once()
    mock_frames[1].clear_selection.assert_called_once() # Should clear selection (redundant but confirms call)

@mock.patch('CHDatasetManager.VideoPlacerV2.messagebox.showwarning')
def test__process_filepaths_for_analysis_too_many_files(mock_showwarning, app_instance):
    """Test processing too many files."""
    filepaths = [f"/fake/file{i}.mp4" for i in range(mock_constants.MAX_VIDEOS + 1)] # Too many files
    app_instance.check_button_state.reset_mock()
    app_instance.clear_analysis_results = mock.Mock()
    app_instance.video_processor.run_analysis_in_thread = mock.Mock()

    app_instance._process_filepaths_for_analysis(filepaths)

    mock_showwarning.assert_called_once()
    assert "maximum of" in mock_showwarning.call_args[0][1]
    assert app_instance.selected_file_paths_tuple == () # Should not update state
    app_instance.clear_analysis_results.assert_not_called()
    app_instance.video_processor.run_analysis_in_thread.assert_not_called()
    app_instance.check_button_state.assert_not_called()

def test__process_filepaths_for_analysis_valid_files(app_instance):
    """Test processing a valid list of files."""
    filepaths = ["/fake/file1.mp4", "/fake/file2.mov"]
    app_instance.check_button_state.reset_mock()
    app_instance.clear_analysis_results = mock.Mock()
    app_instance.video_processor.run_analysis_in_thread = mock.Mock()

    app_instance._process_filepaths_for_analysis(filepaths)

    app_instance.clear_analysis_results.assert_called_once()
    assert app_instance.selected_file_paths_tuple == tuple(filepaths)
    assert app_instance.selected_files_info.get() == f"{len(filepaths)} file(s) selected"
    assert app_instance.status_message.get() == "Analyzing videos..."
    assert app_instance.is_analysis_running is True
    app_instance.video_processor.run_analysis_in_thread.assert_called_once_with(tuple(filepaths))
    app_instance.check_button_state.assert_called_once()

@mock.patch('CHDatasetManager.VideoPlacerV2.messagebox.showinfo')
def test__process_filepaths_for_analysis_skips_processed(mock_showinfo, app_instance):
    """Test skipping files that have already been processed."""
    filepaths = ["/fake/file1.mp4", "/fake/file2.mov"]
    app_instance.globally_processed_video_paths.add("/fake/file1.mp4") # file1 is already processed
    app_instance.check_button_state.reset_mock()
    app_instance.clear_analysis_results = mock.Mock()
    app_instance.video_processor.run_analysis_in_thread = mock.Mock()

    app_instance._process_filepaths_for_analysis(filepaths)

    mock_showinfo.assert_called_once()
    assert "file1.mp4" in mock_showinfo.call_args[0][1]
    # Only file2.mov should be processed
    app_instance.clear_analysis_results.assert_called_once()
    assert app_instance.selected_file_paths_tuple == ("/fake/file2.mov",)
    assert app_instance.selected_files_info.get() == "1 file(s) selected"
    assert app_instance.status_message.get() == "Analyzing videos..."
    assert app_instance.is_analysis_running is True
    app_instance.video_processor.run_analysis_in_thread.assert_called_once_with(("/fake/file2.mov",))
    app_instance.check_button_state.assert_called_once()

@mock.patch('CHDatasetManager.VideoPlacerV2.messagebox.showwarning')
def test__process_filepaths_for_analysis_no_valid_new_files(mock_showwarning, app_instance):
    """Test when all provided files are invalid or already processed."""
    filepaths = ["/fake/file1.txt", "/fake/file2.mp4"]
    app_instance.globally_processed_video_paths.add("/fake/file2.mp4") # file2 is already processed
    app_instance.check_button_state.reset_mock()
    app_instance.clear_analysis_results = mock.Mock()
    app_instance.video_processor.run_analysis_in_thread = mock.Mock()

    app_instance._process_filepaths_for_analysis(filepaths)

    # Should show info about skipped file, then warning about no new valid files
    # The messagebox mocks might need adjustment if multiple are expected
    # For now, let's just check the final state and the warning
    mock_showwarning.assert_called_once()
    assert "No new, valid video files" in mock_showwarning.call_args[0][1]

    assert app_instance.selected_file_paths_tuple == () # Should be reset
    assert app_instance.selected_files_info.get() == "No files selected"
    app_instance.clear_analysis_results.assert_called_once() # Should clear UI
    assert app_instance.status_message.get() == "No new, valid video files found to process."
    app_instance.video_processor.run_analysis_in_thread.assert_not_called()
    app_instance.check_button_state.assert_called_once()

@mock.patch('CHDatasetManager.VideoPlacerV2.filedialog.askopenfilenames', return_value=("/fake/file1.mp4", "/fake/file2.mov"))
def test_select_video_files_success(mock_askopenfilenames, app_instance):
    """Test successful file selection via dialog."""
    app_instance.selected_word.set("Word1") # Ensure word is selected
    app_instance._process_filepaths_for_analysis = mock.Mock() # Mock the processing logic
    app_instance.check_button_state.reset_mock()

    app_instance.select_video_files()

    mock_askopenfilenames.assert_called_once_with(title=f"Select up to {mock_constants.MAX_VIDEOS} Video Files", filetypes=mock_constants.VIDEO_TYPES_FILTER)
    app_instance._process_filepaths_for_analysis.assert_called_once_with(("/fake/file1.mp4", "/fake/file2.mov"))
    app_instance.check_button_state.assert_not_called() # Called by _process_filepaths_for_analysis

@mock.patch('CHDatasetManager.VideoPlacerV2.filedialog.askopenfilenames', return_value=()) # User cancels
def test_select_video_files_cancelled(mock_askopenfilenames, app_instance):
    """Test cancelling file selection via dialog."""
    app_instance.selected_word.set("Word1")
    app_instance._process_filepaths_for_analysis = mock.Mock()
    app_instance.clear_analysis_results = mock.Mock()
    app_instance.check_button_state.reset_mock()

    app_instance.select_video_files()

    mock_askopenfilenames.assert_called_once_with(title=f"Select up to {mock_constants.MAX_VIDEOS} Video Files", filetypes=mock_constants.VIDEO_TYPES_FILTER)
    app_instance._process_filepaths_for_analysis.assert_not_called()
    assert app_instance.selected_file_paths_tuple == ()
    assert app_instance.selected_files_info.get() == "No files selected"
    app_instance.clear_analysis_results.assert_called_once()
    assert app_instance.status_message.get() == "Video file selection cancelled."
    app_instance.check_button_state.assert_called_once()

@mock.patch('CHDatasetManager.VideoPlacerV2.messagebox.showwarning')
def test_select_video_files_no_word_selected(mock_showwarning, app_instance):
    """Test attempting to select files before selecting a word."""
    app_instance.selected_word.set("") # No word selected
    app_instance._process_filepaths_for_analysis = mock.Mock()
    app_instance.check_button_state.reset_mock()

    app_instance.select_video_files()

    mock_showwarning.assert_called_once()
    assert "Please select Word first" in mock_showwarning.call_args[0][1]
    app_instance._process_filepaths_for_analysis.assert_not_called()
    app_instance.check_button_state.assert_not_called()

@mock.patch('CHDatasetManager.VideoPlacerV2.messagebox.showwarning')
def test_select_video_files_analysis_running(mock_showwarning, app_instance):
    """Test attempting to select files while analysis is running."""
    app_instance.selected_word.set("Word1")
    app_instance.is_analysis_running = True # Analysis is running
    app_instance._process_filepaths_for_analysis = mock.Mock()
    app_instance.check_button_state.reset_mock()

    app_instance.select_video_files()

    mock_showwarning.assert_called_once()
    assert "Analysis is already in progress" in mock_showwarning.call_args[0][1]
    app_instance._process_filepaths_for_analysis.assert_not_called()
    app_instance.check_button_state.assert_not_called()

# --- Tests for handle_drop_event ---

# Need to mock TkinterDnD.Tk and its splitlist method if testing DND
# Let's create a fixture that simulates a DND-enabled master
@pytest.fixture
def mock_dnd_master():
    """Provides a mock TkinterDnD root window."""
    root = mock.MagicMock(spec=tk.Tk) # Still mock the Tk part
    root.withdraw = mock.Mock() # Mock withdraw if called
    root.update_idletasks = mock.Mock()
    root.winfo_screenwidth = mock.Mock(return_value=1920)
    root.winfo_screenheight = mock.Mock(return_value=1080)
    root.geometry = mock.Mock()
    root.minsize = mock.Mock()
    root.after = mock.Mock()
    root.protocol = mock.Mock()

    # Add TkinterDnD specific mocks
    root.drop_target_register = mock.Mock()
    root.dnd_bind = mock.Mock()
    root.tk = mock.MagicMock() # Mock the .tk interpreter object
    root.tk.splitlist = mock.Mock() # Mock the splitlist method

    yield root

@pytest.fixture
def app_instance_dnd(mock_dnd_master, mock_frames):
    """Provides a VideoPlacerApp instance with mocked DND dependencies."""
    mock_setup_frame, mock_tree_frame, mock_fileselect_frame, mock_verification_frame = mock_frames

    with mock.patch('CHDatasetManager.VideoPlacerV2.SetupWidgetsFrame', return_value=mock_setup_frame), \
         mock.patch('CHDatasetManager.VideoPlacerV2.CategoryWordTreeFrame', return_value=mock_tree_frame), \
         mock.patch('CHDatasetManager.VideoPlacerV2.FileSelectionWidgetsFrame', return_value=mock_fileselect_frame), \
         mock.patch('CHDatasetManager.VideoPlacerV2.VerificationPanelFrame', return_value=mock_verification_frame), \
         mock.patch('CHDatasetManager.VideoPlacerV2.tkinter_dnd_available', True), \
         mock.patch('CHDatasetManager.VideoPlacerV2.TkinterDnD.Tk', return_value=mock_dnd_master): # Ensure App uses the mocked DND master

        app = VideoPlacerApp(mock_dnd_master)

    app.setup_widgets_frame = mock_setup_frame
    app.category_word_tree_frame = mock_tree_frame
    app.file_selection_widgets_frame = mock_fileselect_frame
    app.verification_panel_frame = mock_verification_frame
    app.video_processor = mock.MagicMock(spec=mock_video_processor_ops.VideoProcessor)

    mock_setup_frame.reset_mock()
    mock_tree_frame.reset_mock()
    mock_fileselect_frame.reset_mock()
    mock_verification_frame.reset_mock()
    mock_VideoProcessor_class.reset_mock()
    mock_dnd_master.after.reset_mock()
    mock_dnd_master.protocol.reset_mock()
    mock_dnd_master.drop_target_register.reset_mock()
    mock_dnd_master.dnd_bind.reset_mock()
    mock_dnd_master.tk.splitlist.reset_mock()

    app.base_directory.set("/fake/base/dir")
    app.selected_interpreter_id.set("001")
    app.initial_setup_done = True
    app.selected_category.set("CategoryA")
    app.selected_word.set("Word1") # Ensure word is selected for DND
    app.selected_file_paths_tuple = ()
    app.video_approval_states = [False] * mock_constants.MAX_VIDEOS
    app.per_video_similarity_scores = [None] * mock_constants.MAX_VIDEOS
    app.initial_confirmation_state = [False] * mock_constants.MAX_VIDEOS
    app.is_analysis_running = False
    app.globally_processed_video_paths = set()
    app.check_button_state = mock.Mock()

    yield app

@mock.patch('CHDatasetManager.VideoPlacerV2.messagebox.showwarning')
def test_handle_drop_event_no_word_selected(mock_showwarning, app_instance_dnd):
    """Test dropping files before selecting a word."""
    app_instance_dnd.selected_word.set("") # No word selected
    mock_event = mock.Mock(data="/fake/file1.mp4")
    app_instance_dnd._process_filepaths_for_analysis = mock.Mock()
    app_instance_dnd.check_button_state.reset_mock()

    app_instance_dnd.handle_drop_event(mock_event)

    mock_showwarning.assert_called_once()
    assert "Please select Category and Word first" in mock_showwarning.call_args[0][1]
    app_instance_dnd.master.tk.splitlist.assert_not_called() # Should return early
    app_instance_dnd._process_filepaths_for_analysis.assert_not_called()
    app_instance_dnd.check_button_state.assert_not_called()

@mock.patch('CHDatasetManager.VideoPlacerV2.messagebox.showwarning')
def test_handle_drop_event_analysis_running(mock_showwarning, app_instance_dnd):
    """Test dropping files while analysis is running."""
    app_instance_dnd.selected_word.set("Word1")
    app_instance_dnd.is_analysis_running = True # Analysis is running
    mock_event = mock.Mock(data="/fake/file1.mp4")
    app_instance_dnd._process_filepaths_for_analysis = mock.Mock()
    app_instance_dnd.check_button_state.reset_mock()

    app_instance_dnd.handle_drop_event(mock_event)

    mock_showwarning.assert_called_once()
    assert "Analysis is already in progress" in mock_showwarning.call_args[0][1]
    app_instance_dnd.master.tk.splitlist.assert_not_called() # Should return early
    app_instance_dnd._process_filepaths_for_analysis.assert_not_called()
    app_instance_dnd.check_button_state.assert_not_called()

def test_handle_drop_event_success(app_instance_dnd):
    """Test successful file drop."""
    app_instance_dnd.selected_word.set("Word1")
    app_instance_dnd.is_analysis_running = False
    mock_event = mock.Mock(data="{/fake/file1.mp4} {/fake/file2.mov}") # Simulate Tcl list format
    app_instance_dnd.master.tk.splitlist.return_value = ["/fake/file1.mp4", "/fake/file2.mov"] # Mock parsed paths
    app_instance_dnd._process_filepaths_for_analysis = mock.Mock()
    app_instance_dnd.check_button_state.reset_mock()

    app_instance_dnd.handle_drop_event(mock_event)

    app_instance_dnd.master.tk.splitlist.assert_called_once_with(mock_event.data)
    app_instance_dnd._process_filepaths_for_analysis.assert_called_once_with(["/fake/file1.mp4", "/fake/file2.mov"])
    app_instance_dnd.check_button_state.assert_not_called() # Called by _process_filepaths_for_analysis

@mock.patch('CHDatasetManager.VideoPlacerV2.messagebox.showerror')
def test_handle_drop_event_error_parsing(mock_showerror, app_instance_dnd):
    """Test handling error during file path parsing from drop event."""
    app_instance_dnd.selected_word.set("Word1")
    app_instance_dnd.is_analysis_running = False
    mock_event = mock.Mock(data="invalid data")
    app_instance_dnd.master.tk.splitlist.side_effect = tk.TclError("Fake Tcl error") # Simulate parsing error
    app_instance_dnd._process_filepaths_for_analysis = mock.Mock()
    app_instance_dnd.clear_analysis_results = mock.Mock()
    app_instance_dnd.check_button_state.reset_mock()

    app_instance_dnd.handle_drop_event(mock_event)

    app_instance_dnd.master.tk.splitlist.assert_called_once_with(mock_event.data)
    mock_showerror.assert_called_once()
    assert "Error handling dropped files" in mock_showerror.call_args[0][1]
    app_instance_dnd._process_filepaths_for_analysis.assert_not_called()
    assert app_instance_dnd.selected_file_paths_tuple == () # Should be reset
    assert app_instance_dnd.selected_files_info.get() == "No files selected"
    app_instance_dnd.clear_analysis_results.assert_called_once()
    assert app_instance_dnd.status_message.get() == "Error processing dropped files."
    app_instance_dnd.check_button_state.assert_called_once()

# --- Tests for _handle_analysis_completion ---

def test__handle_analysis_completion_success(app_instance):
    """Test handling successful analysis completion."""
    num_selected = 2
    filepaths = ["/fake/vid1.mp4", "/fake/vid2.mov"]
    scores = [0.9, 0.8, None, None] # Scores for the 2 selected videos
    previews = [[mock.Mock()] * mock_constants.NUM_PREVIEW_FRAMES, [mock.Mock()] * mock_constants.NUM_PREVIEW_FRAMES, [], []] # Previews for the 2 selected
    errors = []

    analysis_result = {
        'type': 'analysis_complete',
        'scores': scores,
        'previews': previews,
        'errors': errors,
        'filepaths': filepaths
    }

    app_instance.is_analysis_running = True # Simulate analysis was running
    app_instance.selected_file_paths_tuple = tuple(filepaths) # Ensure files are selected
    app_instance._update_scores_display_from_analysis = mock.Mock(return_value=([0.9, 0.8], 0.9))
    app_instance._create_previews_and_init_checkboxes = mock.Mock(return_value=num_selected) # Simulate valid previews for all
    app_instance._apply_pre_marking_logic = mock.Mock()
    app_instance._report_analysis_issues = mock.Mock()
    app_instance._finalize_analysis_ui_updates = mock.Mock()
    app_instance.check_button_state.reset_mock()

    app_instance._handle_analysis_completion(analysis_result)

    assert app_instance.is_analysis_running is False
    assert app_instance.per_video_similarity_scores == scores
    app_instance._update_scores_display_from_analysis.assert_called_once_with(scores, num_selected)
    app_instance._create_previews_and_init_checkboxes.assert_called_once_with(previews, num_selected)
    app_instance._apply_pre_marking_logic.assert_called_once_with(scores, num_selected, filepaths, [0.9, 0.8], 0.9)
    app_instance._report_analysis_issues.assert_called_once_with(errors)
    app_instance._finalize_analysis_ui_updates.assert_called_once_with(num_selected)
    app_instance.check_button_state.assert_not_called() # Called by _finalize_analysis_ui_updates

def test__handle_analysis_completion_with_errors(app_instance):
    """Test handling analysis completion with errors."""
    num_selected = 1
    filepaths = ["/fake/vid1.mp4"]
    scores = [None, None, None, None] # Analysis failed, no score
    previews = [[], [], [], []] # No previews extracted
    errors = ["Error opening video", "Error processing frame"]

    analysis_result = {
        'type': 'analysis_complete',
        'scores': scores,
        'previews': previews,
        'errors': errors,
        'filepaths': filepaths
    }

    app_instance.is_analysis_running = True
    app_instance.selected_file_paths_tuple = tuple(filepaths)
    app_instance._update_scores_display_from_analysis = mock.Mock(return_value=([], 0.0))
    app_instance._create_previews_and_init_checkboxes = mock.Mock(return_value=0) # Simulate no valid previews
    app_instance._apply_pre_marking_logic = mock.Mock()
    app_instance._report_analysis_issues = mock.Mock()
    app_instance._finalize_analysis_ui_updates = mock.Mock()
    app_instance.check_button_state.reset_mock()

    app_instance._handle_analysis_completion(analysis_result)

    assert app_instance.is_analysis_running is False
    assert app_instance.per_video_similarity_scores == scores
    app_instance._update_scores_display_from_analysis.assert_called_once_with(scores, num_selected)
    app_instance._create_previews_and_init_checkboxes.assert_called_once_with(previews, num_selected)
    app_instance._apply_pre_marking_logic.assert_called_once_with(scores, num_selected, filepaths, [], 0.0)
    app_instance._report_analysis_issues.assert_called_once_with(errors)
    app_instance._finalize_analysis_ui_updates.assert_called_once_with(0) # Should pass 0 valid previews
    app_instance.check_button_state.assert_not_called() # Called by _finalize_analysis_ui_updates

# --- Tests for _update_scores_display_from_analysis ---

def test__update_scores_display_from_analysis(app_instance, mock_frames):
    """Test updating score labels and returning valid scores/max."""
    scores = [0.9, 0.7, None, 0.85] # 4 slots, 3 selected, one failed
    num_selected = 3

    # Ensure verification panel frame mock is used
    mock_verification_frame = mock_frames[3]
    mock_verification_frame.set_slot_score.reset_mock()

    valid_scores_list, max_score_val = app_instance._update_scores_display_from_analysis(scores, num_selected)

    assert valid_scores_list == [0.9, 0.7, 0.85]
    assert max_score_val == 0.9

    # Verify set_slot_score calls
    mock_verification_frame.set_slot_score.assert_any_call(0, 0.9, score_text_prefix="Score: ")
    mock_verification_frame.set_slot_score.assert_any_call(1, 0.7, score_text_prefix="Score: ")
    mock_verification_frame.set_slot_score.assert_any_call(2, 0.85, score_text_prefix="Score: ")
    mock_verification_frame.set_slot_score.assert_any_call(3, None, score_text_prefix="") # Index 3 is not selected
    assert mock_verification_frame.set_slot_score.call_count == mock_constants.MAX_VIDEOS

def test__update_scores_display_from_analysis_no_selected_videos(app_instance, mock_frames):
    """Test updating scores when no videos are selected."""
    scores = [None] * mock_constants.MAX_VIDEOS
    num_selected = 0

    mock_verification_frame = mock_frames[3]
    mock_verification_frame.set_slot_score.reset_mock()

    valid_scores_list, max_score_val = app_instance._update_scores_display_from_analysis(scores, num_selected)

    assert valid_scores_list == []
    assert max_score_val == 0.0

    # Verify set_slot_score calls (should all be "Score: -")
    for i in range(mock_constants.MAX_VIDEOS):
         mock_verification_frame.set_slot_score.assert_any_call(i, None, score_text_prefix="")
    assert mock_verification_frame.set_slot_score.call_count == mock_constants.MAX_VIDEOS

# --- Tests for _create_previews_and_init_checkboxes ---

@mock.patch('CHDatasetManager.VideoPlacerV2.ImageTk.PhotoImage')
def test__create_previews_and_init_checkboxes_success(mock_PhotoImage, app_instance, mock_frames):
    """Test creating previews and enabling checkboxes for valid previews."""
    num_selected = 2
    # Simulate PIL images for the first 2 videos, None for others
    pil_images_list = [
        [mock.Mock(spec=mock_PIL_Image.Image)] * mock_constants.NUM_PREVIEW_FRAMES, # Video 0 has valid PILs
        [mock.Mock(spec=mock_PIL_Image.Image)] * mock_constants.NUM_PREVIEW_FRAMES, # Video 1 has valid PILs
        [], # Video 2 (not selected)
        []  # Video 3 (not selected)
    ]

    # Mock PhotoImage creation
    mock_photo_image = mock.MagicMock(spec=tk.PhotoImage)
    mock_PhotoImage.return_value = mock_photo_image

    mock_verification_frame = mock_frames[3]
    mock_verification_frame.update_slot_preview.reset_mock()
    mock_verification_frame.enable_disable_slot_controls.reset_mock()
    mock_verification_frame.set_slot_approved.reset_mock() # Ensure approvals are not set here

    num_valid_previews = app_instance._create_previews_and_init_checkboxes(pil_images_list, num_selected)

    assert num_valid_previews == 2

    # Verify PhotoImage calls (for each valid PIL image)
    assert mock_PhotoImage.call_count == num_selected * mock_constants.NUM_PREVIEW_FRAMES

    # Verify update_slot_preview calls
    mock_verification_frame.update_slot_preview.assert_any_call(0, [mock_photo_image] * mock_constants.NUM_PREVIEW_FRAMES)
    mock_verification_frame.update_slot_preview.assert_any_call(1, [mock_photo_image] * mock_constants.NUM_PREVIEW_FRAMES)
    mock_verification_frame.update_slot_preview.assert_any_call(2, []) # No PILs for video 2
    mock_verification_frame.update_slot_preview.assert_any_call(3, []) # No PILs for video 3
    assert mock_verification_frame.update_slot_preview.call_count == mock_constants.MAX_VIDEOS

    # Verify enable_disable_slot_controls calls
    mock_verification_frame.enable_disable_slot_controls.assert_any_call(0, True)
    mock_verification_frame.enable_disable_slot_controls.assert_any_call(1, True)
    mock_verification_frame.enable_disable_slot_controls.assert_any_call(2, False)
    mock_verification_frame.enable_disable_slot_controls.assert_any_call(3, False)
    assert mock_verification_frame.enable_disable_slot_controls.call_count == mock_constants.MAX_VIDEOS

    # Verify set_slot_approved was NOT called (pre-marking happens later)
    mock_verification_frame.set_slot_approved.assert_not_called()

def test__create_previews_and_init_checkboxes_no_valid_previews(app_instance, mock_frames):
    """Test creating previews when no valid PIL images are provided."""
    num_selected = 2
    # Simulate no valid PIL images for any video
    pil_images_list = [
        [None] * mock_constants.NUM_PREVIEW_FRAMES, # Video 0 failed
        [None] * mock_constants.NUM_PREVIEW_FRAMES, # Video 1 failed
        [], # Video 2 (not selected)
        []  # Video 3 (not selected)
    ]

    mock_verification_frame = mock_frames[3]
    mock_verification_frame.update_slot_preview.reset_mock()
    mock_verification_frame.enable_disable_slot_controls.reset_mock()
    mock_verification_frame.set_slot_approved.reset_mock()

    num_valid_previews = app_instance._create_previews_and_init_checkboxes(pil_images_list, num_selected)

    assert num_valid_previews == 0

    # Verify update_slot_preview calls (should pass empty lists)
    for i in range(mock_constants.MAX_VIDEOS):
        mock_verification_frame.update_slot_preview.assert_any_call(i, [])
    assert mock_verification_frame.update_slot_preview.call_count == mock_constants.MAX_VIDEOS

    # Verify enable_disable_slot_controls calls (should all be False)
    for i in range(mock_constants.MAX_VIDEOS):
        mock_verification_frame.enable_disable_slot_controls.assert_any_call(i, False)
    assert mock_verification_frame.enable_disable_slot_controls.call_count == mock_constants.MAX_VIDEOS

    # Verify set_slot_approved was NOT called
    mock_verification_frame.set_slot_approved.assert_not_called()

def test__create_previews_and_init_checkboxes_unapproves_failed_preview(app_instance, mock_frames):
    """Test that a video is unapproved if its preview fails."""
    num_selected = 2
    filepaths = ["/fake/vid1.mp4", "/fake/vid2.mov"]
    app_instance.selected_file_paths_tuple = tuple(filepaths)
    app_instance.video_approval_states = [True, False, False, False] # Video 0 was somehow approved

    # Simulate Video 0 preview fails, Video 1 succeeds
    pil_images_list = [
        [None] * mock_constants.NUM_PREVIEW_FRAMES, # Video 0 failed
        [mock.Mock(spec=mock_PIL_Image.Image)] * mock_constants.NUM_PREVIEW_FRAMES, # Video 1 succeeded
        [], [],
    ]
    mock_photo_image = mock.MagicMock(spec=tk.PhotoImage)
    with mock.patch('CHDatasetManager.VideoPlacerV2.ImageTk.PhotoImage', return_value=mock_photo_image):
        mock_verification_frame = mock_frames[3]
        mock_verification_frame.update_slot_preview.reset_mock()
        mock_verification_frame.enable_disable_slot_controls.reset_mock()
        mock_verification_frame.set_slot_approved.reset_mock()

        num_valid_previews = app_instance._create_previews_and_init_checkboxes(pil_images_list, num_selected)

    assert num_valid_previews == 1
    assert app_instance.video_approval_states == [False, False, False, False] # Video 0 should be unapproved

    mock_verification_frame.enable_disable_slot_controls.assert_any_call(0, False) # Video 0 disabled
    mock_verification_frame.enable_disable_slot_controls.assert_any_call(1, True) # Video 1 enabled

    # Verify set_slot_approved was called to unapprove video 0
    mock_verification_frame.set_slot_approved.assert_any_call(0, False, is_enabled=False)
    # It should also be called for video 1 to set its state (False) and enable it
    mock_verification_frame.set_slot_approved.assert_any_call(1, False, is_enabled=True)
    assert mock_verification_frame.set_slot_approved.call_count == 2

# --- Tests for _apply_pre_marking_logic ---

@pytest.mark.parametrize("sd_factor, threshold, scores, expected_approvals", [
    # Basic case: scores above fixed threshold
    (0.5, 0.85, [0.9, 0.8, 0.95, 0.7], [True, False, True, False]),
    # SD factor matters: scores clustered high, SD threshold is high
    (0.5, 0.85, [0.95, 0.92, 0.91, 0.88], [True, True, True, True]), # SD threshold might be around 0.9
    # SD factor matters: scores clustered low, SD threshold is low, fixed threshold still applies
    (0.5, 0.85, [0.7, 0.75, 0.8, 0.82], [False, False, False, False]), # All below fixed threshold
    # SD factor matters: scores spread, SD threshold below fixed threshold
    (0.5, 0.85, [0.95, 0.6, 0.7, 0.86], [True, False, False, True]), # 0.95 and 0.86 >= 0.85
    # Edge case: only one valid score
    (0.5, 0.85, [0.9, None, None, None], [False, False, False, False]), # Pre-marking skipped
    # Edge case: scores are None
    (0.5, 0.85, [None, None, None, None], [False, False, False, False]),
    # Edge case: scores are all the same (SD=0)
    (0.5, 0.85, [0.9, 0.9, 0.9, 0.9], [True, True, True, True]), # SD threshold = 0.9, Fixed = 0.85
])
@mock.patch('CHDatasetManager.VideoPlacerV2.np.std')
def test__apply_pre_marking_logic(mock_np_std, sd_factor, threshold, scores, expected_approvals, app_instance, mock_frames):
    """Test the pre-marking logic based on scores and thresholds."""
    num_selected = len([s for s in scores if s is not None]) # Count non-None scores as potentially selected
    filepaths = [f"/fake/vid{i}.mp4" for i in range(num_selected)]
    valid_scores_list = [s for s in scores[:num_selected] if s is not None]
    max_score_val = max(valid_scores_list) if valid_scores_list else 0.0

    # Configure mock constants for this test
    with mock.patch('CHDatasetManager.VideoPlacerV2.PRE_MARKING_SD_FACTOR', sd_factor), \
         mock.patch('CHDatasetManager.VideoPlacerV2.PRE_MARKING_SCORE_THRESHOLD', threshold):

        # Mock np.std to return a predictable value or calculate it from valid_scores_list
        if len(valid_scores_list) >= 2:
             mock_np_std.return_value = np.std(valid_scores_list)
        else:
             mock_np_std.return_value = 0.0 # Should not be called if < 2 valid scores

        mock_verification_frame = mock_frames[3]
        mock_verification_frame.set_slot_approved.reset_mock()
        # Ensure checkboxes are enabled for selected videos
        for i in range(num_selected):
             mock_verification_frame.confirm_checkboxes[i].cget.return_value = 'normal'
        for i in range(num_selected, mock_constants.MAX_VIDEOS):
             mock_verification_frame.confirm_checkboxes[i].cget.return_value = 'disabled'

        # Initialize approval states to False before calling
        app_instance.video_approval_states = [False] * mock_constants.MAX_VIDEOS

        app_instance._apply_pre_marking_logic(scores, num_selected, filepaths, valid_scores_list, max_score_val)

        # Check the resulting approval states in the app instance
        assert app_instance.video_approval_states[:num_selected] == expected_approvals[:num_selected]

        # Check calls to set_slot_approved
        for i in range(num_selected):
            # set_slot_approved is called if the state changes to True AND the checkbox is enabled
            if expected_approvals[i] is True:
                 mock_verification_frame.set_slot_approved.assert_any_call(i, True, is_enabled=True)
            # If expected_approvals[i] is False, set_slot_approved is NOT called by this method
            # (it only sets True, clearing is done elsewhere)

        # Check initial_confirmation_state reflects the state after pre-marking
        assert app_instance.initial_confirmation_state[:num_selected] == app_instance.video_approval_states[:num_selected]

    if len(valid_scores_list) >= 2:
        mock_np_std.assert_called_once_with(valid_scores_list)
    else:
        mock_np_std.assert_not_called()

# --- Tests for _report_analysis_issues ---

@mock.patch('CHDatasetManager.VideoPlacerV2.messagebox.showwarning')
def test__report_analysis_issues_with_errors(mock_showwarning, app_instance):
    """Test reporting analysis issues when errors exist."""
    errors = ["Error 1", "Error 2"]
    app_instance._report_analysis_issues(errors)
    mock_showwarning.assert_called_once()
    assert "Analysis Issues" in mock_showwarning.call_args[0][0]
    assert "Error 1" in mock_showwarning.call_args[0][1]
    assert "Error 2" in mock_showwarning.call_args[0][1]

@mock.patch('CHDatasetManager.VideoPlacerV2.messagebox.showwarning')
def test__report_analysis_issues_no_errors(mock_showwarning, app_instance):
    """Test reporting analysis issues when no errors exist."""
    errors = []
    app_instance._report_analysis_issues(errors)
    mock_showwarning.assert_not_called()

# --- Tests for _finalize_analysis_ui_updates ---

def test__finalize_analysis_ui_updates_valid_previews(app_instance):
    """Test finalizing UI updates when valid previews were found."""
    app_instance.calculate_and_display_take_assignment = mock.Mock()
    app_instance.check_button_state.reset_mock()

    app_instance._finalize_analysis_ui_updates(num_videos_with_valid_previews=2)

    app_instance.calculate_and_display_take_assignment.assert_called_once()
    app_instance.check_button_state.assert_called_once()

def test__finalize_analysis_ui_updates_no_valid_previews(app_instance):
    """Test finalizing UI updates when no valid previews were found."""
    app_instance.calculate_and_display_take_assignment = mock.Mock()
    app_instance.check_button_state.reset_mock()
    app_instance.take_assignment_display.set("Old Takes")
    app_instance.status_message.set("Old Status")

    app_instance._finalize_analysis_ui_updates(num_videos_with_valid_previews=0)

    app_instance.calculate_and_display_take_assignment.assert_not_called()
    assert app_instance.take_assignment_display.get() == "Takes: Error"
    assert app_instance.status_message.get() == "Analysis failed for all videos. Cannot assign takes."
    app_instance.check_button_state.assert_called_once()

# --- Tests for check_analysis_queue ---

def test_check_analysis_queue_processes_message(app_instance, mock_master):
    """Test that check_analysis_queue processes a message from the queue."""
    mock_result = {'type': 'analysis_complete', 'data': 'some_data'}
    app_instance.analysis_queue.put(mock_result)
    app_instance._handle_analysis_completion = mock.Mock()

    # Call the method directly once
    app_instance.check_analysis_queue()

    app_instance._handle_analysis_completion.assert_called_once_with(mock_result)
    # The after method should be called to schedule the next check
    mock_master.after.assert_called_once_with(100, app_instance.check_analysis_queue)

def test_check_analysis_queue_empty_queue(app_instance, mock_master):
    """Test that check_analysis_queue does nothing when the queue is empty."""
    # Ensure queue is empty
    while not app_instance.analysis_queue.empty():
        app_instance.analysis_queue.get_nowait()

    app_instance._handle_analysis_completion = mock.Mock()

    # Call the method directly once
    app_instance.check_analysis_queue()

    app_instance._handle_analysis_completion.assert_not_called()
    # The after method should still be called to schedule the next check
    mock_master.after.assert_called_once_with(100, app_instance.check_analysis_queue)

@mock.patch('CHDatasetManager.VideoPlacerV2.messagebox.showerror')
def test_check_analysis_queue_exception(mock_showerror, app_instance, mock_master, caplog):
    """Test handling an unexpected exception in check_analysis_queue."""
    # Put a message that will cause an error (e.g., unexpected type)
    mock_result = {'type': 'unexpected_type'}
    app_instance.analysis_queue.put(mock_result)

    # Simulate an error in the processing logic (e.g., if _handle_analysis_completion wasn't mocked and failed)
    # Or, we can just raise an error directly in the check logic itself
    original_get_nowait = app_instance.analysis_queue.get_nowait
    def side_effect_get():
        # Get the item first, then raise error on next call or during processing
        item = original_get_nowait()
        raise Exception("Fake queue processing error")

    with mock.patch.object(app_instance.analysis_queue, 'get_nowait', side_effect=side_effect_get):
        # Call the method directly once
        app_instance.check_analysis_queue()

    # Check that the error was logged
    assert "Unexpected error in check_analysis_queue" in caplog.text
    # messagebox.showerror should NOT be called for internal queue errors, only critical startup ones
    mock_showerror.assert_not_called()
    # The after method should still be called to schedule the next check
    mock_master.after.assert_called_once_with(100, app_instance.check_analysis_queue)

# --- Tests for calculate_and_display_take_assignment ---

@mock.patch('CHDatasetManager.VideoPlacerV2.calculate_take_assignment_details')
def test_calculate_and_display_take_assignment_success(mock_calculate_details, app_instance):
    """Test successful take assignment calculation and display."""
    app_instance.base_directory.set("/fake/base")
    app_instance.selected_category.set("CatA")
    app_instance.selected_word.set("Word1")
    app_instance.selected_interpreter_id.set("001")
    app_instance.selected_file_paths_tuple = ("/fake/vid1.mp4", "/fake/vid2.mov")
    app_instance.status_message.set("Initial status")

    mock_assignment_info = {
        'start_take': 5, 'end_take': 6, 'is_full': False,
        'error_condition': False, 'message_short': "Takes: 5-6",
        'message_long': "Ready to process.", 'available_slots': 3
    }
    mock_calculate_details.return_value = mock_assignment_info

    app_instance.calculate_and_display_take_assignment()

    target_folder = os.path.join("/fake/base", "CatA", "Word1", "001")
    mock_calculate_details.assert_called_once_with(target_folder, "001", 2) # 2 selected files
    assert app_instance.take_assignment_display.get() == "Takes: 5-6"
    assert app_instance.status_message.get() == "Ready to process."

@mock.patch('CHDatasetManager.VideoPlacerV2.calculate_take_assignment_details')
def test_calculate_and_display_take_assignment_error(mock_calculate_details, app_instance):
    """Test take assignment calculation resulting in an error."""
    app_instance.base_directory.set("/fake/base")
    app_instance.selected_category.set("CatA")
    app_instance.selected_word.set("Word1")
    app_instance.selected_interpreter_id.set("001")
    app_instance.selected_file_paths_tuple = ("/fake/vid1.mp4", "/fake/vid2.mov", "/fake/vid3.mp4", "/fake/vid4.mov", "/fake/vid5.avi") # Too many files
    app_instance.status_message.set("Initial status")

    mock_assignment_info = {
        'start_take': 1, 'end_take': 5, 'is_full': False,
        'error_condition': True, 'message_short': "Takes: Error (Need 5",
        'message_long': "Error: Too many files selected (5) for available takes (4 starting from 1). Approve carefully.", 'available_slots': 4
    }
    mock_calculate_details.return_value = mock_assignment_info

    app_instance.calculate_and_display_take_assignment()

    target_folder = os.path.join("/fake/base", "CatA", "Word1", "001")
    mock_calculate_details.assert_called_once_with(target_folder, "001", 5) # 5 selected files
    assert app_instance.take_assignment_display.get() == "Takes: Error (Need 5"
    assert app_instance.status_message.get() == mock_assignment_info['message_long']

def test_calculate_and_display_take_assignment_missing_prerequisites(app_instance):
    """Test take assignment calculation when prerequisites are missing."""
    app_instance.base_directory.set("") # Missing base dir
    app_instance.selected_category.set("CatA")
    app_instance.selected_word.set("Word1")
    app_instance.selected_interpreter_id.set("001")
    app_instance.selected_file_paths_tuple = ("/fake/vid1.mp4",)
    app_instance.status_message.set("Initial status")
    mock_file_system_ops.calculate_take_assignment_details.reset_mock() # Reset mock from fixture

    app_instance.calculate_and_display_take_assignment()

    mock_file_system_ops.calculate_take_assignment_details.assert_not_called()
    assert app_instance.take_assignment_display.get() == "Takes: Error"
    assert app_instance.status_message.get() == "Error: Prerequisite selection missing for take calculation."

# --- Tests for check_button_state ---

def test_check_button_state_initial_setup_not_done(app_instance, mock_frames):
    """Test button states when initial setup is not complete."""
    app_instance.initial_setup_done = False
    app_instance.base_directory.set("") # Base dir not set
    mock_frames[0].update_widget_states.reset_mock()
    mock_frames[1].unbind_select_event.reset_mock()
    mock_frames[2].update_button_state.reset_mock()
    app_instance.process_button.config.reset_mock()

    app_instance.check_button_state()

    mock_frames[0].update_widget_states.assert_called_once_with(initial_setup_done=False, base_dir_set=False)
    mock_frames[1].unbind_select_event.assert_called_once()
    mock_frames[2].update_button_state.assert_called_once_with(word_selected=False)
    app_instance.process_button.config.assert_called_once_with(state='disabled')

def test_check_button_state_initial_setup_base_dir_set(app_instance, mock_frames):
    """Test button states when initial setup is not complete but base dir is set."""
    app_instance.initial_setup_done = False
    app_instance.base_directory.set("/fake/base") # Base dir is set
    mock_frames[0].update_widget_states.reset_mock()
    mock_frames[1].unbind_select_event.reset_mock()
    mock_frames[2].update_button_state.reset_mock()
    app_instance.process_button.config.reset_mock()

    app_instance.check_button_state()

    mock_frames[0].update_widget_states.assert_called_once_with(initial_setup_done=False, base_dir_set=True)
    mock_frames[1].unbind_select_event.assert_called_once()
    mock_frames[2].update_button_state.assert_called_once_with(word_selected=False)
    app_instance.process_button.config.assert_called_once_with(state='disabled')

def test_check_button_state_post_setup_no_word(app_instance, mock_frames):
    """Test button states after initial setup, but no word selected."""
    app_instance.initial_setup_done = True
    app_instance.base_directory.set("/fake/base")
    app_instance.selected_word.set("") # No word selected
    app_instance.selected_file_paths_tuple = ()
    app_instance.is_analysis_running = False
    mock_frames[0].update_widget_states.reset_mock()
    mock_frames[1].bind_select_event.reset_mock()
    mock_frames[2].update_button_state.reset_mock()
    app_instance.process_button.config.reset_mock()

    app_instance.check_button_state()

    mock_frames[0].update_widget_states.assert_called_once_with(initial_setup_done=True, base_dir_set=True)
    mock_frames[1].bind_select_event.assert_called_once_with(app_instance.on_tree_item_select)
    mock_frames[2].update_button_state.assert_called_once_with(word_selected=False)
    app_instance.process_button.config.assert_called_once_with(state='disabled')

def test_check_button_state_post_setup_word_selected_no_files(app_instance, mock_frames):
    """Test button states after initial setup, word selected, but no files selected."""
    app_instance.initial_setup_done = True
    app_instance.base_directory.set("/fake/base")
    app_instance.selected_word.set("Word1") # Word selected
    app_instance.selected_file_paths_tuple = () # No files selected
    app_instance.is_analysis_running = False
    mock_frames[0].update_widget_states.reset_mock()
    mock_frames[1].bind_select_event.reset_called_with() # Reset specific call check
    mock_frames[2].update_button_state.reset_mock()
    app_instance.process_button.config.reset_mock()

    app_instance.check_button_state()

    mock_frames[0].update_widget_states.assert_called_once_with(initial_setup_done=True, base_dir_set=True)
    mock_frames[1].bind_select_event.assert_called_once_with(app_instance.on_tree_item_select)
    mock_frames[2].update_button_state.assert_called_once_with(word_selected=True)
    app_instance.process_button.config.assert_called_once_with(state='disabled') # Cannot process without files

@pytest.mark.parametrize("take_display, approved_count, analysis_running, expected_process_state", [
    ("Takes: 1-2", 2, False, 'normal'), # Ready to process
    ("Takes: 1-1", 1, False, 'normal'), # Ready to process (single)
    ("Takes: FULL", 1, False, 'disabled'), # Full
    ("Takes: Error (Need", 1, False, 'disabled'), # Error in take assignment
    ("Takes: 1-2", 0, False, 'disabled'), # No files approved
    ("Takes: 1-2", 2, True, 'disabled'), # Analysis running
    ("Takes: -", 1, False, 'disabled'), # No valid take display
])
def test_check_button_state_post_setup_files_selected(
    take_display, approved_count, analysis_running, expected_process_state,
    app_instance, mock_frames
):
    """Test process button state based on various conditions."""
    app_instance.initial_setup_done = True
    app_instance.base_directory.set("/fake/base")
    app_instance.selected_word.set("Word1")
    app_instance.selected_file_paths_tuple = tuple([f"/fake/vid{i}.mp4" for i in range(approved_count + 1)]) # Assume enough files selected
    app_instance.take_assignment_display.set(take_display)
    app_instance.is_analysis_running = analysis_running

    # Set approval states
    app_instance.video_approval_states = [False] * mock_constants.MAX_VIDEOS
    for i in range(approved_count):
        app_instance.video_approval_states[i] = True

    mock_frames[0].update_widget_states.reset_mock()
    mock_frames[1].bind_select_event.reset_called_with()
    mock_frames[2].update_button_state.reset_mock()
    app_instance.process_button.config.reset_mock()

    app_instance.check_button_state()

    mock_frames[0].update_widget_states.assert_called_once_with(initial_setup_done=True, base_dir_set=True)
    mock_frames[1].bind_select_event.assert_called_once_with(app_instance.on_tree_item_select)
    mock_frames[2].update_button_state.assert_called_once_with(word_selected=True)
    app_instance.process_button.config.assert_called_once_with(state=expected_process_state)

# --- Tests for _trigger_csv_logging ---

@mock.patch('CHDatasetManager.VideoPlacerV2.log_verification_to_csv', return_value=True)
def test__trigger_csv_logging_success(mock_log_to_csv, app_instance):
    """Test successful CSV logging."""
    app_instance.base_directory.set("/fake/base")
    app_instance.selected_category.set("CatA")
    app_instance.selected_word.set("Word1")
    app_instance.selected_interpreter_id.set("001")
    app_instance.selected_file_paths_tuple = ("/fake/vid1.mp4", "/fake/vid2.mov")
    app_instance.per_video_similarity_scores = [0.9, 0.8, None, None]
    app_instance.initial_confirmation_state = [True, False, False, False] # Pre-marked state
    app_instance.video_approval_states = [True, True, False, False] # Final approved state

    processed_indices = [0, 1]
    assigned_take_numbers = [5, 6]

    app_instance._trigger_csv_logging(processed_indices, assigned_take_numbers)

    mock_log_to_csv.assert_called_once()
    log_entry = mock_log_to_csv.call_args[0][0] # Get the dictionary passed to log_verification_to_csv

    assert log_entry["BaseDirectory"] == "/fake/base"
    assert log_entry["Category"] == "CatA"
    assert log_entry["Word"] == "Word1"
    assert log_entry["InterpreterID"] == "001"
    assert log_entry["NumFilesSelected"] == 2
    assert log_entry["OriginalFileNames"] == "vid1.mp4; vid2.mov"
    assert log_entry["PerVideoScores"] == "0.9000; 0.8000"
    assert log_entry["PreMarkedConfirmation"] == "True; False"
    assert log_entry["FinalConfirmation"] == "True; True"
    assert log_entry["ProcessedIndices"] == "0; 1"
    assert log_entry["AssignedTakeNumbers"] == "5; 6"
    assert "Timestamp" in log_entry # Timestamp is added by the logging function itself

@mock.patch('CHDatasetManager.VideoPlacerV2.log_verification_to_csv', return_value=False)
@mock.patch('CHDatasetManager.VideoPlacerV2.messagebox.showerror')
def test__trigger_csv_logging_failure(mock_showerror, mock_log_to_csv, app_instance):
    """Test CSV logging failure."""
    app_instance.base_directory.set("/fake/base")
    app_instance.selected_category.set("CatA")
    app_instance.selected_word.set("Word1")
    app_instance.selected_interpreter_id.set("001")
    app_instance.selected_file_paths_tuple = ("/fake/vid1.mp4",)
    app_instance.per_video_similarity_scores = [0.9, None, None, None]
    app_instance.initial_confirmation_state = [True, False, False, False]
    app_instance.video_approval_states = [True, False, False, False]
    app_instance.status_message.set("Initial status")

    processed_indices = [0]
    assigned_take_numbers = [5]

    app_instance._trigger_csv_logging(processed_indices, assigned_take_numbers)

    mock_log_to_csv.assert_called_once()
    mock_showerror.assert_called_once()
    assert "Logging Error" in mock_showerror.call_args[0][0]
    assert "Failed to write verification data" in mock_showerror.call_args[0][1]
    assert app_instance.status_message.get() == "Error logging data (check log file and console)."

# --- Tests for _validate_processing_prerequisites ---

@mock.patch('CHDatasetManager.VideoPlacerV2.messagebox.showwarning')
def test__validate_processing_prerequisites_analysis_running(mock_showwarning, app_instance):
    """Test validation fails if analysis is running."""
    app_instance.is_analysis_running = True
    app_instance.check_button_state.reset_mock()

    is_valid = app_instance._validate_processing_prerequisites()

    assert is_valid is False
    mock_showwarning.assert_called_once()
    assert "Analysis is in progress" in mock_showwarning.call_args[0][1]
    app_instance.check_button_state.assert_called_once()

@mock.patch('CHDatasetManager.VideoPlacerV2.messagebox.showerror')
def test__validate_processing_prerequisites_no_files_selected(mock_showerror, app_instance):
    """Test validation fails if no files are selected."""
    app_instance.is_analysis_running = False
    app_instance.selected_file_paths_tuple = () # No files
    app_instance.check_button_state.reset_mock()

    is_valid = app_instance._validate_processing_prerequisites()

    assert is_valid is False
    mock_showerror.assert_called_once()
    assert "No video files selected" in mock_showerror.call_args[0][1]
    app_instance.check_button_state.assert_called_once()

@mock.patch('CHDatasetManager.VideoPlacerV2.messagebox.showerror')
def test__validate_processing_prerequisites_no_videos_approved(mock_showerror, app_instance):
    """Test validation fails if files are selected but none are approved."""
    app_instance.is_analysis_running = False
    app_instance.selected_file_paths_tuple = ("/fake/vid1.mp4", "/fake/vid2.mov")
    app_instance.video_approval_states = [False, False, False, False] # None approved
    app_instance.check_button_state.reset_mock()

    is_valid = app_instance._validate_processing_prerequisites()

    assert is_valid is False
    mock_showerror.assert_called_once()
    assert "Please approve at least one video" in mock_showerror.call_args[0][1]
    app_instance.check_button_state.assert_called_once()

@mock.patch('CHDatasetManager.VideoPlacerV2.messagebox.showerror')
def test__validate_processing_prerequisites_missing_config(mock_showerror, app_instance):
    """Test validation fails if configuration is missing."""
    app_instance.is_analysis_running = False
    app_instance.selected_file_paths_tuple = ("/fake/vid1.mp4",)
    app_instance.video_approval_states = [True, False, False, False]
    app_instance.base_directory.set("") # Missing base dir
    app_instance.selected_category.set("CatA")
    app_instance.selected_word.set("Word1")
    app_instance.selected_interpreter_id.set("001")
    app_instance.check_button_state.reset_mock()

    is_valid = app_instance._validate_processing_prerequisites()

    assert is_valid is False
    mock_showerror.assert_called_once()
    assert "Base directory, Category, Word, or Interpreter ID is missing" in mock_showerror.call_args[0][1]
    app_instance.check_button_state.assert_called_once()

def test__validate_processing_prerequisites_success(app_instance):
    """Test successful validation of prerequisites."""
    app_instance.is_analysis_running = False
    app_instance.selected_file_paths_tuple = ("/fake/vid1.mp4", "/fake/vid2.mov")
    app_instance.video_approval_states = [True, False, True, False] # vid1 and vid3 approved (indices 0 and 2)
    app_instance.base_directory.set("/fake/base")
    app_instance.selected_category.set("CatA")
    app_instance.selected_word.set("Word1")
    app_instance.selected_interpreter_id.set("001")
    app_instance.check_button_state.reset_mock()

    is_valid = app_instance._validate_processing_prerequisites()

    assert is_valid is True
    assert app_instance.confirmed_indices_for_processing_ == [0, 2]
    assert app_instance.target_folder_path_for_processing_ == os.path.join("/fake/base", "CatA", "Word1", "001")
    assert app_instance.interpreter_id_for_processing_ == "001"
    app_instance.check_button_state.assert_not_called() # Should not be called on success

# --- Tests for _verify_and_calculate_takes_for_processing ---

@mock.patch('CHDatasetManager.VideoPlacerV2.calculate_take_assignment_details')
def test__verify_and_calculate_takes_for_processing_success(mock_calculate_details, app_instance):
    """Test successful take verification before processing."""
    app_instance.confirmed_indices_for_processing_ = [0, 2] # 2 approved files
    app_instance.target_folder_path_for_processing_ = "/fake/target"
    app_instance.interpreter_id_for_processing_ = "001"

    mock_assignment_info = {
        'start_take': 5, 'end_take': 6, 'is_full': False,
        'error_condition': False, 'message_short': "Takes: 5-6",
        'message_long': "Ready to process.", 'available_slots': 3
    }
    mock_calculate_details.return_value = mock_assignment_info

    start_take = app_instance._verify_and_calculate_takes_for_processing()

    mock_calculate_details.assert_called_once_with("/fake/target", "001", 2) # Pass count of approved files
    assert start_take == 5
    assert app_instance.start_take_for_processing_ == 5

@mock.patch('CHDatasetManager.VideoPlacerV2.calculate_take_assignment_details')
@mock.patch('CHDatasetManager.VideoPlacerV2.messagebox.showerror')
def test__verify_and_calculate_takes_for_processing_error(mock_showerror, mock_calculate_details, app_instance):
    """Test take verification fails before processing."""
    app_instance.confirmed_indices_for_processing_ = [0, 1, 2, 3] # 4 approved files
    app_instance.target_folder_path_for_processing_ = "/fake/target"
    app_instance.interpreter_id_for_processing_ = "001"

    mock_assignment_info = {
        'start_take': 3, 'end_take': 6, 'is_full': False,
        'error_condition': True, 'message_short': "Takes: Error (Need 4",
        'message_long': "Error: Too many files selected (4) for available takes (2 starting from 3). Approve carefully.", 'available_slots': 2
    }
    mock_calculate_details.return_value = mock_assignment_info

    start_take = app_instance._verify_and_calculate_takes_for_processing()

    mock_calculate_details.assert_called_once_with("/fake/target", "001", 4)
    assert start_take is None
    mock_showerror.assert_called_once()
    assert "Take Assignment Error" in mock_showerror.call_args[0][0]
    assert mock_showerror.call_args[0][1].startswith("Cannot process due to take assignment issues:")

# --- Tests for _execute_file_operations ---

@mock.patch('CHDatasetManager.VideoPlacerV2.execute_video_processing_fs', return_value=(["/fake/vid1.mp4", "/fake/vid3.avi"], []))
def test__execute_file_operations_success(mock_execute_fs, app_instance):
    """Test successful file operations."""
    app_instance.selected_file_paths_tuple = ("/fake/vid1.mp4", "/fake/vid2.mov", "/fake/vid3.avi")
    app_instance.confirmed_indices_for_processing_ = [0, 2]
    app_instance.target_folder_path_for_processing_ = "/fake/target"
    app_instance.interpreter_id_for_processing_ = "001"
    app_instance.start_take_for_processing_ = 5
    app_instance.status_message.set("Initial status")
    app_instance.process_button.config.reset_mock()
    app_instance.master.update_idletasks = mock.Mock()

    successfully_moved_paths, errors, num_approved_initially = app_instance._execute_file_operations()

    assert successfully_moved_paths == ["/fake/vid1.mp4", "/fake/vid3.avi"]
    assert not errors
    assert num_approved_initially == 2
    assert app_instance.status_message.get() == "Processing 2 approved file(s)..."
    app_instance.process_button.config.assert_called_once_with(state='disabled')
    app_instance.master.update_idletasks.assert_called_once()
    mock_execute_fs.assert_called_once_with(
        app_instance.selected_file_paths_tuple,
        app_instance.confirmed_indices_for_processing_,
        app_instance.target_folder_path_for_processing_,
        app_instance.interpreter_id_for_processing_,
        app_instance.start_take_for_processing_
    )
    # Check globally processed set
    assert "/fake/vid1.mp4" in app_instance.globally_processed_video_paths
    assert "/fake/vid3.avi" in app_instance.globally_processed_video_paths
    assert "/fake/vid2.mov" not in app_instance.globally_processed_video_paths

@mock.patch('CHDatasetManager.VideoPlacerV2.execute_video_processing_fs', return_value=(["/fake/vid1.mp4"], ["Error moving vid2"]))
def test__execute_file_operations_with_errors(mock_execute_fs, app_instance):
    """Test file operations with errors."""
    app_instance.selected_file_paths_tuple = ("/fake/vid1.mp4", "/fake/vid2.mov")
    app_instance.confirmed_indices_for_processing_ = [0, 1]
    app_instance.target_folder_path_for_processing_ = "/fake/target"
    app_instance.interpreter_id_for_processing_ = "001"
    app_instance.start_take_for_processing_ = 5
    app_instance.status_message.set("Initial status")
    app_instance.process_button.config.reset_mock()
    app_instance.master.update_idletasks = mock.Mock()

    successfully_moved_paths, errors, num_approved_initially = app_instance._execute_file_operations()

    assert successfully_moved_paths == ["/fake/vid1.mp4"]
    assert errors == ["Error moving vid2"]
    assert num_approved_initially == 2
    assert app_instance.status_message.get() == "Processing 2 approved file(s)..."
    app_instance.process_button.config.assert_called_once_with(state='disabled')
    app_instance.master.update_idletasks.assert_called_once()
    mock_execute_fs.assert_called_once()
    # Check globally processed set
    assert "/fake/vid1.mp4" in app_instance.globally_processed_video_paths
    assert "/fake/vid2.mov" not in app_instance.globally_processed_video_paths

# --- Tests for _finalize_processing_and_reset_ui ---

@mock.patch('CHDatasetManager.VideoPlacerV2.messagebox.showinfo')
def test__finalize_processing_and_reset_ui_success(mock_showinfo, app_instance, mock_frames):
    """Test finalizing UI after successful processing."""
    app_instance.selected_file_paths_tuple = ("/fake/vid1.mp4", "/fake/vid2.mov") # Simulate state before reset
    app_instance.selected_files_info.set("2 files selected")
    app_instance.take_assignment_display.set("Takes: 5-6")
    app_instance.status_message.set("Processing...")
    app_instance.clear_analysis_results = mock.Mock()
    app_instance.selected_word.set("Word1") # Simulate word selected before reset
    app_instance.category_word_tree_frame.clear_selection = mock.Mock()
    app_instance.check_button_state.reset_mock()

    app_instance._finalize_processing_and_reset_ui(success_count=2, errors=[], num_approved_initially=2)

    mock_showinfo.assert_called_once()
    assert "Processed 2/2 approved files." in mock_showinfo.call_args[0][1]
    assert app_instance.status_message.get() == "Processing complete."

    # Verify UI reset
    assert app_instance.selected_file_paths_tuple == ()
    assert app_instance.selected_files_info.get() == "No files selected"
    app_instance.clear_analysis_results.assert_called_once()
    assert app_instance.selected_word.get() == "" # Should be cleared by clear_selection triggering on_tree_item_select
    app_instance.category_word_tree_frame.clear_selection.assert_called_once()
    assert app_instance.take_assignment_display.get() == "Takes: -"
    # Status message is set by on_tree_item_select after clear_selection
    # assert app_instance.status_message.get() == "Select a Word from the tree."

    app_instance.check_button_state.assert_called_once()

@mock.patch('CHDatasetManager.VideoPlacerV2.messagebox.showwarning')
def test__finalize_processing_and_reset_ui_with_errors(mock_showwarning, app_instance, mock_frames):
    """Test finalizing UI after processing with errors."""
    app_instance.selected_file_paths_tuple = ("/fake/vid1.mp4", "/fake/vid2.mov")
    app_instance.selected_files_info.set("2 files selected")
    app_instance.take_assignment_display.set("Takes: 5-6")
    app_instance.status_message.set("Processing...")
    app_instance.clear_analysis_results = mock.Mock()
    app_instance.selected_word.set("Word1")
    app_instance.category_word_tree_frame.clear_selection = mock.Mock()
    app_instance.check_button_state.reset_mock()

    errors = ["Error 1", "Error 2"]
    app_instance._finalize_processing_and_reset_ui(success_count=1, errors=errors, num_approved_initially=2)

    mock_showwarning.assert_called_once()
    assert "Processing Complete with Errors" in mock_showwarning.call_args[0][0]
    assert "Processed 1/2 approved files." in mock_showwarning.call_args[0][1]
    assert "Error 1" in mock_showwarning.call_args[0][1]
    assert "Error 2" in mock_showwarning.call_args[0][1]
    assert app_instance.status_message.get() == "Processing finished with errors."

    # Verify UI reset (same as success case)
    assert app_instance.selected_file_paths_tuple == ()
    assert app_instance.selected_files_info.get() == "No files selected"
    app_instance.clear_analysis_results.assert_called_once()
    assert app_instance.selected_word.get() == ""
    app_instance.category_word_tree_frame.clear_selection.assert_called_once()
    assert app_instance.take_assignment_display.get() == "Takes: -"

    app_instance.check_button_state.assert_called_once()

# --- Tests for process_selected_videos ---

@mock.patch.object(VideoPlacerApp, '_validate_processing_prerequisites', return_value=True)
@mock.patch.object(VideoPlacerApp, '_verify_and_calculate_takes_for_processing', return_value=5) # Simulate start take 5
@mock.patch.object(VideoPlacerApp, '_trigger_csv_logging')
@mock.patch.object(VideoPlacerApp, '_execute_file_operations', return_value=(["/fake/vid1.mp4"], [], 1)) # 1 success, 0 errors, 1 approved
@mock.patch.object(VideoPlacerApp, '_finalize_processing_and_reset_ui')
def test_process_selected_videos_success(
    mock_finalize, mock_execute, mock_log, mock_verify_takes, mock_validate,
    app_instance
):
    """Test the main process flow when all steps succeed."""
    # Set minimal state needed for validation to pass
    app_instance.selected_file_paths_tuple = ("/fake/vid1.mp4",)
    app_instance.video_approval_states = [True, False, False, False]
    app_instance.base_directory.set("/fake/base")
    app_instance.selected_category.set("CatA")
    app_instance.selected_word.set("Word1")
    app_instance.selected_interpreter_id.set("001")

    app_instance.process_selected_videos()

    mock_validate.assert_called_once()
    mock_verify_takes.assert_called_once()
    mock_log.assert_called_once_with([0], [5]) # Assuming index 0 was approved, starting take 5
    mock_execute.assert_called_once()
    mock_finalize.assert_called_once_with(["/fake/vid1.mp4"], [], 1)
    # Check that temporary attributes are deleted
    assert not hasattr(app_instance, 'confirmed_indices_for_processing_')
    assert not hasattr(app_instance, 'target_folder_path_for_processing_')
    assert not hasattr(app_instance, 'interpreter_id_for_processing_')
    assert not hasattr(app_instance, 'start_take_for_processing_')
    app_instance.check_button_state.assert_not_called() # Called by finalize

@mock.patch.object(VideoPlacerApp, '_validate_processing_prerequisites', return_value=False)
@mock.patch.object(VideoPlacerApp, '_verify_and_calculate_takes_for_processing')
@mock.patch.object(VideoPlacerApp, '_trigger_csv_logging')
@mock.patch.object(VideoPlacerApp, '_execute_file_operations')
@mock.patch.object(VideoPlacerApp, '_finalize_processing_and_reset_ui')
def test_process_selected_videos_validation_fails(
    mock_finalize, mock_execute, mock_log, mock_verify_takes, mock_validate,
    app_instance
):
    """Test the main process flow when validation fails."""
    app_instance.process_selected_videos()

    mock_validate.assert_called_once()
    mock_verify_takes.assert_not_called()
    mock_log.assert_not_called()
    mock_execute.assert_not_called()
    mock_finalize.assert_not_called()
    app_instance.check_button_state.assert_called_once() # Should call check_button_state on failure

@mock.patch.object(VideoPlacerApp, '_validate_processing_prerequisites', return_value=True)
@mock.patch.object(VideoPlacerApp, '_verify_and_calculate_takes_for_processing', return_value=None) # Simulate take verification fails
@mock.patch.object(VideoPlacerApp, '_trigger_csv_logging')
@mock.patch.object(VideoPlacerApp, '_execute_file_operations')
@mock.patch.object(VideoPlacerApp, '_finalize_processing_and_reset_ui')
def test_process_selected_videos_take_verification_fails(
    mock_finalize, mock_execute, mock_log, mock_verify_takes, mock_validate,
    app_instance
):
    """Test the main process flow when take verification fails."""
    # Set minimal state needed for validation to pass
    app_instance.selected_file_paths_tuple = ("/fake/vid1.mp4",)
    app_instance.video_approval_states = [True, False, False, False]
    app_instance.base_directory.set("/fake/base")
    app_instance.selected_category.set("CatA")
    app_instance.selected_word.set("Word1")
    app_instance.selected_interpreter_id.set("001")

    app_instance.process_selected_videos()

    mock_validate.assert_called_once()
    mock_verify_takes.assert_called_once()
    mock_log.assert_not_called()
    mock_execute.assert_not_called()
    mock_finalize.assert_not_called()
    app_instance.check_button_state.assert_called_once() # Should call check_button_state on failure

# --- Tests for _handle_preview_click_app_level ---

def test__handle_preview_click_app_level_enabled_checkbox(app_instance, mock_frames):
    """Test clicking preview toggles state when checkbox is enabled."""
    slot_index = 1
    mock_verification_frame = mock_frames[3]
    mock_verification_frame.confirm_checkboxes[slot_index].cget.return_value = 'normal' # Checkbox is enabled
    app_instance.video_approval_states[slot_index] = False # Initial state is False
    app_instance.check_button_state.reset_mock()

    app_instance._handle_preview_click_app_level(slot_index)

    assert app_instance.video_approval_states[slot_index] is True # State should be toggled
    mock_verification_frame.set_slot_approved.assert_called_once_with(slot_index, True, is_enabled=True)
    app_instance.check_button_state.assert_called_once()

def test__handle_preview_click_app_level_enabled_checkbox_already_true(app_instance, mock_frames):
    """Test clicking preview toggles state from True to False when checkbox is enabled."""
    slot_index = 1
    mock_verification_frame = mock_frames[3]
    mock_verification_frame.confirm_checkboxes[slot_index].cget.return_value = 'normal' # Checkbox is enabled
    app_instance.video_approval_states[slot_index] = True # Initial state is True
    app_instance.check_button_state.reset_mock()

    app_instance._handle_preview_click_app_level(slot_index)

    assert app_instance.video_approval_states[slot_index] is False # State should be toggled
    mock_verification_frame.set_slot_approved.assert_called_once_with(slot_index, False, is_enabled=True)
    app_instance.check_button_state.assert_called_once()

def test__handle_preview_click_app_level_disabled_checkbox(app_instance, mock_frames):
    """Test clicking preview does nothing when checkbox is disabled."""
    slot_index = 1
    mock_verification_frame = mock_frames[3]
    mock_verification_frame.confirm_checkboxes[slot_index].cget.return_value = 'disabled' # Checkbox is disabled
    app_instance.video_approval_states[slot_index] = False # Initial state
    app_instance.check_button_state.reset_mock()

    app_instance._handle_preview_click_app_level(slot_index)

    assert app_instance.video_approval_states[slot_index] is False # State should NOT be toggled
    mock_verification_frame.set_slot_approved.assert_not_called()
    app_instance.check_button_state.assert_not_called()

def test__handle_preview_click_app_level_invalid_index(app_instance, caplog):
    """Test clicking preview with an invalid index."""
    slot_index = 99 # Invalid index
    app_instance.check_button_state.reset_mock()

    app_instance._handle_preview_click_app_level(slot_index)

    assert "Invalid slot_index 99" in caplog.text
    app_instance.check_button_state.assert_not_called()

# --- Tests for _handle_checkbox_change_app_level ---

def test__handle_checkbox_change_app_level_true(app_instance, mock_frames):
    """Test checkbox changing to True."""
    slot_index = 2
    mock_verification_frame = mock_frames[3]
    mock_verification_frame.get_slot_approved_state.return_value = True # Checkbox is now True
    app_instance.video_approval_states[slot_index] = False # Initial state
    app_instance.check_button_state.reset_mock()

    app_instance._handle_checkbox_change_app_level(slot_index)

    assert app_instance.video_approval_states[slot_index] is True # App state updated
    mock_verification_frame.get_slot_approved_state.assert_called_once_with(slot_index)
    app_instance.check_button_state.assert_called_once()

def test__handle_checkbox_change_app_level_false(app_instance, mock_frames):
    """Test checkbox changing to False."""
    slot_index = 2
    mock_verification_frame = mock_frames[3]
    mock_verification_frame.get_slot_approved_state.return_value = False # Checkbox is now False
    app_instance.video_approval_states[slot_index] = True # Initial state
    app_instance.check_button_state.reset_mock()

    app_instance._handle_checkbox_change_app_level(slot_index)

    assert app_instance.video_approval_states[slot_index] is False # App state updated
    mock_verification_frame.get_slot_approved_state.assert_called_once_with(slot_index)
    app_instance.check_button_state.assert_called_once()

def test__handle_checkbox_change_app_level_invalid_index(app_instance):
    """Test checkbox change with an invalid index."""
    slot_index = 99 # Invalid index
    app_instance.check_button_state.reset_mock()
    mock_file_system_ops.log_verification_to_csv.reset_mock() # Ensure no side effects

    app_instance._handle_checkbox_change_app_level(slot_index)

    # No error message expected, just return early
    app_instance.check_button_state.assert_not_called()

# --- Tests for on_closing ---

def test_on_closing(app_instance, mock_master, mock_frames):
    """Test the application closing sequence."""
    mock_verification_frame = mock_frames[3]
    mock_master.destroy.reset_mock()

    app_instance.on_closing()

    mock_verification_frame.on_app_closing.assert_called_once()
    mock_master.destroy.assert_called_once()

# --- Tests for clear_analysis_results ---

def test_clear_analysis_results(app_instance, mock_frames):
    """Test clearing analysis results and UI."""
    # Set some non-default states
    app_instance.per_video_similarity_scores = [0.9, 0.8, None, None]
    app_instance.video_approval_states = [True, False, False, False]
    app_instance.take_assignment_display.set("Takes: 1-1")
    app_instance.initial_confirmation_state = [True, False, False, False]

    mock_verification_frame = mock_frames[3]
    mock_verification_frame.clear_all_slots.reset_mock()

    app_instance.clear_analysis_results()

    mock_verification_frame.clear_all_slots.assert_called_once()
    assert app_instance.per_video_similarity_scores == [None] * mock_constants.MAX_VIDEOS
    assert app_instance.video_approval_states == [False] * mock_constants.MAX_VIDEOS
    assert app_instance.take_assignment_display.get() == "Takes: -"
    assert app_instance.initial_confirmation_state == [False] * mock_constants.MAX_VIDEOS

# --- Test for the logging decorator (optional, can be done separately) ---
# Testing decorators is often done by applying them to dummy functions
# and checking if the logger mock is called correctly.

def test_log_method_call_decorator(app_instance, caplog):
    """Test the log_method_call decorator."""
    # Use the actual decorator imported for testing
    decorator_to_test = actual_log_method_call
    if decorator_to_test is None:
        # Fallback or skip if not imported (e.g. if it's in VideoPlacerV2.py and not imported separately)
        # from CHDatasetManager.VideoPlacerV2 import log_method_call as decorator_to_test # Alternative
        pytest.skip("Actual log_method_call decorator not available for testing.")

    # Create a dummy method on the app instance
    @decorator_to_test(level=logging.INFO, log_args=True, log_return=True)
    def dummy_method(self, arg1, kwarg1="default"):
        return f"Processed {arg1} and {kwarg1}"

    # Temporarily add the method to the app instance
    app_instance.dummy_method = dummy_method.__get__(app_instance, VideoPlacerApp)

    # Call the decorated method
    result = app_instance.dummy_method("value1", kwarg1="value2")

    assert result == "Processed value1 and value2"

    # Check log messages
    # caplog.text captures logs from all loggers
    assert "CALLING: VideoPlacerApp.dummy_method('value1', kwarg1='value2')" in caplog.text
    assert "RETURNED: VideoPlacerApp.dummy_method -> 'Processed value1 and value2'" in caplog.text

def test_log_method_call_decorator_exception(app_instance, caplog):
    """Test the log_method_call decorator with an exception."""
    decorator_to_test = actual_log_method_call
    if decorator_to_test is None:
        # from CHDatasetManager.VideoPlacerV2 import log_method_call as decorator_to_test # Alternative
        pytest.skip("Actual log_method_call decorator not available for testing.")

    @decorator_to_test(log_exception=True)
    def dummy_method_raises(self):
        raise ValueError("Something went wrong")

    app_instance.dummy_method_raises = dummy_method_raises.__get__(app_instance, VideoPlacerApp)

    with pytest.raises(ValueError, match="Something went wrong"):
        app_instance.dummy_method_raises()

    # Check log messages
    assert "CALLING: VideoPlacerApp.dummy_method_raises()" in caplog.text
    assert "EXCEPTION in VideoPlacerApp.dummy_method_raises: ValueError('Something went wrong')" in caplog.text

# Note: Testing the __main__ block requires more advanced techniques like
# patching sys.exit and potentially mocking the Tkinter mainloop.
# Often, the logic in __main__ is minimal (just creating the app and running mainloop)
# and less critical to unit test compared to the core application logic.