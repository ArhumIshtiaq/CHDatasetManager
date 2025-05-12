import pytest
import os
import shutil
import datetime
import sys
from unittest import mock
# pyfakefs.fake_filesystem_unittest.TestCase is not needed if using pytest fixture
# from pyfakefs.fake_filesystem_unittest import TestCase
import csv

# Assuming your project structure allows this import.
from CHDatasetManager.file_system_operations import (
    get_app_base_path,
    get_directory_structure,
    determine_next_take_number,
    move_and_rename_video,
    log_verification_to_csv,
    calculate_take_assignment_details,
    execute_video_processing_fs
)
from CHDatasetManager.constants import (
    MAX_TAKES_PER_WORD_INTERPRETER, VERIFICATION_LOG_FILE
)

# --- Fixture for pyfakefs --- 
# The pytest-pyfakefs plugin (if installed) provides the 'fs' fixture automatically.
# A custom 'fs' fixture like the one previously here can conflict with the plugin's fixture
# or be insufficient if the plugin is not active/installed and manual patching is needed.
# For plugin usage, this custom fixture should be removed.


# --- Tests for get_app_base_path ---

@mock.patch('CHDatasetManager.file_system_operations.sys')
@mock.patch('CHDatasetManager.file_system_operations.os.path.abspath')
def test_get_app_base_path_not_frozen(mock_abspath, mock_sys_in_module):
    # Configure the mock_sys object as seen by file_system_operations
    # To simulate 'frozen' not being there or being False:
    # We need to ensure that when getattr(sys, 'frozen', False) is called, it returns False.
    # A simple way is to ensure 'frozen' is not an attribute, or set it to False.
    # If 'frozen' is not set on mock_sys_in_module, getattr with default will work.
    # For clarity, let's explicitly set it.
    mock_sys_in_module.frozen = False
    mock_sys_in_module.argv = ['/fake/path/to/script.py']
    mock_abspath.return_value = '/fake/path/to/script.py'
    base_path = get_app_base_path()
    assert base_path == '/fake/path/to'
    mock_abspath.assert_called_once_with('/fake/path/to/script.py')

@mock.patch('CHDatasetManager.file_system_operations.sys')
@mock.patch('CHDatasetManager.file_system_operations.os.path.dirname') # Mock dirname for frozen path
def test_get_app_base_path_frozen(mock_dirname, mock_sys_in_module):
    mock_sys_in_module.frozen = True
    mock_sys_in_module.executable = '/fake/path/to/frozen_app.exe'
    # Simulate _MEIPASS attribute for PyInstaller
    # To make hasattr(mock_sys_in_module, '_MEIPASS') return True and allow access
    # We need to ensure that hasattr(mock_sys_in_module, '_MEIPASS') returns True
    # and that mock_sys_in_module._MEIPASS can be accessed.
    # Setting it directly on the mock should suffice for both.
    mock_sys_in_module._MEIPASS = '/tmp/_MEIxxxxxx' # For access
    # To ensure hasattr works as expected if the attribute might not always be present on a real sys:
    mock_sys_in_module.has_attr = lambda name: name == '_MEIPASS' # Simplistic mock for hasattr

    mock_dirname.return_value = "/fake/path/to"
    base_path = get_app_base_path()
    assert base_path == '/fake/path/to'
    mock_dirname.assert_called_once_with('/fake/path/to/frozen_app.exe')


# --- Tests for get_directory_structure ---

def test_get_directory_structure_empty_base_dir(fs):
    """Test scanning an empty base directory."""
    base_dir = "/fake/base"
    fs.create_dir(base_dir)
    structure = get_directory_structure(base_dir)
    assert structure == {}

def test_get_directory_structure_with_categories_and_words(fs):
    """Test scanning a directory with categories and words."""
    base_dir = "/fake/base"
    fs.create_dir(os.path.join(base_dir, "CategoryA", "Word1"))
    fs.create_dir(os.path.join(base_dir, "CategoryA", "Word2"))
    fs.create_dir(os.path.join(base_dir, "CategoryB", "WordX"))
    fs.create_file(os.path.join(base_dir, "CategoryA", "file.txt")) # Should be ignored
    # fs.create_file(os.path.join(base_dir, "CategoryC", "file.txt")) # CategoryC should not appear as it has no subdirs
    fs.create_dir(os.path.join(base_dir, "CategoryC")) # CategoryC should appear but with empty word list

    structure = get_directory_structure(base_dir)

    assert "CategoryA" in structure
    assert "CategoryB" in structure
    assert "CategoryC" in structure
    assert sorted(structure["CategoryA"]) == ["Word1", "Word2"]
    assert sorted(structure["CategoryB"]) == ["WordX"]
    assert structure["CategoryC"] == []
    assert len(structure) == 3

def test_get_directory_structure_invalid_base_dir(fs, caplog):
    """Test scanning a non-existent or invalid base directory."""
    base_dir = "/fake/non_existent"
    structure = get_directory_structure(base_dir)
    assert structure == {}
    assert "Invalid base directory for structure scan" in caplog.text

    structure_none = get_directory_structure(None) # Test with None explicitly
    assert structure_none == {}
    assert "Invalid base directory for structure scan" in caplog.text # Check log for None case too

def test_get_directory_structure_permission_error(fs, caplog):
    """Test handling permission errors during scan."""
    base_dir = "/fake/base"
    fs.create_dir(base_dir)
    with mock.patch('CHDatasetManager.file_system_operations.os.listdir', side_effect=PermissionError("Fake permission error")):
         structure = get_directory_structure(base_dir)
         assert structure == {}
         assert "Error scanning directory structure" in caplog.text


# --- Tests for determine_next_take_number ---

def test_determine_next_take_number_empty_dir(fs):
    target_dir = "/fake/path/interpreter_id"
    fs.create_dir(target_dir)
    interpreter_id = "001"
    next_take = determine_next_take_number(target_dir, interpreter_id)
    assert next_take == 1

def test_determine_next_take_number_existing_takes(fs):
    target_dir = "/fake/path/interpreter_id"
    fs.create_dir(target_dir)
    fs.create_file(os.path.join(target_dir, "001_1.mp4"))
    fs.create_file(os.path.join(target_dir, "001_2.mov"))
    fs.create_file(os.path.join(target_dir, "unrelated_file.txt"))
    interpreter_id = "001"
    next_take = determine_next_take_number(target_dir, interpreter_id)
    assert next_take == 3

def test_determine_next_take_number_all_takes_exist(fs):
    target_dir = "/fake/path/interpreter_id"
    fs.create_dir(target_dir)
    for i in range(MAX_TAKES_PER_WORD_INTERPRETER):
        fs.create_file(os.path.join(target_dir, f"001_{i+1}.mp4"))
    interpreter_id = "001"
    next_take = determine_next_take_number(target_dir, interpreter_id)
    assert next_take == MAX_TAKES_PER_WORD_INTERPRETER + 1

def test_determine_next_take_number_dir_not_exist(fs):
    target_dir = "/fake/non_existent_path/interpreter_id"
    interpreter_id = "001"
    next_take = determine_next_take_number(target_dir, interpreter_id)
    assert next_take == 1

def test_determine_next_take_number_non_integer_take(fs, caplog):
    target_dir = "/fake/path/interpreter_id"
    fs.create_dir(target_dir)
    fs.create_file(os.path.join(target_dir, "001_abc.mp4"))
    fs.create_file(os.path.join(target_dir, "001_1.mp4"))
    interpreter_id = "001"
    next_take = determine_next_take_number(target_dir, interpreter_id)
    assert next_take == 2
    assert "Non-integer take number in filename 001_abc.mp4: abc" in caplog.text

def test_determine_next_take_number_permission_error(fs, caplog):
    target_dir = "/fake/path/interpreter_id"
    fs.create_dir(target_dir)
    interpreter_id = "001"
    with mock.patch('CHDatasetManager.file_system_operations.os.listdir', side_effect=PermissionError("Fake permission error")):
        next_take = determine_next_take_number(target_dir, interpreter_id)
        assert next_take == -1
        assert "Error determining next take" in caplog.text


# --- Tests for move_and_rename_video ---

def test_move_and_rename_video_success(fs):
    source_dir = "/fake/source"
    target_dir = "/fake/target/category/word/001"
    source_file_name = "original_video.mp4"
    source_file_path = os.path.join(source_dir, source_file_name)
    new_filename = "001_1.mp4"
    final_destination = os.path.join(target_dir, new_filename)

    fs.create_dir(source_dir)
    fs.create_file(source_file_path, contents="dummy video data")
    original_stat = os.stat(source_file_path) # Get stat before move

    success, error_msg = move_and_rename_video(source_file_path, target_dir, new_filename)

    assert success is True
    assert error_msg is None
    assert not fs.exists(source_file_path)
    assert fs.exists(final_destination)
    with open(final_destination, 'r') as f:
        assert f.read() == "dummy video data"
    # Permissions check might be tricky with pyfakefs default stat, focus on existence and content.
    # assert os.stat(final_destination).st_mode == original_stat.st_mode


def test_move_and_rename_video_source_not_found(fs, caplog):
    source_file = "/fake/source/non_existent.mp4"
    target_dir = "/fake/target"
    new_filename = "new_name.mp4"
    success, error_msg = move_and_rename_video(source_file, target_dir, new_filename)
    assert success is False
    assert "Source file not found" in error_msg
    assert "Source file not found for move" in caplog.text
    assert not fs.exists(os.path.join(target_dir, new_filename))

def test_move_and_rename_video_destination_exists(fs, caplog):
    source_dir = "/fake/source"
    target_dir = "/fake/target"
    source_file = os.path.join(source_dir, "original.mp4")
    new_filename = "existing_file.mp4"
    final_destination = os.path.join(target_dir, new_filename)
    fs.create_dir(source_dir)
    fs.create_dir(target_dir)
    fs.create_file(source_file)
    fs.create_file(final_destination)
    success, error_msg = move_and_rename_video(source_file, target_dir, new_filename)
    assert success is False
    assert "Destination file already exists" in error_msg
    assert "Destination file already exists" in caplog.text
    assert fs.exists(source_file)

def test_move_and_rename_video_creates_target_dir(fs):
    source_dir = "/fake/source"
    target_dir = "/fake/target/new/path"
    source_file = os.path.join(source_dir, "original.mp4")
    new_filename = "new_name.mp4"
    final_destination = os.path.join(target_dir, new_filename)
    fs.create_dir(source_dir)
    fs.create_file(source_file)
    assert not fs.exists(target_dir)
    success, error_msg = move_and_rename_video(source_file, target_dir, new_filename)
    assert success is True
    assert error_msg is None
    assert fs.exists(target_dir)
    assert fs.exists(final_destination)

def test_move_and_rename_video_shutil_error(fs, caplog):
    source_dir = "/fake/source"
    target_dir = "/fake/target"
    source_file = os.path.join(source_dir, "original.mp4")
    new_filename = "new_name.mp4"
    fs.create_dir(source_dir)
    fs.create_dir(target_dir)
    fs.create_file(source_file)
    with mock.patch('CHDatasetManager.file_system_operations.shutil.move', side_effect=shutil.Error("Fake move error")):
        success, error_msg = move_and_rename_video(source_file, target_dir, new_filename)
        assert success is False
        assert "Failed to move" in error_msg
        assert "Error moving file" in caplog.text
        assert fs.exists(source_file)


# --- Tests for log_verification_to_csv ---

@mock.patch('CHDatasetManager.file_system_operations.get_app_base_path', return_value='/fake/app/base')
def test_log_verification_to_csv_new_file(mock_get_base_path, fs):
    log_file_path = os.path.join('/fake/app/base', VERIFICATION_LOG_FILE)
    log_entry = {"Field1": "Value1", "Field2": "Value2"}
    mock_now = datetime.datetime(2023, 10, 27, 10, 30, 0)
    with mock.patch('CHDatasetManager.file_system_operations.datetime.datetime') as mock_dt:
        mock_dt.now.return_value = mock_now
        success = log_verification_to_csv(log_entry.copy())
    assert success is True
    assert fs.exists(log_file_path)
    with open(log_file_path, 'r', newline='', encoding='utf-8') as f:
         reader = csv.DictReader(f)
         assert reader.fieldnames == ["Field1", "Field2", "Timestamp"] # Order matters for new file header
         rows = list(reader)
         assert len(rows) == 1
         assert rows[0]["Field1"] == "Value1"
         assert rows[0]["Field2"] == "Value2"
         assert rows[0]["Timestamp"] == "2023-10-27 10:30:00"

@mock.patch('CHDatasetManager.file_system_operations.get_app_base_path', return_value='/fake/app/base')
def test_log_verification_to_csv_append_to_existing(mock_get_base_path, fs):
    log_file_path = os.path.join('/fake/app/base', VERIFICATION_LOG_FILE)
    header = "Field1,Field2,Timestamp\n"
    existing_data = "ValueA,ValueB,2023-10-26 09:00:00\n"
    fs.create_file(log_file_path, contents=header + existing_data, encoding='utf-8')
    log_entry = {"Field1": "ValueX", "Field2": "ValueY"} # New entry has same fields
    mock_now = datetime.datetime(2023, 10, 27, 11, 0, 0)
    with mock.patch('CHDatasetManager.file_system_operations.datetime.datetime') as mock_dt:
        mock_dt.now.return_value = mock_now
        success = log_verification_to_csv(log_entry.copy())
    assert success is True
    with open(log_file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["Field1"] == "ValueA"
    assert rows[1]["Field1"] == "ValueX"
    assert rows[1]["Timestamp"] == "2023-10-27 11:00:00"
    with open(log_file_path, 'r', encoding='utf-8') as f: # Check header not rewritten
        content = f.read()
        assert content.count(header.strip()) == 1


@mock.patch('CHDatasetManager.file_system_operations.get_app_base_path', return_value='/fake/app/base')
def test_log_verification_to_csv_error(mock_get_base_path, fs, caplog):
    log_entry = {"Field1": "Value1"}
    with mock.patch('CHDatasetManager.file_system_operations.open', side_effect=IOError("Fake write error")):
        success = log_verification_to_csv(log_entry)
        assert success is False
        assert "Error logging data to CSV" in caplog.text


# --- Tests for calculate_take_assignment_details ---

@pytest.mark.parametrize(
    "mock_next_take, num_selected, expected_start, expected_end, expected_is_full, expected_error, short_msg_contains, long_msg_contains, expected_avail_slots",
    [
        (1, 0, 1, None, False, False, "Next: 1", "Select videos to assign takes.", MAX_TAKES_PER_WORD_INTERPRETER),
        (1, 1, 1, 1, False, False, "Potential Take: 1", "Ready for approval.", MAX_TAKES_PER_WORD_INTERPRETER),
        (1, 2, 1, 2, False, False, "Potential Takes: 1-2", "Ready for approval.", MAX_TAKES_PER_WORD_INTERPRETER),
        (1, MAX_TAKES_PER_WORD_INTERPRETER, 1, MAX_TAKES_PER_WORD_INTERPRETER, False, False, f"Potential Takes: 1-{MAX_TAKES_PER_WORD_INTERPRETER}", "Ready for approval.", MAX_TAKES_PER_WORD_INTERPRETER),
        (3, 1, 3, 3, False, False, "Potential Take: 3", "Ready for approval.", MAX_TAKES_PER_WORD_INTERPRETER - 2), # MAX_TAKES - (3-1)
        (MAX_TAKES_PER_WORD_INTERPRETER + 1, 1, MAX_TAKES_PER_WORD_INTERPRETER + 1, None, True, True, "Takes: FULL", "Error: Maximum " + str(MAX_TAKES_PER_WORD_INTERPRETER) + " takes already exist.", 0),
        (-1, 1, None, None, False, True, "Takes: Error", "Error checking existing takes", 0),
        (1, MAX_TAKES_PER_WORD_INTERPRETER + 1, 1, MAX_TAKES_PER_WORD_INTERPRETER + 1, False, True, "Takes: Error (Need", "Too many files selected", MAX_TAKES_PER_WORD_INTERPRETER),
        (2, 3, 2, 4, False, False, "Potential Takes: 2-4", "Ready for approval.", MAX_TAKES_PER_WORD_INTERPRETER - 1), # MAX_TAKES - (2-1)
        (MAX_TAKES_PER_WORD_INTERPRETER, 1, MAX_TAKES_PER_WORD_INTERPRETER, MAX_TAKES_PER_WORD_INTERPRETER, False, False, f"Potential Take: {MAX_TAKES_PER_WORD_INTERPRETER}", "Ready for approval.", 1),
        (MAX_TAKES_PER_WORD_INTERPRETER, 0, MAX_TAKES_PER_WORD_INTERPRETER, None, False, False, f"Next: {MAX_TAKES_PER_WORD_INTERPRETER}", "Select videos to assign takes.", 1),
        (MAX_TAKES_PER_WORD_INTERPRETER, 2, MAX_TAKES_PER_WORD_INTERPRETER, MAX_TAKES_PER_WORD_INTERPRETER + 1, False, True, "Takes: Error (Need", "Too many files selected", 1),
    ]
)
@mock.patch('CHDatasetManager.file_system_operations.determine_next_take_number')
def test_calculate_take_assignment_details(
    mock_determine_next_take,
    mock_next_take, num_selected, expected_start, expected_end,
    expected_is_full, expected_error, short_msg_contains, long_msg_contains, expected_avail_slots
):
    mock_determine_next_take.return_value = mock_next_take
    target_folder = "/fake/base/category/word/001"
    interpreter_id = "001"
    result = calculate_take_assignment_details(target_folder, interpreter_id, num_selected)
    assert result['start_take'] == expected_start
    assert result['end_take'] == expected_end
    assert result['is_full'] == expected_is_full
    assert result['error_condition'] == expected_error
    assert short_msg_contains in result['message_short']
    # For regex matching in long message if needed:
    if ".*" in long_msg_contains: # Simple check for regex pattern
        assert mock.re.match(long_msg_contains, result['message_long'])
    else:
        assert long_msg_contains in result['message_long']
    assert result['available_slots'] == expected_avail_slots
    mock_determine_next_take.assert_called_once_with(target_folder, interpreter_id)

def test_calculate_take_assignment_details_invalid_num_selected(caplog):
    target_folder = "/fake/base/category/word/001"
    interpreter_id = "001"
    num_selected = -1
    # The function calls determine_next_take_number even for invalid num_selected to get available_slots
    with mock.patch('CHDatasetManager.file_system_operations.determine_next_take_number', return_value=1) as mock_determine:
        result = calculate_take_assignment_details(target_folder, interpreter_id, num_selected)
        assert result['error_condition'] is True
        assert "Invalid number of selected videos" in result['message_long']
        assert result['start_take'] is None
        mock_determine.assert_called_once_with(target_folder, interpreter_id) # It IS called


# --- Tests for execute_video_processing_fs ---

@mock.patch('CHDatasetManager.file_system_operations.move_and_rename_video')
def test_execute_video_processing_fs_success(mock_move_and_rename):
    source_paths = ("/fake/source/vid1.mp4", "/fake/source/vid2.mov", "/fake/source/vid3.avi")
    confirmed_indices = [0, 2]
    target_folder = "/fake/target/cat/word/001"
    interpreter_id = "001"
    start_take = 5
    mock_move_and_rename.return_value = (True, None)
    processed_paths, errors = execute_video_processing_fs(
        list(source_paths), confirmed_indices, target_folder, interpreter_id, start_take
    )
    assert len(processed_paths) == 2
    assert source_paths[0] in processed_paths
    assert source_paths[2] in processed_paths
    assert source_paths[1] not in processed_paths
    assert not errors
    mock_move_and_rename.assert_any_call(source_paths[0], target_folder, "001_5.mp4")
    mock_move_and_rename.assert_any_call(source_paths[2], target_folder, "001_6.avi")
    assert mock_move_and_rename.call_count == 2

@mock.patch('CHDatasetManager.file_system_operations.move_and_rename_video')
def test_execute_video_processing_fs_with_errors(mock_move_and_rename):
    source_paths = ("/fake/source/vidA.mp4", "/fake/source/vidB.mov", "/fake/source/vidC.avi")
    confirmed_indices = [0, 1, 2]
    target_folder = "/fake/target/cat/word/002"
    interpreter_id = "002"
    start_take = 1
    def mock_move_side_effect(src, dest_folder, new_name):
        if "vidB.mov" in src:
            return False, f"Failed to move {os.path.basename(src)}"
        return True, None
    mock_move_and_rename.side_effect = mock_move_side_effect
    processed_paths, errors = execute_video_processing_fs(
        list(source_paths), confirmed_indices, target_folder, interpreter_id, start_take
    )
    assert len(processed_paths) == 2
    assert source_paths[0] in processed_paths
    assert source_paths[2] in processed_paths
    assert source_paths[1] not in processed_paths
    assert len(errors) == 1
    assert "Failed to move vidB.mov" in errors[0]
    mock_move_and_rename.assert_any_call(source_paths[0], target_folder, "002_1.mp4") # Take based on start_take + 0
    mock_move_and_rename.assert_any_call(source_paths[1], target_folder, "002_2.mov") # Take based on start_take + 1
    mock_move_and_rename.assert_any_call(source_paths[2], target_folder, "002_3.avi") # Take based on start_take + 2 (if using index i)
    assert mock_move_and_rename.call_count == 3

@mock.patch('CHDatasetManager.file_system_operations.move_and_rename_video')
def test_execute_video_processing_fs_no_approved_files(mock_move_and_rename):
    source_paths = ("/fake/source/vid1.mp4", "/fake/source/vid2.mov")
    confirmed_indices = []
    target_folder = "/fake/target/cat/word/001"
    interpreter_id = "001"
    start_take = 1
    processed_paths, errors = execute_video_processing_fs(
        list(source_paths), confirmed_indices, target_folder, interpreter_id, start_take
    )
    assert len(processed_paths) == 0
    assert not errors
    mock_move_and_rename.assert_not_called()
