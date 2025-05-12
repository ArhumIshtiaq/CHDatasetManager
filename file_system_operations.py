# c:\Users\XC\Desktop\Projects\ConnectHear\CHDatasetManager\file_system_operations.py
import os
import shutil
import re
import csv
import datetime
import logging
import sys # Added for get_app_base_path

# Assuming constants are in a sibling file or accessible via package structure
from .constants import VERIFICATION_LOG_FILE

logger = logging.getLogger(__name__)

# NEW helper function
def get_app_base_path():
    """
    Returns the base path for the application.
    For a frozen app (PyInstaller), it's the directory of the executable.
    For a script, it's the directory of the main script (sys.argv[0]).
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running in a PyInstaller bundle
        return os.path.dirname(sys.executable)
    else:
        # Running as a normal Python script
        return os.path.dirname(os.path.abspath(sys.argv[0]))

def get_directory_structure(base_dir_path):
    """Scans the base directory and returns a_dict of categories and their words."""
    structure = {}
    if not base_dir_path or not os.path.isdir(base_dir_path):
        logger.warning(f"Invalid base directory for structure scan: {base_dir_path}")
        return structure
    try:
        categories = sorted([d for d in os.listdir(base_dir_path) if os.path.isdir(os.path.join(base_dir_path, d))])
        for category_name in categories:
            category_path = os.path.join(base_dir_path, category_name)
            words = sorted([w for w in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, w))])
            structure[category_name] = words
        logger.debug(f"Directory structure scanned for {base_dir_path}: {len(structure)} categories.")
    except Exception as e:
        logger.error(f"Error scanning directory structure {base_dir_path}: {e}", exc_info=True)
    return structure

def determine_next_take_number(target_dir_path, interpreter_id_str):
    """Determines the next available take number in the target directory."""
    highest_take = 0
    if os.path.isdir(target_dir_path):
        try:
            pattern = re.compile(re.escape(f"{interpreter_id_str}_") + r"([1-4])\..+$", re.IGNORECASE)
            for filename in os.listdir(target_dir_path):
                match = pattern.match(filename)
                if match:
                    try:
                        take_number = int(match.group(1))
                        highest_take = max(highest_take, take_number)
                    except ValueError:
                        logger.warning(f"Non-integer take in {filename} for pattern {pattern.pattern}")
            logger.debug(f"Highest take in {target_dir_path} for ID {interpreter_id_str} is {highest_take}.")
            return highest_take + 1
        except Exception as e:
            logger.error(f"Error determining next take in {target_dir_path}: {e}", exc_info=True)
            return -1 # Indicate error
    return 1 # Default if directory doesn't exist

def move_and_rename_video(source_path, target_folder_path, new_filename):
    """Moves and renames a video file. Creates target_folder_path if it doesn't exist."""
    if not os.path.isfile(source_path):
        logger.error(f"Source file not found for move: {source_path}")
        return False, f"Source file not found: {os.path.basename(source_path)}"
    
    try:
        if not os.path.isdir(target_folder_path):
            os.makedirs(target_folder_path)
            logger.info(f"Created target directory: {target_folder_path}")
        
        final_destination_path = os.path.join(target_folder_path, new_filename)
        if os.path.exists(final_destination_path):
            logger.error(f"Destination file already exists: {final_destination_path}")
            return False, f"Destination file already exists: {new_filename}"
            
        shutil.move(source_path, final_destination_path)
        logger.info(f"Moved '{os.path.basename(source_path)}' to '{final_destination_path}'")
        return True, None
    except Exception as e:
        logger.error(f"Error moving file {source_path} to {target_folder_path}/{new_filename}: {e}", exc_info=True)
        return False, f"Failed to move {os.path.basename(source_path)}: {e}"

def log_verification_to_csv(log_entry_data):
    """Logs verification data to the CSV file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry_data["Timestamp"] = timestamp # Add/overwrite timestamp

    log_file_path = os.path.join(get_app_base_path(), VERIFICATION_LOG_FILE)
    try:
        file_exists = os.path.isfile(log_file_path)
        # Ensure fieldnames are derived from the provided dict to maintain order and completeness
        fieldnames = list(log_entry_data.keys()) 

        with open(log_file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists or os.path.getsize(VERIFICATION_LOG_FILE) == 0:
                writer.writeheader()
            writer.writerow(log_entry_data)
        logger.info(f"Successfully logged verification data to {VERIFICATION_LOG_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error logging data to CSV '{log_file_path}': {e}", exc_info=True)
        return False

def calculate_take_assignment_details(target_folder_path: str, interpreter_id: str, num_selected_videos: int) -> dict:
    """
    Calculates take assignment details based on existing files and number of selected videos.

    Args:
        target_folder_path: The specific folder for the interpreter (base/category/word/interpreter_id).
        interpreter_id: The ID of the interpreter.
        num_selected_videos: The number of videos currently selected/approved for processing.

    Returns:
        A dictionary containing:
        - 'start_take' (int | None): The determined starting take number. None if error.
        - 'end_take' (int | None): The potential ending take number. None if error or single take.
        - 'is_full' (bool): True if the maximum number of takes already exists.
        - 'error_condition' (bool): True if an error prevents assignment (e.g., too many selected).
        - 'message_short' (str): A concise message for UI display (e.g., "Takes: 1-2").
        - 'message_long' (str): A more descriptive status message for the UI.
        - 'available_slots' (int): Number of slots available from the start_take.
    """
    from .constants import MAX_TAKES_PER_WORD_INTERPRETER # Import here or at module level

    result = {
        'start_take': None, 'end_take': None, 'is_full': False,
        'error_condition': False, 'message_short': "Takes: -",
        'message_long': "Error in take calculation.", 'available_slots': 0
    }

    if num_selected_videos < 0: # Should not happen if called correctly
        result['error_condition'] = True
        result['message_long'] = "Invalid number of selected videos."
        return result

    start_take = determine_next_take_number(target_folder_path, interpreter_id)
    if start_take == -1:  # Error from determine_next_take_number
        result['error_condition'] = True
        result['message_short'] = "Takes: Error"
        result['message_long'] = "Error checking existing takes in target folder."
        return result

    result['start_take'] = start_take
    result['available_slots'] = MAX_TAKES_PER_WORD_INTERPRETER - start_take + 1

    if start_take > MAX_TAKES_PER_WORD_INTERPRETER:
        result['is_full'] = True
        result['error_condition'] = True # Cannot assign new takes
        result['message_short'] = f"Takes: FULL ({MAX_TAKES_PER_WORD_INTERPRETER}/{MAX_TAKES_PER_WORD_INTERPRETER})"
        result['message_long'] = f"Error: Maximum {MAX_TAKES_PER_WORD_INTERPRETER} takes already exist."
        return result
    
    if num_selected_videos == 0: # No videos to assign takes to yet
        result['message_short'] = f"Next: {start_take} (Avail: {result['available_slots']})"
        result['message_long'] = "Select videos to see potential take assignment."
        return result # No error, but no assignment yet

    end_take = start_take + num_selected_videos - 1
    result['end_take'] = end_take

    if end_take > MAX_TAKES_PER_WORD_INTERPRETER:
        result['error_condition'] = True # Too many videos for available slots
        result['message_short'] = f"Takes: Error (Need {num_selected_videos}, Avail: {result['available_slots']})"
        result['message_long'] = (
            f"Error: Too many files selected ({num_selected_videos}) for available "
            f"takes ({result['available_slots']} starting from {start_take}). Approve carefully."
        )
    else:
        if num_selected_videos == 1:
            result['message_short'] = f"Potential Take: {start_take}"
        else:
            result['message_short'] = f"Potential Takes: {start_take}-{end_take}"
        result['message_long'] = "Ready for approval. Review videos and approve below."
    return result


def execute_video_processing_fs(
    source_video_paths: tuple,
    confirmed_indices: list,
    target_folder_path: str,
    interpreter_id: str,
    start_take_number: int
) -> tuple[list[str], list[str]]:
    """
    Moves and renames approved video files based on their confirmed indices.

    Args:
        source_video_paths: Tuple of all selected source video file paths.
        target_folder_path: The destination directory.
        interpreter_id: The interpreter ID for naming.
        start_take_number: The first take number to assign.

    Returns:
        A tuple containing:
        - success_count (int): Number of files successfully processed.
        - successfully_processed_source_paths (list[str]): List of original source paths that were successfully processed.
        - errors_list (list[str]): List of error messages encountered.
    """
    # (Ensure os and shutil are imported in file_system_operations.py)
    # from .file_system_operations import move_and_rename_video # if it's in the same file
    
    errors_list = []
    successfully_processed_source_paths = []
    current_take_offset = 0

    for original_idx, source_path in enumerate(source_video_paths):
        if original_idx in confirmed_indices:
            assigned_take = start_take_number + current_take_offset
            _, file_extension = os.path.splitext(source_path)
            new_filename = f"{interpreter_id}_{assigned_take}{file_extension}"

            # Assuming move_and_rename_video is defined in the same module or imported
            success, error_msg = move_and_rename_video(
                source_path,
                target_folder_path,
                new_filename
            )
            if success:
                successfully_processed_source_paths.append(source_path)
                current_take_offset += 1
            else:
                errors_list.append(error_msg)
    return successfully_processed_source_paths, errors_list
