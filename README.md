# CHDatasetManager (ConnectHear Dataset Manager)

The CHDatasetManager is a Python-based desktop application designed to streamline the process of managing video datasets, specifically for sign language glosses. It helps users select, analyze, and organize video files into a structured directory format, assigning take numbers and ensuring consistency.

## Overview

This tool provides a graphical user interface (GUI) built with Tkinter to:
*   Browse and select a base directory for datasets.
*   Manage interpreter IDs and categories for glosses.
*   Visually inspect and select multiple video files for a specific gloss.
*   Analyze selected videos by extracting preview frames and calculating similarity scores (using SSIM) against a reference (the first selected video).
*   Automatically pre-mark videos for approval based on similarity scores.
*   Allow manual approval/rejection of videos.
*   Process approved videos by renaming them according to a defined convention (`InterpreterID_TakeNumber.extension`) and moving them to the correct subfolder within the dataset structure.
*   Log processing actions to a CSV file for record-keeping.

## Features

*   **GUI Interface:** Easy-to-use interface for managing video files.
*   **Directory Structure Management:** Organizes videos into `BaseDirectory/Category/Word/InterpreterID/` structure.
*   **Video Preview:** Displays animated previews of selected videos.
*   **Similarity Scoring:** Uses Structural Similarity Index (SSIM) to compare video frames and suggest the best takes.
*   **Automated Take Assignment:** Determines the next available take number for a given gloss and interpreter.
*   **Batch Processing:** Allows selection and processing of multiple videos at once (up to a configurable limit).
*   **CSV Logging:** Records details of processed videos, including original and new filenames, scores, and user decisions.
*   **Error Handling:** Provides feedback on issues during video analysis or file operations.
*   **Multithreaded Analysis:** Video frame extraction and analysis are performed in a separate thread to keep the GUI responsive.

## Prerequisites

*   Python 3.7+
*   The following Python libraries (can be installed via pip):
    *   `opencv-python`
    *   `Pillow`
    *   `scikit-image`
    *   `numpy`

## Setup & Installation

1.  **Clone the repository (if applicable) or download the source files.**
    ```bash
    # Example if you were using Git
    # git clone <repository_url>
    # cd CHDatasetManager
    ```

2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate

    pip install opencv-python Pillow scikit-image numpy
    ```
    Alternatively, you can create a `requirements.txt` file with the following content:
    ```
    opencv-python
    Pillow
    scikit-image
    numpy
    ```
    And then install using:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the application:**
    ```bash
    python VideoPlacerv2.py
    ```

2.  **Initial Setup (if first time):**
    *   The application might prompt you to select a base directory if not found in `config.ini`.
    *   Interpreter IDs can be managed via the "Manage Interpreter IDs" option in the "Settings" menu.

3.  **Workflow:**
    *   **Select Base Directory:** Use the "File" > "Select Base Directory" menu option if you need to change it.
    *   **Select Interpreter ID:** Choose an Interpreter ID from the dropdown menu.
    *   **Select Category and Word:**
        *   The tree view on the left will populate with categories found in your base directory.
        *   Expand a category to see the words (subfolders) within it.
        *   Click on a "Word" to select it. This will enable the "Select Video Files" button.
    *   **Select Video Files:**
        *   Click the "Select Video Files" button.
        *   A file dialog will open, allowing you to select one or more video files (e.g., `.mp4`, `.mov`).
        *   The selected files will be analyzed. Previews and similarity scores (compared to the first selected video) will be displayed.
    *   **Approve Videos:**
        *   Videos with higher similarity scores or those meeting certain thresholds might be pre-approved (checkbox ticked).
        *   Manually check/uncheck the "Approve" checkbox below each video preview to confirm which ones you want to keep.
    *   **Process Videos:**
        *   Once you've approved the desired videos, click the "Process Approved Videos" button.
        *   The application will:
            *   Determine the next available take numbers for the selected Word and Interpreter ID.
            *   Rename the approved videos to `InterpreterID_TakeNumber.extension`.
            *   Move them to the `BaseDirectory/Category/Word/InterpreterID/` folder.
            *   Log the operation in `processing_log.csv`.
    *   **Repeat:** Select another Word or change Category/Interpreter ID to continue organizing your dataset.

## Logging

All processing actions are logged in a `processing_log.csv` file located in the same directory as the application. This log includes:
*   Timestamp
*   Base Directory, Category, Word, Interpreter ID
*   Original Filename, New Filename
*   Similarity Score (if applicable)
*   Initial Approval State (auto-marked), Final User Confirmation
*   Assigned Take Number
*   Status (e.g., "Moved", "Error")

## Key Files

*   `VideoPlacerv2.py`: Main application script containing the GUI logic and overall workflow orchestration.
*   `video_processing_operations.py`: Handles video frame extraction, resizing, SSIM calculation, and preview generation.
*   `file_system_operations.py`: Manages file/folder operations like creating directories, moving/renaming files, and determining take numbers.
*   `constants.py`: Defines various constants used throughout the application (e.g., UI dimensions, processing parameters).
*   `logger_config.py`: Configures the application's logging behavior.

## Troubleshooting / Notes

*   Ensure that the video files you are trying to process are not currently open in another application.
*   If video analysis seems slow, it's often due to the number and resolution of frames being processed. The current implementation processes a few keyframes for SSIM and previews.
*   The maximum number of videos that can be selected and analyzed at once is defined by `MAX_VIDEOS` in `constants.py`.
*   The application expects a directory structure where `Category` and `Word` are folder names.
