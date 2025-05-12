import pytest
import numpy as np
import sys # Import the sys module
from unittest import mock
import os # Needed for mocking os.path.basename

# Mock external libraries before importing the module that uses them
# This is crucial for mocking modules like cv2, PIL, skimage
mock_cv2 = mock.MagicMock()
mock_Image = mock.MagicMock()
mock_ImageTk = mock.MagicMock() # If ImageTk is used directly in the module
mock_ssim = mock.MagicMock()
mock_numpy = mock.MagicMock() # Mock numpy if specific functions are used in ways hard to test otherwise
mock_threading = mock.MagicMock()
mock_concurrent_futures = mock.MagicMock()

sys.modules['cv2'] = mock_cv2
sys.modules['PIL'] = mock_Image
sys.modules['PIL.Image'] = mock_Image # Ensure PIL.Image is mocked
sys.modules['PIL.ImageTk'] = mock_ImageTk # Ensure PIL.ImageTk is mocked if used
sys.modules['skimage.metrics'] = mock.MagicMock() # Mock the metrics submodule
sys.modules['skimage.metrics.structural_similarity'] = mock_ssim
sys.modules['numpy'] = mock_numpy # We mock numpy globally, but tests might use real np for test data
sys.modules['threading'] = mock_threading
sys.modules['concurrent'] = mock.MagicMock() # Mock concurrent package
sys.modules['concurrent.futures'] = mock_concurrent_futures

# Explicitly import the module to help with patching lookups
import CHDatasetManager.video_processing_operations
# Now import the module under test
from CHDatasetManager.video_processing_operations import VideoProcessor
from CHDatasetManager.constants import (
    NUM_SSIM_KEYFRAMES, SSIM_RESIZE_WIDTH, SSIM_RESIZE_HEIGHT,
    NUM_PREVIEW_FRAMES, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT, MAX_VIDEOS
)


@pytest.fixture
def video_processor_instance():
    """Fixture to create a VideoProcessor instance for tests."""
    mock_analysis_queue_put = mock.Mock()
    return VideoProcessor(analysis_queue_put_callback=mock_analysis_queue_put)

# --- Tests for _crop_frame_center_third ---

def test_crop_frame_center_third_valid_frame(video_processor_instance):
    """Test cropping a valid frame."""
    # We are testing the actual implementation of _crop_frame_center_third.
    frame = np.zeros((100, 300, 3), dtype=np.uint8)
    frame[:, 100:200, :] = 255
    cropped_frame = video_processor_instance._crop_frame_center_third(frame)
    assert cropped_frame is not None
    assert cropped_frame.shape == (100, 100, 3)
    assert np.all(cropped_frame == 255)

def test_crop_frame_center_third_narrow_frame(video_processor_instance):
    """Test cropping a very narrow frame (width < 3)."""
    # We are testing the actual implementation of _crop_frame_center_third.
    frame = np.zeros((100, 2, 3), dtype=np.uint8)
    cropped_frame = video_processor_instance._crop_frame_center_third(frame)
    assert cropped_frame is not None
    assert cropped_frame.shape == (100, 2, 3)
    assert np.array_equal(cropped_frame, frame)

def test_crop_frame_center_third_none_frame(video_processor_instance):
    """Test with a None frame."""
    # We are testing the actual implementation of _crop_frame_center_third.
    cropped_frame = video_processor_instance._crop_frame_center_third(None)
    assert cropped_frame is None

def test_crop_frame_center_third_invalid_dims_frame(video_processor_instance):
    """Test with a frame that doesn't have 3 dimensions (e.g. grayscale already)."""
    # We are testing the actual implementation of _crop_frame_center_third.
    frame = np.zeros((100,300), dtype=np.uint8)
    cropped_frame = video_processor_instance._crop_frame_center_third(frame)
    assert cropped_frame is None

def test_crop_frame_center_third_empty_frame(video_processor_instance):
    """Test with an empty frame."""
    # We are testing the actual implementation of _crop_frame_center_third.
    frame = np.zeros((0, 0, 3), dtype=np.uint8)
    cropped_frame = video_processor_instance._crop_frame_center_third(frame)
    assert cropped_frame is not None
    assert cropped_frame.shape == (0, 0, 3)


# --- Tests for _resize_to_thumbnail_aspect_ratio ---

@mock.patch('CHDatasetManager.video_processing_operations.cv2.resize')
def test_resize_to_thumbnail_aspect_ratio_wider(mock_cv2_resize, video_processor_instance):
    """Test resizing a frame wider than target aspect ratio."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    target_w, target_h = 160, 90
    # Consistent scaling logic: scale to fit within target_w and target_h while preserving aspect ratio
    # frame_h = 100, frame_w = 200
    # scale_w_ratio = target_w / frame_w = 160 / 200 = 0.8
    # scale_h_ratio = target_h / frame_h = 90 / 100 = 0.9
    # overall_scale = min(scale_w_ratio, scale_h_ratio) = min(0.8, 0.9) = 0.8
    expected_final_w = int(200 * 0.8) # 160
    expected_final_h = int(100 * 0.8) # 80
    mock_cv2_resize.return_value = np.zeros((expected_final_h, expected_final_w, 3), dtype=np.uint8)
    video_processor_instance._resize_to_thumbnail_aspect_ratio(frame, target_w, target_h)
    mock_cv2_resize.assert_called_once_with(frame, (expected_final_w, expected_final_h), interpolation=mock_cv2.INTER_AREA)

@mock.patch('CHDatasetManager.video_processing_operations.cv2.resize')
def test_resize_to_thumbnail_aspect_ratio_taller(mock_cv2_resize, video_processor_instance):
    """Test resizing a frame taller than target aspect ratio."""
    frame = np.zeros((200, 100, 3), dtype=np.uint8)
    target_w, target_h = 160, 90
    expected_scale = min(target_w / 100, target_h / 200)
    expected_final_w = int(100 * expected_scale)
    expected_final_h = int(200 * expected_scale)
    mock_cv2_resize.return_value = np.zeros((expected_final_h, expected_final_w, 3), dtype=np.uint8)
    video_processor_instance._resize_to_thumbnail_aspect_ratio(frame, target_w, target_h)
    mock_cv2_resize.assert_called_once_with(frame, (expected_final_w, expected_final_h), interpolation=mock_cv2.INTER_AREA)

@mock.patch('CHDatasetManager.video_processing_operations.cv2.resize')
def test_resize_to_thumbnail_aspect_ratio_none_frame(mock_cv2_resize, video_processor_instance):
    """Test resizing a None frame."""
    resized_frame = video_processor_instance._resize_to_thumbnail_aspect_ratio(None, 160, 90)
    assert resized_frame is None
    mock_cv2_resize.assert_not_called()

@mock.patch('CHDatasetManager.video_processing_operations.cv2.resize')
def test_resize_to_thumbnail_aspect_ratio_empty_frame(mock_cv2_resize, video_processor_instance):
    """Test resizing an empty frame."""
    frame = np.zeros((0, 0, 3), dtype=np.uint8)
    resized_frame = video_processor_instance._resize_to_thumbnail_aspect_ratio(frame, 160, 90)
    assert resized_frame is None
    mock_cv2_resize.assert_not_called()

# --- Tests for _extract_frames_for_video ---

@mock.patch('CHDatasetManager.video_processing_operations.cv2.VideoCapture')
@mock.patch('CHDatasetManager.video_processing_operations.cv2.cvtColor')
@mock.patch('CHDatasetManager.video_processing_operations.cv2.resize')
@mock.patch.object(VideoProcessor, '_crop_frame_center_third')
@mock.patch.object(VideoProcessor, '_resize_to_thumbnail_aspect_ratio')
@mock.patch('CHDatasetManager.video_processing_operations.Image.fromarray')
@mock.patch('CHDatasetManager.video_processing_operations.os.path.basename', return_value="fake_video.mp4")
def test_extract_frames_for_video_success(
    mock_basename, mock_fromarray, mock_resize_thumb, mock_crop, mock_cv2_resize, mock_cvtColor, mock_VideoCapture,
    video_processor_instance
):
    """Test successful frame extraction."""
    video_path = "/fake/path/fake_video.mp4"
    mock_cap = mock.MagicMock()
    mock_VideoCapture.return_value = mock_cap
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda x: 100 if x == mock_cv2.CAP_PROP_FRAME_COUNT else 0
    dummy_frame_raw = np.zeros((200, 300, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_frame_raw)
    dummy_cropped_frame = np.zeros((200, 100, 3), dtype=np.uint8)
    mock_crop.return_value = dummy_cropped_frame
    dummy_gray_frame = np.zeros((200, 100), dtype=np.uint8)
    mock_cvtColor.return_value = dummy_gray_frame
    dummy_resized_ssim = np.zeros((SSIM_RESIZE_HEIGHT, SSIM_RESIZE_WIDTH), dtype=np.uint8)
    mock_cv2_resize.return_value = dummy_resized_ssim
    dummy_resized_thumb = np.zeros((THUMBNAIL_HEIGHT, THUMBNAIL_WIDTH, 3), dtype=np.uint8)
    mock_resize_thumb.return_value = dummy_resized_thumb
    dummy_pil_image = mock.MagicMock(spec=mock_Image.Image)
    mock_fromarray.return_value = dummy_pil_image

    with mock.patch('CHDatasetManager.video_processing_operations.NUM_SSIM_KEYFRAMES', 2), \
         mock.patch('CHDatasetManager.video_processing_operations.SSIM_KEYFRAME_PERCENTAGES', [0.25, 0.75]), \
         mock.patch('CHDatasetManager.video_processing_operations.NUM_PREVIEW_FRAMES', 2), \
         mock.patch('CHDatasetManager.video_processing_operations.PREVIEW_FRAME_PERCENTAGES', [0.25, 0.75]):
        keyframes, previews, errors = video_processor_instance._extract_frames_for_video(video_path)

    assert len(keyframes) == 2
    assert all(kf is not None for kf in keyframes)
    assert len(previews) == 2
    assert all(p is not None for p in previews)
    assert not errors
    mock_VideoCapture.assert_called_once_with(video_path)
    mock_cap.isOpened.assert_called_once()
    mock_cap.get.assert_called_once_with(mock_cv2.CAP_PROP_FRAME_COUNT)
    mock_cap.set.assert_any_call(mock_cv2.CAP_PROP_POS_FRAMES, 25)
    mock_cap.set.assert_any_call(mock_cv2.CAP_PROP_POS_FRAMES, 75) # Corrected indices
    assert mock_cap.read.call_count == 4 # Called for each frame extraction attempt (2 SSIM, 2 Preview)
    assert mock_crop.call_count == 4
    assert mock_cvtColor.call_count == 2
    assert mock_cv2_resize.call_count == 2
    assert mock_resize_thumb.call_count == 2
    assert mock_fromarray.call_count == 2
    mock_cap.release.assert_called_once()

@mock.patch('CHDatasetManager.video_processing_operations.cv2.VideoCapture')
@mock.patch('CHDatasetManager.video_processing_operations.os.path.basename', return_value="fake_video.mp4")
def test_extract_frames_for_video_open_error(mock_basename, mock_VideoCapture, video_processor_instance):
    """Test handling error when opening video file."""
    video_path = "/fake/path/fake_video.mp4"
    mock_cap = mock.MagicMock()
    mock_VideoCapture.return_value = mock_cap
    mock_cap.isOpened.return_value = False
    keyframes, previews, errors = video_processor_instance._extract_frames_for_video(video_path)
    assert len(keyframes) == NUM_SSIM_KEYFRAMES
    assert all(kf is None for kf in keyframes)
    assert len(previews) == NUM_PREVIEW_FRAMES
    assert all(p is None for p in previews)
    assert len(errors) == 1
    assert "Error opening: fake_video.mp4" in errors[0]
    mock_VideoCapture.assert_called_once_with(video_path)
    mock_cap.isOpened.assert_called_once()
    mock_cap.get.assert_not_called()
    mock_cap.release.assert_called_once()

@mock.patch('CHDatasetManager.video_processing_operations.cv2.VideoCapture')
@mock.patch('CHDatasetManager.video_processing_operations.os.path.basename', return_value="fake_video.mp4")
def test_extract_frames_for_video_invalid_frame_count(mock_basename, mock_VideoCapture, video_processor_instance):
    """Test handling invalid frame count."""
    video_path = "/fake/path/fake_video.mp4"
    mock_cap = mock.MagicMock()
    mock_VideoCapture.return_value = mock_cap
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda x: 0 if x == mock_cv2.CAP_PROP_FRAME_COUNT else 0
    keyframes, previews, errors = video_processor_instance._extract_frames_for_video(video_path)
    assert len(keyframes) == NUM_SSIM_KEYFRAMES
    assert all(kf is None for kf in keyframes)
    assert len(previews) == NUM_PREVIEW_FRAMES
    assert all(p is None for p in previews)
    assert len(errors) == 1
    assert "Invalid frame count (0) for: fake_video.mp4" in errors[0]
    mock_VideoCapture.assert_called_once_with(video_path)
    mock_cap.isOpened.assert_called_once()
    mock_cap.get.assert_called_once_with(mock_cv2.CAP_PROP_FRAME_COUNT)
    mock_cap.set.assert_not_called()
    mock_cap.read.assert_not_called()
    mock_cap.release.assert_called_once()

@mock.patch('CHDatasetManager.video_processing_operations.cv2.VideoCapture')
@mock.patch.object(VideoProcessor, '_crop_frame_center_third', return_value=None)
@mock.patch('CHDatasetManager.video_processing_operations.os.path.basename', return_value="fake_video.mp4")
def test_extract_frames_for_video_crop_fails(
    mock_basename, mock_crop_failure, mock_VideoCapture, video_processor_instance, caplog
):
    """Test _extract_frames_for_video when _crop_frame_center_third returns None."""
    video_path = "/fake/path/fake_video.mp4"
    mock_cap = mock.MagicMock()
    mock_VideoCapture.return_value = mock_cap
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda x: 100 if x == mock_cv2.CAP_PROP_FRAME_COUNT else 0
    dummy_frame_raw = np.zeros((200, 300, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_frame_raw)
    with mock.patch('CHDatasetManager.video_processing_operations.NUM_SSIM_KEYFRAMES', 1), \
         mock.patch('CHDatasetManager.video_processing_operations.SSIM_KEYFRAME_PERCENTAGES', [0.5]), \
         mock.patch('CHDatasetManager.video_processing_operations.NUM_PREVIEW_FRAMES', 1), \
         mock.patch('CHDatasetManager.video_processing_operations.PREVIEW_FRAME_PERCENTAGES', [0.5]):
        with mock.patch('CHDatasetManager.video_processing_operations.cv2.cvtColor'), \
             mock.patch('CHDatasetManager.video_processing_operations.cv2.resize'), \
             mock.patch.object(VideoProcessor, '_resize_to_thumbnail_aspect_ratio'), \
             mock.patch('CHDatasetManager.video_processing_operations.Image.fromarray'):
            keyframes, previews, errors = video_processor_instance._extract_frames_for_video(video_path)
    assert keyframes[0] is None
    assert previews[0] is None
    assert len(errors) > 0
    assert any("Cropped frame is None for fake_video.mp4 at frame index" in e for e in errors)
    mock_crop_failure.assert_called()
    mock_cap.release.assert_called_once()

@mock.patch('CHDatasetManager.video_processing_operations.cv2.VideoCapture')
@mock.patch('CHDatasetManager.video_processing_operations.os.path.basename', return_value="fake_video.mp4")
def test_extract_frames_for_video_read_error_specific_frame(
    mock_basename, mock_VideoCapture, video_processor_instance, caplog
):
    """Test handling a read error for a specific frame during extraction."""
    video_path = "/fake/path/fake_video.mp4"
    mock_cap = mock.MagicMock()
    mock_VideoCapture.return_value = mock_cap
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda x: 100 if x == mock_cv2.CAP_PROP_FRAME_COUNT else 0
    read_results = [
        (True, np.zeros((200, 300, 3), dtype=np.uint8)),
        (False, None)
    ]
    mock_cap.read.side_effect = read_results * 2
    with mock.patch('CHDatasetManager.video_processing_operations.NUM_SSIM_KEYFRAMES', 2), \
         mock.patch('CHDatasetManager.video_processing_operations.SSIM_KEYFRAME_PERCENTAGES', [0.25, 0.75]), \
         mock.patch('CHDatasetManager.video_processing_operations.NUM_PREVIEW_FRAMES', 2), \
         mock.patch('CHDatasetManager.video_processing_operations.PREVIEW_FRAME_PERCENTAGES', [0.25, 0.75]):
        with mock.patch.object(VideoProcessor, '_crop_frame_center_third', return_value=np.zeros((10,10,3), dtype=np.uint8)), \
             mock.patch('CHDatasetManager.video_processing_operations.cv2.cvtColor', return_value=np.zeros((10,10), dtype=np.uint8)), \
             mock.patch('CHDatasetManager.video_processing_operations.cv2.resize', return_value=np.zeros((10,10), dtype=np.uint8)), \
             mock.patch.object(VideoProcessor, '_resize_to_thumbnail_aspect_ratio', return_value=np.zeros((10,10,3), dtype=np.uint8)), \
             mock.patch('CHDatasetManager.video_processing_operations.Image.fromarray', return_value=mock.Mock()):
            keyframes, previews, errors = video_processor_instance._extract_frames_for_video(video_path)
    assert keyframes[0] is not None
    assert keyframes[1] is None
    assert previews[0] is not None
    assert previews[1] is None
    assert len(errors) > 0
    assert any("Error reading frame" in e for e in errors)
    assert any("fake_video.mp4" in e for e in errors)
    mock_cap.release.assert_called_once()

# --- Tests for _calculate_ssim_scores_for_videos ---

@mock.patch('CHDatasetManager.video_processing_operations.ssim')
@mock.patch('CHDatasetManager.video_processing_operations.np.mean')
def test_calculate_ssim_scores_two_identical_videos(mock_np_mean, mock_ssim, video_processor_instance):
    """Test SSIM calculation with two identical videos."""
    mock_ssim.return_value = 1.0
    mock_np_mean.side_effect = lambda x: x[0] if len(x) == 1 else np.mean(x)
    kf_template = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    all_keyframes = [
        [kf_template] * NUM_SSIM_KEYFRAMES,
        [kf_template] * NUM_SSIM_KEYFRAMES
    ]
    filepaths = ["vid1.mp4", "vid2.mp4"]
    scores, errors = video_processor_instance._calculate_ssim_scores_for_videos(all_keyframes, filepaths)
    assert len(scores) == 2
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(1.0)
    assert not errors
    # For 2 videos, ssim is called for (vid0 vs vid1) and (vid1 vs vid0) effectively
    assert mock_ssim.call_count == 2 * (len(filepaths) - 1) * NUM_SSIM_KEYFRAMES if len(filepaths) > 1 else 0
    for kf_idx in range(NUM_SSIM_KEYFRAMES):
        mock_ssim.assert_any_call(all_keyframes[0][kf_idx], all_keyframes[1][kf_idx], data_range=all_keyframes[0][kf_idx].max())
    assert mock_np_mean.call_count == 2

@mock.patch('CHDatasetManager.video_processing_operations.ssim')
@mock.patch('CHDatasetManager.video_processing_operations.np.mean')
def test_calculate_ssim_scores_two_different_videos(mock_np_mean, mock_ssim, video_processor_instance):
    """Test SSIM calculation with two different videos."""
    mock_ssim.side_effect = [0.5, 0.6, 0.7, 0.8, 0.9]
    mock_np_mean.side_effect = lambda x: np.mean(x)
    kf_v1 = [np.zeros((10, 10), dtype=np.uint8)] * NUM_SSIM_KEYFRAMES
    kf_v2 = [np.ones((10, 10), dtype=np.uint8) * 255] * NUM_SSIM_KEYFRAMES
    all_keyframes = [kf_v1, kf_v2]
    filepaths = ["vid1.mp4", "vid2.mp4"]
    scores, errors = video_processor_instance._calculate_ssim_scores_for_videos(all_keyframes, filepaths)
    assert len(scores) == 2
    expected_avg_score = np.mean([0.5, 0.6, 0.7, 0.8, 0.9][:NUM_SSIM_KEYFRAMES])
    assert scores[0] == pytest.approx(expected_avg_score)
    assert scores[1] == pytest.approx(expected_avg_score)
    assert not errors
    assert mock_ssim.call_count == 2 * (len(filepaths) - 1) * NUM_SSIM_KEYFRAMES if len(filepaths) > 1 else 0
    assert mock_np_mean.call_count == 2

@mock.patch('CHDatasetManager.video_processing_operations.ssim')
def test_calculate_ssim_scores_one_video(mock_ssim, video_processor_instance):
    """Test SSIM calculation with only one video."""
    kf_v1 = [np.zeros((10, 10), dtype=np.uint8)] * NUM_SSIM_KEYFRAMES
    all_keyframes = [kf_v1]
    filepaths = ["vid1.mp4"]
    scores, errors = video_processor_instance._calculate_ssim_scores_for_videos(all_keyframes, filepaths)
    assert len(scores) == 1
    assert scores[0] == pytest.approx(1.0)
    assert not errors
    mock_ssim.assert_not_called()

@mock.patch('CHDatasetManager.video_processing_operations.ssim')
def test_calculate_ssim_scores_no_valid_keyframes(mock_ssim, video_processor_instance):
    """Test SSIM calculation when keyframe extraction failed for all videos."""
    all_keyframes = [
        [None] * NUM_SSIM_KEYFRAMES,
        [None] * NUM_SSIM_KEYFRAMES
    ]
    filepaths = ["vid1.mp4", "vid2.mp4"]
    scores, errors = video_processor_instance._calculate_ssim_scores_for_videos(all_keyframes, filepaths)
    assert len(scores) == 2
    assert scores[0] is None
    assert scores[1] is None
    assert not errors
    mock_ssim.assert_not_called()

@mock.patch('CHDatasetManager.video_processing_operations.ssim', side_effect=ValueError("Fake SSIM error"))
@mock.patch('CHDatasetManager.video_processing_operations.np.mean')
def test_calculate_ssim_scores_ssim_error(mock_np_mean, mock_ssim, video_processor_instance):
    """Test handling errors during SSIM calculation."""
    kf_v1 = [np.zeros((10, 10), dtype=np.uint8)] * NUM_SSIM_KEYFRAMES
    kf_v2 = [np.ones((10, 10), dtype=np.uint8) * 255] * NUM_SSIM_KEYFRAMES
    all_keyframes = [kf_v1, kf_v2]
    filepaths = ["vid1.mp4", "vid2.mp4"]
    scores, errors = video_processor_instance._calculate_ssim_scores_for_videos(all_keyframes, filepaths)
    assert len(scores) == 2
    assert scores[0] is None
    assert scores[1] is None
    assert len(errors) == NUM_SSIM_KEYFRAMES
    assert "SSIM error pair (0,1) kf 1: Fake SSIM error" in errors[0] # Adjusted to match actual error format
    assert mock_ssim.call_count == NUM_SSIM_KEYFRAMES
    mock_np_mean.assert_not_called()

# --- Tests for run_analysis_in_thread ---

@mock.patch('CHDatasetManager.video_processing_operations.concurrent.futures.ThreadPoolExecutor')
@mock.patch.object(VideoProcessor, '_extract_frames_for_video')
@mock.patch.object(VideoProcessor, '_calculate_ssim_scores_for_videos')
def test_run_analysis_in_thread_success(
    mock_calculate_ssim, mock_extract_frames, mock_ThreadPoolExecutor,
    video_processor_instance
):
    """Test successful execution of the analysis thread."""
    filepaths = ["/fake/vid1.mp4", "/fake/vid2.mp4"]
    num_videos = len(filepaths)
    mock_executor = mock.MagicMock()
    mock_ThreadPoolExecutor.return_value.__enter__.return_value = mock_executor
    mock_future1 = mock.MagicMock()
    mock_future2 = mock.MagicMock()
    mock_executor.submit.side_effect = [mock_future1, mock_future2]
    mock_extract_result1 = ([mock.Mock()] * NUM_SSIM_KEYFRAMES, [mock.Mock()] * NUM_PREVIEW_FRAMES, [])
    mock_extract_result2 = ([mock.Mock()] * NUM_SSIM_KEYFRAMES, [mock.Mock()] * NUM_PREVIEW_FRAMES, [])
    mock_future1.result.return_value = mock_extract_result1
    mock_future2.result.return_value = mock_extract_result2
    mock_concurrent_futures.as_completed.return_value = [mock_future1, mock_future2]
    mock_ssim_scores = [0.9, 0.8]
    mock_ssim_errors = []
    mock_calculate_ssim.return_value = (mock_ssim_scores, mock_ssim_errors)
    video_processor_instance.run_analysis_in_thread(filepaths)
    mock_ThreadPoolExecutor.assert_called_once_with(max_workers=min(MAX_VIDEOS, os.cpu_count() or 1))
    assert mock_executor.submit.call_count == num_videos
    mock_executor.submit.assert_any_call(video_processor_instance._extract_frames_for_video, filepaths[0])
    mock_executor.submit.assert_any_call(video_processor_instance._extract_frames_for_video, filepaths[1])
    mock_concurrent_futures.as_completed.assert_called_once_with([mock_future1, mock_future2])
    mock_future1.result.assert_called_once()
    mock_future2.result.assert_called_once()
    expected_all_ssim_kf = [mock_extract_result1[0], mock_extract_result2[0]]
    mock_calculate_ssim.assert_called_once_with(expected_all_ssim_kf, filepaths)
    video_processor_instance.analysis_queue_put.assert_called_once_with({
        'type': 'analysis_complete',
        'scores': mock_ssim_scores,
        'previews': [mock_extract_result1[1], mock_extract_result2[1]],
        'errors': [],
        'filepaths': filepaths
    })

@mock.patch('CHDatasetManager.video_processing_operations.concurrent.futures.ThreadPoolExecutor')
@mock.patch.object(VideoProcessor, '_extract_frames_for_video')
@mock.patch.object(VideoProcessor, '_calculate_ssim_scores_for_videos')
@mock.patch('CHDatasetManager.video_processing_operations.os.path.basename', side_effect=lambda x: os.path.basename(x))
def test_run_analysis_in_thread_extraction_error(
    mock_basename, mock_calculate_ssim, mock_extract_frames, mock_ThreadPoolExecutor,
    video_processor_instance
):
    """Test handling an error during frame extraction."""
    filepaths = ["/fake/vid1.mp4", "/fake/vid2.mp4"]
    num_videos = len(filepaths)
    mock_executor = mock.MagicMock()
    mock_ThreadPoolExecutor.return_value.__enter__.return_value = mock_executor
    mock_future1 = mock.MagicMock()
    mock_future2 = mock.MagicMock()
    mock_executor.submit.side_effect = [mock_future1, mock_future2]
    mock_extract_result1 = ([mock.Mock()] * NUM_SSIM_KEYFRAMES, [mock.Mock()] * NUM_PREVIEW_FRAMES, [])
    mock_future1.result.return_value = mock_extract_result1
    mock_future2.result.side_effect = Exception("Fake extraction error")
    mock_concurrent_futures.as_completed.return_value = [mock_future1, mock_future2]
    mock_ssim_scores = [1.0, None]
    mock_ssim_errors = []
    mock_calculate_ssim.return_value = (mock_ssim_scores, mock_ssim_errors)
    video_processor_instance.run_analysis_in_thread(filepaths)
    assert mock_executor.submit.call_count == num_videos
    mock_future1.result.assert_called_once()
    mock_future2.result.assert_called_once()
    expected_all_ssim_kf = [mock_extract_result1[0], [None] * NUM_SSIM_KEYFRAMES]
    mock_calculate_ssim.assert_called_once_with(expected_all_ssim_kf, filepaths)
    video_processor_instance.analysis_queue_put.assert_called_once()
    call_args, call_kwargs = video_processor_instance.analysis_queue_put.call_args
    result_payload = call_args[0]
    assert result_payload['type'] == 'analysis_complete'
    assert result_payload['scores'] == mock_ssim_scores
    assert len(result_payload['previews']) == num_videos
    assert result_payload['previews'][0] == mock_extract_result1[1]
    assert len(result_payload['previews'][1]) == NUM_PREVIEW_FRAMES
    assert all(p is None for p in result_payload['previews'][1])
    assert len(result_payload['errors']) == 1
    assert "Extraction failed for vid2.mp4: Fake extraction error" in result_payload['errors'][0]
    assert result_payload['filepaths'] == filepaths

@mock.patch('CHDatasetManager.video_processing_operations.concurrent.futures.ThreadPoolExecutor')
@mock.patch.object(VideoProcessor, '_extract_frames_for_video')
@mock.patch.object(VideoProcessor, '_calculate_ssim_scores_for_videos')
def test_run_analysis_in_thread_ssim_calculation_error(
    mock_calculate_ssim, mock_extract_frames, mock_ThreadPoolExecutor,
    video_processor_instance
):
    """Test run_analysis_in_thread when _calculate_ssim_scores_for_videos reports errors."""
    filepaths = ["/fake/vid1.mp4", "/fake/vid2.mp4"]
    mock_executor = mock.MagicMock()
    mock_ThreadPoolExecutor.return_value.__enter__.return_value = mock_executor
    mock_future1 = mock.MagicMock()
    mock_future2 = mock.MagicMock()
    mock_executor.submit.side_effect = [mock_future1, mock_future2]
    mock_extract_result1 = ([mock.Mock()] * NUM_SSIM_KEYFRAMES, [mock.Mock()] * NUM_PREVIEW_FRAMES, [])
    mock_extract_result2 = ([mock.Mock()] * NUM_SSIM_KEYFRAMES, [mock.Mock()] * NUM_PREVIEW_FRAMES, [])
    mock_future1.result.return_value = mock_extract_result1
    mock_future2.result.return_value = mock_extract_result2
    mock_concurrent_futures.as_completed.return_value = [mock_future1, mock_future2]
    mock_ssim_scores_returned = [0.9, None]
    mock_ssim_errors_reported = ["SSIM failed for vid2 pair"]
    mock_calculate_ssim.return_value = (mock_ssim_scores_returned, mock_ssim_errors_reported)
    video_processor_instance.run_analysis_in_thread(filepaths)
    expected_all_ssim_kf = [mock_extract_result1[0], mock_extract_result2[0]]
    mock_calculate_ssim.assert_called_once_with(expected_all_ssim_kf, filepaths)
    video_processor_instance.analysis_queue_put.assert_called_once()
    call_args, _ = video_processor_instance.analysis_queue_put.call_args
    result_payload = call_args[0]
    assert result_payload['type'] == 'analysis_complete'
    assert result_payload['scores'] == mock_ssim_scores_returned
    assert result_payload['previews'] == [mock_extract_result1[1], mock_extract_result2[1]]
    assert result_payload['errors'] == mock_ssim_errors_reported
    assert result_payload['filepaths'] == filepaths

@mock.patch('CHDatasetManager.video_processing_operations.concurrent.futures.ThreadPoolExecutor')
def test_run_analysis_in_thread_empty_filepaths(
    mock_ThreadPoolExecutor, video_processor_instance
):
    """Test run_analysis_in_thread with an empty list of filepaths."""
    filepaths = []
    video_processor_instance.run_analysis_in_thread(filepaths)
    mock_ThreadPoolExecutor.assert_not_called()
    video_processor_instance.analysis_queue_put.assert_called_once_with({
        'type': 'analysis_complete',
        'scores': [],
        'previews': [],
        'errors': ['No filepaths provided for analysis.'],
        'filepaths': []
    })

@mock.patch('CHDatasetManager.video_processing_operations.concurrent.futures.ThreadPoolExecutor')
@mock.patch.object(VideoProcessor, '_extract_frames_for_video')
@mock.patch.object(VideoProcessor, '_calculate_ssim_scores_for_videos')
@pytest.mark.parametrize("mock_cpu_count_val, max_videos_const, expected_workers", [
    (4, 5, 4),
    (8, 5, 5),
    (1, 5, 1),
    (None, 5, 1),
    (4, 1, 1)
])
def test_run_analysis_in_thread_max_workers_calculation(
    mock_calculate_ssim, mock_extract_frames, mock_ThreadPoolExecutor,
    mock_cpu_count_val, max_videos_const, expected_workers,
    video_processor_instance
):
    """Test the calculation of max_workers for ThreadPoolExecutor."""
    filepaths = ["/fake/vid1.mp4"]
    with mock.patch('CHDatasetManager.video_processing_operations.os.cpu_count', return_value=mock_cpu_count_val):
        with mock.patch('CHDatasetManager.video_processing_operations.MAX_VIDEOS', max_videos_const):
            mock_executor = mock.MagicMock()
            mock_ThreadPoolExecutor.return_value.__enter__.return_value = mock_executor
            mock_future = mock.MagicMock()
            mock_executor.submit.return_value = mock_future
            mock_future.result.return_value = ([], [], [])
            mock_concurrent_futures.as_completed.return_value = [mock_future]
            mock_calculate_ssim.return_value = ([], [])
            video_processor_instance.run_analysis_in_thread(filepaths)
            mock_ThreadPoolExecutor.assert_called_once_with(max_workers=expected_workers)
