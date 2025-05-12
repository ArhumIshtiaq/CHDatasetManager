# c:\Users\XC\Desktop\Projects\ConnectHear\CHDatasetManager\video_processing_operations.py
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import os # For basename
import logging
import threading
import concurrent.futures

# Assuming constants are in a sibling file or accessible via package structure
from .constants import (
    NUM_SSIM_KEYFRAMES, SSIM_KEYFRAME_PERCENTAGES, SSIM_RESIZE_WIDTH, SSIM_RESIZE_HEIGHT,
    NUM_PREVIEW_FRAMES, PREVIEW_FRAME_PERCENTAGES, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT, MAX_VIDEOS
)

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, analysis_queue_put_callback):
        self.analysis_queue_put = analysis_queue_put_callback

    def _crop_frame_center_third(self, frame):
        if frame is None or frame.ndim < 3: # Basic check
            return None
        height, width, _ = frame.shape
        if width < 3: # Avoid issues with very narrow frames
            return frame 
        start_col, end_col = width // 3, 2 * width // 3
        return frame[:, start_col:end_col]

    def _resize_to_thumbnail_aspect_ratio(self, frame, target_width, target_height):
        if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            return None
        fh, fw, _ = frame.shape
        scale_w, scale_h = target_width / fw, target_height / fh
        scale = min(scale_w, scale_h)
        final_w, final_h = max(1, int(fw * scale)), max(1, int(fh * scale))
        return cv2.resize(frame, (final_w, final_h), interpolation=cv2.INTER_AREA)

    def _extract_frames_for_video(self, video_path):
        video_filename = os.path.basename(video_path)
        logger.debug(f"Extracting frames from: {video_filename}")
        keyframes_for_ssim = [None] * NUM_SSIM_KEYFRAMES
        preview_pil_images = []
        error_messages = []
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                error_messages.append(f"Error opening: {video_filename}")
                return keyframes_for_ssim, preview_pil_images, error_messages
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                error_messages.append(f"Invalid frame count ({frame_count}) for: {video_filename}")
                return keyframes_for_ssim, preview_pil_images, error_messages
            
            # Extract SSIM Keyframes
            for kf_idx, percentage in enumerate(SSIM_KEYFRAME_PERCENTAGES):
                frame_idx = max(0, min(int(frame_count * percentage), frame_count - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret_key, frame_key_raw = cap.read()
                if ret_key and frame_key_raw is not None:
                    cropped_frame_key = self._crop_frame_center_third(frame_key_raw)
                    if cropped_frame_key.shape[0] > 0 and cropped_frame_key.shape[1] > 0:
                        gray_frame = cv2.cvtColor(cropped_frame_key, cv2.COLOR_BGR2GRAY)
                        keyframes_for_ssim[kf_idx] = cv2.resize(gray_frame, (SSIM_RESIZE_WIDTH, SSIM_RESIZE_HEIGHT), interpolation=cv2.INTER_AREA)
                    else: error_messages.append(f"Invalid crop for SSIM keyframe {kf_idx+1}: {video_filename}")
                else: error_messages.append(f"Error reading SSIM keyframe {kf_idx+1}: {video_filename}")

            # Extract Preview Frames (Original only)
            for pv_idx, percentage in enumerate(PREVIEW_FRAME_PERCENTAGES):
                frame_num_current = max(0, min(int(frame_count * percentage), frame_count - 1))
                pil_img_orig_processed = None

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num_current)
                ret_curr, frame_curr_raw = cap.read()

                if ret_curr and frame_curr_raw is not None:
                    cropped_preview_frame = self._crop_frame_center_third(frame_curr_raw)
                    if cropped_preview_frame is not None and cropped_preview_frame.shape[0] > 0 and cropped_preview_frame.shape[1] > 0:
                        resized_display = self._resize_to_thumbnail_aspect_ratio(cropped_preview_frame, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)
                        if resized_display is not None:
                            pil_img_orig_processed = Image.fromarray(cv2.cvtColor(resized_display, cv2.COLOR_BGR2RGB))
                        else: error_messages.append(f"Error resizing preview frame {pv_idx+1}: {video_filename}")
                    else: error_messages.append(f"Invalid crop for preview frame {pv_idx+1}: {video_filename}")
                else: error_messages.append(f"Error reading preview frame {pv_idx+1}: {video_filename}")
                
                preview_pil_images.append(pil_img_orig_processed)

        except Exception as e:
            error_messages.append(f"Unexpected error processing {video_filename}: {e}")
            logger.error(f"Unexpected error in _extract_frames_for_video for {video_filename}", exc_info=True)
        finally:
            if cap: cap.release()
        return keyframes_for_ssim, preview_pil_images, error_messages

    def _calculate_ssim_scores_for_videos(self, all_keyframes, filepaths):
        num_videos = len(all_keyframes)
        per_video_avg_scores = [None] * num_videos
        error_messages = []
        valid_indices = [i for i, kf_list in enumerate(all_keyframes) if kf_list and all(kf is not None for kf in kf_list)]

        if len(valid_indices) > 1:
            sums = {idx: 0.0 for idx in valid_indices}
            counts = {idx: 0 for idx in valid_indices}
            for i in range(len(valid_indices)):
                idx_i = valid_indices[i]
                for j in range(i + 1, len(valid_indices)):
                    idx_j = valid_indices[j]
                    pair_scores = []
                    for kf_idx in range(NUM_SSIM_KEYFRAMES):
                        f_i, f_j = all_keyframes[idx_i][kf_idx], all_keyframes[idx_j][kf_idx]
                        try:
                            score = ssim(f_i, f_j, data_range=f_i.max() - f_i.min())
                            pair_scores.append(score)
                        except ValueError as ve: error_messages.append(f"SSIM error pair ({idx_i},{idx_j}) kf {kf_idx+1}: {ve}")
                    if pair_scores:
                        avg_pair = np.mean(pair_scores)
                        sums[idx_i] += avg_pair; sums[idx_j] += avg_pair
                        counts[idx_i] += 1; counts[idx_j] += 1
            for idx in valid_indices:
                if counts[idx] > 0: per_video_avg_scores[idx] = sums[idx] / counts[idx]
        elif len(valid_indices) == 1:
            per_video_avg_scores[valid_indices[0]] = 1.0
        return per_video_avg_scores, error_messages

    def run_analysis_in_thread(self, filepaths):
        logger.info(f"Analysis thread started for {len(filepaths)} videos.")
        num_videos = len(filepaths)
        all_previews = [[] for _ in range(num_videos)]
        all_ssim_kf = [[] for _ in range(num_videos)]
        all_errors = []
        results_from_extraction = [None] * num_videos

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_VIDEOS, os.cpu_count() or 1)) as executor:
            future_to_idx = {executor.submit(self._extract_frames_for_video, path): i for i, path in enumerate(filepaths)}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    kf, prev, err = future.result()
                    results_from_extraction[idx] = (kf, prev, err)
                    if err: all_errors.extend(err)
                except Exception as exc:
                    all_errors.append(f"Extraction failed for {os.path.basename(filepaths[idx])}: {exc}")
        
        for idx in range(num_videos):
            if results_from_extraction[idx]:
                all_ssim_kf[idx], all_previews[idx], _ = results_from_extraction[idx]
            else: # Placeholder if extraction failed catastrophically
                all_ssim_kf[idx] = [None] * NUM_SSIM_KEYFRAMES
                all_previews[idx] = [None] * NUM_PREVIEW_FRAMES

        scores, ssim_errs = self._calculate_ssim_scores_for_videos(all_ssim_kf, filepaths)
        if ssim_errs: all_errors.extend(ssim_errs)

        self.analysis_queue_put({
            'type': 'analysis_complete',
            'scores': scores,
            'previews': all_previews,
            'errors': all_errors,
            'filepaths': filepaths
        })
        logger.info("Analysis thread finished.")
