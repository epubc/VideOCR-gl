from __future__ import annotations

import sys

from . import utils
from .video import Video


def save_subtitles_to_file(
        video_path: str, file_path: str = 'subtitle.srt', lang: str = 'ch', time_start: str = '0:00', time_end: str = '',
        conf_threshold: int = 75, sim_threshold: int = 80, max_merge_gap_sec: float = 0.1, use_fullframe: bool = False,
        use_gpu: bool = False, use_angle_cls: bool = False, use_server_model: bool = False, brightness_threshold: int | None = None,
        ssim_threshold: int = 92, subtitle_position: str = "center", frames_to_skip: int = 1, crop_zones: list[dict[str, int]] | None = None,
        ocr_image_max_width=960, post_processing=False, min_subtitle_duration_sec=0.2,
        normalize_to_simplified_chinese=True, google_credentials="",):

    if crop_zones is None:
        crop_zones = []

    if subtitle_alignments is None:
        subtitle_alignments = [None, None]
    elif len(subtitle_alignments) == 1:
        subtitle_alignments.append(None)

    # Bypassing PaddleOCR checks since we are using Google Cloud Vision
    paddleocr_path = ""
    det_model_dir, rec_model_dir, cls_model_dir = "", "", ""

    v = Video(video_path, paddleocr_path, det_model_dir, rec_model_dir, cls_model_dir, time_end)
    try:
        v.run_ocr(
            use_gpu, lang, use_angle_cls, time_start, time_end, conf_threshold,
            use_fullframe, brightness_threshold, ssim_threshold, subtitle_position,
            frames_to_skip, crop_zones, ocr_image_max_width, normalize_to_simplified_chinese,
            google_credentials
        )
    except Exception as e:
        print(f"Error: {e}", flush=True)
        sys.exit(1)
    subtitles = v.get_subtitles(sim_threshold, max_merge_gap_sec, lang, post_processing, min_subtitle_duration_sec, subtitle_alignments)

    with open(file_path, 'w+', encoding='utf-8') as f:
        f.write(subtitles)
