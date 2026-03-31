import sys

from . import utils
from .video import Video


def save_subtitles_to_file(
        video_path: str, file_path='subtitle.srt', lang='ch', time_start='0:00', time_end='',
        conf_threshold=75, sim_threshold=80, max_merge_gap_sec=0.1, use_fullframe=False,
        brightness_threshold=None, ssim_threshold=92, subtitle_position="center", frames_to_skip=1, crop_zones=None,
        ocr_image_max_width=960, post_processing=False, min_subtitle_duration_sec=0.2,
        normalize_to_simplified_chinese=True, subtitle_alignments=None, threads=6) -> None:

    if crop_zones is None: crop_zones = []
    if subtitle_alignments is None: subtitle_alignments = ["", ""]

    # Không cần các biến model dir của Paddle nữa
    v = Video(video_path) 
    try:
        v.run_ocr(
            lang, time_start, time_end, conf_threshold,
            use_fullframe, brightness_threshold, ssim_threshold, subtitle_position,
            frames_to_skip, crop_zones, ocr_image_max_width, normalize_to_simplified_chinese,
            threads=threads
        )
    except Exception as e:
        print(f"Error: {e}", flush=True)
        sys.exit(1)
        
    subtitles = v.get_subtitles(sim_threshold, max_merge_gap_sec, lang, post_processing, min_subtitle_duration_sec, subtitle_alignments)
    with open(file_path, 'w+', encoding='utf-8') as f:
        f.write(subtitles)
