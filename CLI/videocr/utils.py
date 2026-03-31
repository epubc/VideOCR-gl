import datetime
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import IO, Any

import av
import numpy as np

from .lang_dictionaries import (
    ARABIC_LANGS,
    CYRILLIC_LANGS,
    DEVANAGARI_LANGS,
    ESLAV_LANGS,
    LATIN_LANGS,
)
from .models import PredictedText

ALIGNMENT_MAP = {
    'bottom-left': 'an1', 'bottom-center': 'an2', 'bottom-right': 'an3',
    'middle-left': 'an4', 'middle-center': 'an5', 'middle-right': 'an6',
    'top-left': 'an7', 'top-center': 'an8', 'top-right': 'an9',
}
VALID_ALIGNMENT_NAMES = set(ALIGNMENT_MAP.keys())


def get_ms_from_time_str(time_str: str) -> float:
    """Convert time string to milliseconds."""
    t = [float(x) for x in time_str.split(":")]
    if len(t) == 3:
        td = datetime.timedelta(hours=t[0], minutes=t[1], seconds=t[2])
    elif len(t) == 2:
        td = datetime.timedelta(minutes=t[0], seconds=t[1])
    else:
        raise ValueError(f'Time data "{time_str}" does not match format "%H:%M:%S"')
    return td.total_seconds() * 1000


def get_srt_timestamp(frame_index: int, fps: float, offset_ms: float = 0.0) -> str:
    """Convert frame index into SRT timestamp."""
    td = datetime.timedelta(milliseconds=(frame_index / fps * 1000 + offset_ms))
    ms = td.microseconds // 1000
    m, s = divmod(td.seconds, 60)
    h, m = divmod(m, 60)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def get_srt_timestamp_from_ms(ms: float) -> str:
    """Convert milliseconds into SRT timestamp."""
    td = datetime.timedelta(milliseconds=ms)
    minutes, seconds = divmod(td.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = td.microseconds // 1000
    return f'{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}'


def frame_to_array(frame: av.VideoFrame, fmt: str) -> np.ndarray[Any, Any]:
    """Converts a frame to an array, safely falls back if threads arg is unsupported."""
    if not hasattr(frame_to_array, "supports_threads"):
        frame_to_array.supports_threads = True  # type: ignore

    if frame_to_array.supports_threads:  # type: ignore
        try:
            return frame.to_ndarray(format=fmt, threads=1)
        except TypeError:
            frame_to_array.supports_threads = False  # type: ignore

    return frame.to_ndarray(format=fmt)


def is_on_same_line(word1: PredictedText, word2: PredictedText) -> bool:
    """Checks if two words are on the same line based on vertical overlap."""
    y_min1 = min(p[1] for p in word1.bounding_box)
    y_max1 = max(p[1] for p in word1.bounding_box)
    y_min2 = min(p[1] for p in word2.bounding_box)
    y_max2 = max(p[1] for p in word2.bounding_box)

    midpoint1 = (y_min1 + y_max1) / 2
    midpoint2 = (y_min2 + y_max2) / 2

    return (y_min1 < midpoint2 < y_max1) or (y_min2 < midpoint1 < y_max2)


def extract_non_chinese_segments(text: str) -> list[tuple[str, str]]:
    """Extracts non chinese segments out of the detected text for post processing."""
    segments: list[tuple[str, str]] = []
    current_segment = ''

    def is_chinese(char: str) -> bool:
        return '\u4e00' <= char <= '\u9fff'

    for char in text:
        if is_chinese(char):
            if current_segment:
                segments.append(('non_chinese', current_segment))
                current_segment = ''
            segments.append(('chinese', char))
        else:
            current_segment += char

    if current_segment:
        segments.append(('non_chinese', current_segment))

    return segments


def read_pipe(pipe: IO[str], output_list: list[str]) -> None:
    """Reads lines from a pipe and appends them to a list."""
    try:
        for line in iter(pipe.readline, ''):
            output_list.append(line)
    finally:
        pipe.close()


def is_process_running(pid: int) -> bool:
    """Check if a process with given PID is still running."""
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5
            )
            return str(pid) in result.stdout
        else:
            if os.path.exists(f"/proc/{pid}"):
                return True
    except (OSError, ProcessLookupError, subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return False


def create_clean_temp_dir() -> str:
    """Cleans up orphaned temporary directories from previous crashed runs and creates a fresh one for the current process."""
    current_pid = os.getpid()
    temp_prefix = f"videocr_temp_{current_pid}_"
    base_temp = tempfile.gettempdir()

    for name in os.listdir(base_temp):
        if name.startswith("videocr_temp_"):
            temp_path = os.path.join(base_temp, name)
            try:
                match = re.match(r"videocr_temp_(\d+)_", name)
                if match:
                    dir_pid = int(match.group(1))

                    if dir_pid == current_pid:
                        continue

                    if os.path.isdir(temp_path):
                        if not is_process_running(dir_pid):
                            shutil.rmtree(temp_path, ignore_errors=True)
            except Exception as e:
                print(f"Could not remove leftover temp dir '{name}': {e}", flush=True)

    return tempfile.mkdtemp(prefix=temp_prefix)


def log_error(message: str, log_name: str = "error_log.txt") -> str:
    """Saves errors to a log file."""
    if sys.platform == "win32":
        log_dir = os.path.join(os.getenv('LOCALAPPDATA') or os.path.expanduser('~'), "VideOCR")
    else:
        log_dir = os.path.join(os.path.expanduser('~'), ".config", "VideOCR")

    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, log_name)
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")

    return log_path
