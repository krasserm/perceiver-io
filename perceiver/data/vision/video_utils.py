import os
from typing import List, Tuple

import cv2
import numpy as np


def read_video_frames(video_path: str) -> List[np.ndarray]:
    if not os.path.exists(video_path):
        raise ValueError(f"Invalid video path supplied. Path '{video_path}' does not exist.")

    frames = []
    cap = cv2.VideoCapture(video_path)
    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        if cap is not None:
            cap.release()

    return frames


def read_video_frame_pairs(video_path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    frames = read_video_frames(video_path)

    return list(zip(frames, frames[1:]))


def write_video(video_path: str, frames: List[np.ndarray], fps: int) -> None:
    _, ext = os.path.splitext(video_path)
    if ext != ".mp4":
        raise ValueError("Invalid video path supplied. Only files of type 'mp4' are supported.")

    frame_shape = frames[0].shape

    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_shape[1], frame_shape[0]))
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()
