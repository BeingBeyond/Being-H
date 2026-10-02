# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0


import av
import cv2
import os
import io
import imageio
import re
import random
import decord
import numpy as np
import torchvision
from PIL import Image
from decord import VideoReader
from typing import List, Optional, Tuple, Union


# ==============================================================================
# Frame Extraction by Indices
# ==============================================================================

def get_frames_by_indices(
    video_path: str,
    indices: list[int] | np.ndarray,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    """
    Extract frames from video at specified frame indices.
    
    Args:
        video_path: Path to video file
        indices: Frame indices to extract
        video_backend: Backend to use ('decord' or 'opencv')
        video_backend_kwargs: Additional kwargs for video backend
        
    Returns:
        Frames as numpy array of shape (N, H, W, C) where N is number of frames
        
    Raises:
        ValueError: If unable to read frame or invalid backend
        NotImplementedError: If backend not supported
        
    Example:
        >>> frames = get_frames_by_indices("video.mp4", [0, 10, 20])
        >>> frames.shape
        (3, 480, 640, 3)
    """
    if video_backend_kwargs is None:
        video_backend_kwargs = {}

    if video_backend == "decord":
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(indices).asnumpy()
    elif video_backend == "opencv":
        frames = []
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)

        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")
        
        try:
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"Unable to read frame at index {idx}")
                frames.append(frame)
        finally:
            cap.release()

        frames = np.array(frames)
    else:
        raise NotImplementedError

    return frames


# ==============================================================================
# Frame Extraction by Timestamps
# ==============================================================================

def get_frames_by_timestamps(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    """
    Extract frames from video at specified timestamps.
    
    Args:
        video_path: Path to video file
        timestamps: Timestamps in seconds
        video_backend: Backend to use ('decord', 'opencv', or 'torchvision_av')
        video_backend_kwargs: Additional kwargs for video backend
        
    Returns:
        Frames as numpy array of shape (N, H, W, C)
        
    Raises:
        ValueError: If unable to read frame or open video
        NotImplementedError: If backend not supported
        
    Example:
        >>> frames = get_frames_by_timestamps("video.mp4", [0.0, 1.5, 3.0])
        >>> frames.shape
        (3, 480, 640, 3)
    """
    if video_backend_kwargs is None:
        video_backend_kwargs = {}

    if video_backend == "decord":
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        num_frames = len(vr)
        # Retrieve the timestamps for each frame in the video
        frame_ts: np.ndarray = vr.get_frame_timestamp(range(num_frames))
        # Map each requested timestamp to the closest frame index
        # Only take the first element of the frame_ts array which corresponds to start_seconds
        indices = np.abs(frame_ts[:, :1] - timestamps).argmin(axis=0)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "opencv":
        # Open the video file
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        # Retrieve the total number of frames
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Calculate timestamps for each frame
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_ts = np.arange(num_frames) / fps
        frame_ts = frame_ts[:, np.newaxis]  # Reshape to (num_frames, 1) for broadcasting
        # Map each requested timestamp to the closest frame index
        indices = np.abs(frame_ts - timestamps).argmin(axis=0)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    elif video_backend == "torchvision_av":
        torchvision.set_video_backend("pyav")
        reader = torchvision.io.VideoReader(video_path, "video")

        # Sort timestamps to process in chronological order, track original indices
        ts_array = np.array(timestamps, dtype=np.float64)
        sort_idx = np.argsort(ts_array)
        sorted_ts = ts_array[sort_idx]

        first_ts = float(sorted_ts[0])

        # Seek to the keyframe at or before the first target timestamp, then iterate
        # forward frame-by-frame until each target timestamp is reached, keeping only
        # the closest frame per requested timestamp. Returning every decoded frame
        # instead would hand the caller more frames than it asked for.
        # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
        reader.seek(first_ts, keyframes_only=True)

        result = {}
        prev_frame = None
        prev_pts = None
        target_ptr = 0

        for frame in reader:
            current_pts = frame["pts"]
            # frame["data"] is (C, H, W), convert to (H, W, C)
            current_frame = frame["data"].numpy().transpose(1, 2, 0)

            # Assign timestamps that we have now reached or passed
            while target_ptr < len(sorted_ts) and current_pts >= sorted_ts[target_ptr]:
                orig_idx = int(sort_idx[target_ptr])
                target_ts = sorted_ts[target_ptr]
                # Pick the temporally closer frame between prev and current
                if prev_pts is not None and abs(prev_pts - target_ts) < abs(current_pts - target_ts):
                    result[orig_idx] = prev_frame.copy()
                else:
                    result[orig_idx] = current_frame.copy()
                target_ptr += 1

            prev_frame = current_frame
            prev_pts = current_pts

            if target_ptr >= len(sorted_ts):
                break

        reader.container.close()
        reader = None

        # Fill remaining timestamps with last decoded frame (end-of-video edge case)
        while target_ptr < len(sorted_ts):
            if prev_frame is not None:
                result[int(sort_idx[target_ptr])] = prev_frame.copy()
            target_ptr += 1

        frames_list = []
        for i in range(len(timestamps)):
            if i not in result:
                raise ValueError(
                    f"No frame could be decoded for timestamp index {i} "
                    f"(ts={timestamps[i]:.4f}s). "
                    f"Video may be corrupted or seek failed: {video_path}"
                )
            frames_list.append(result[i])
        frames = np.array(frames_list)

        if len(frames) != len(timestamps):
            raise ValueError(
                f"Video frame extraction incomplete: "
                f"requested {len(timestamps)} frames, got {len(frames)} frames. "
                f"Video may be corrupted: {video_path}"
            )

        return frames
    else:
        raise NotImplementedError


# ==============================================================================
# Full Video Loading
# ==============================================================================

def get_all_frames(
    video_path: str,
    video_backend: str = "decord",
    video_backend_kwargs: dict = {},
    resize_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Extract all frames from a video.
    
    Args:
        video_path: Path to video file
        video_backend: Backend to use ('decord', 'pyav', or 'torchvision_av')
        video_backend_kwargs: Additional kwargs for video backend
        resize_size: Optional (width, height) to resize frames
        
    Returns:
        All frames as numpy array of shape (N, H, W, C)
        
    Raises:
        NotImplementedError: If backend not supported
        
    Example:
        >>> frames = get_all_frames("video.mp4", resize_size=(320, 240))
        >>> frames.shape
        (100, 240, 320, 3)
    """
    if video_backend == "decord":
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(range(len(vr))).asnumpy()
    elif video_backend == "pyav":
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            frame = frame.to_ndarray(format="rgb24")
            frames.append(frame)
        frames = np.array(frames)
    elif video_backend == "torchvision_av":
        # set backend and reader
        torchvision.set_video_backend("pyav")
        reader = torchvision.io.VideoReader(video_path, "video")
        frames = []
        for frame in reader:
            frames.append(frame["data"].numpy())
        frames = np.array(frames)
        frames = frames.transpose(0, 2, 3, 1)
    else:
        raise NotImplementedError(f"Video backend {video_backend} not implemented")
    # resize frames if specified
    if resize_size is not None:
        frames = [cv2.resize(frame, resize_size) for frame in frames]
        frames = np.array(frames)
    return frames



# ==============================================================================
# Frame Sampling
# ==============================================================================
def get_frame_indices(
    num_frames: int,
    vlen: int,
    sample: str = 'rand',
    fix_start: Optional[int] = None,
    input_fps: float = 1.0,
    max_num_frames: int = -1
) -> List[int]:
    """
    Get frame indices for sampling from video.
    
    Args:
        num_frames: Target number of frames
        vlen: Total video length in frames
        sample: Sampling strategy ('rand', 'middle', 'fps{X}')
        fix_start: Fixed start offset for each interval
        input_fps: Input video FPS
        max_num_frames: Maximum frames to sample
        
    Returns:
        List of frame indices
        
    Raises:
        ValueError: If invalid sampling strategy
    """
    if sample in ['rand', 'middle']: 
        # Uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = [(intervals[i], intervals[i + 1] - 1) for i in range(len(intervals) - 1)]

        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        # Pad with last frame if needed
        if len(frame_indices) < num_frames:  # padded with last frame
            frame_indices = (
                frame_indices + [frame_indices[-1]] * (num_frames - len(frame_indices))
            )

    elif 'fps' in sample:  
        # FPS-based sampling (e.g., 'fps0.5' for 0.5 FPS)
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents

        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]

        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    
    return frame_indices


# ==============================================================================
# Frame Readers
# ==============================================================================

def read_frames_gif(
    video_path, 
    num_frames, sample='rand', 
    fix_start=None,
    client=None, 
    min_num_frames=4
):
    """
    Read frames from GIF file.
    
    Args:
        video_path: Path to GIF file (local or S3)
        num_frames: Maximum number of frames
        sample: Sampling strategy
        fix_start: Fixed start offset
        client: Petrel client for S3 access
        min_num_frames: Minimum frames to sample
        
    Returns:
        List of PIL images
    """
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        gif = imageio.get_reader(io.BytesIO(video_bytes))
    else:
        gif = imageio.get_reader(video_path)

    vlen = len(gif)
    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start)

    frames = []
    for index, frame in enumerate(gif):
        if index in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB).astype(np.uint8)
            frame = Image.fromarray(frame)
            frames.append(frame)

    return frames


def read_frames_decord(
        video_path, 
        num_frames, 
        sample='rand', 
        fix_start=None,
        client=None, 
        clip=None, 
        min_num_frames=4
):
    """
    Read frames from video using decord.
    
    Args:
        video_path: Path to video file (local or S3)
        num_frames: Maximum number of frames
        sample: Sampling strategy
        fix_start: Fixed start offset
        client: Petrel client for S3 access
        clip: Optional (start_sec, end_sec) for temporal clipping
        min_num_frames: Minimum frames to sample
        
    Returns:
        List of PIL images
    """
    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    else:
        video_reader = VideoReader(video_path, num_threads=1)

    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()

    start_index = 0
    if clip:
        start, end = clip
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)

    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    frame_indices = get_frame_indices(
        t_num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps
    )

    if clip:
        frame_indices = [f + start_index for f in frame_indices]
    
    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), np.uint8
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    
    return frames


def read_frames_folder(
        video_path, 
        num_frames, 
        sample='rand', 
        fix_start=None,
        client=None, 
        min_num_frames=4
):
    """
    Read frames from folder of images.
    
    Args:
        video_path: Path to folder (local or S3)
        num_frames: Maximum number of frames
        sample: Sampling strategy
        fix_start: Fixed start offset
        client: Petrel client for S3 access
        min_num_frames: Minimum frames to sample
        
    Returns:
        List of PIL images
    """

    def extract_frame_number(filename):
        # Extract the numeric part from the filename using regular expressions
        match = re.search(r'_(\d+).jpg$', filename)
        return int(match.group(1)) if match else -1

    def sort_frames(frame_paths):
        # Extract filenames from each path and sort by their numeric part
        return sorted(frame_paths, key=lambda x: extract_frame_number(os.path.basename(x)))

    if 's3://' in video_path:
        image_list = sort_frames(client.list(video_path))
        frames = []
        for image in image_list:
            fp = os.path.join(video_path, image)
            frame = Image.open(io.BytesIO(client.get(fp)))
            frames.append(frame)
    else:
        image_list = sort_frames(list(os.listdir(video_path)))
        frames = []
        for image in image_list:
            fp = os.path.join(video_path, image)
            frame = Image.open(fp).convert('RGB')
            frames.append(frame)
    vlen = len(frames)

    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)

    if vlen > t_num_frames:
        frame_indices = get_frame_indices(
            t_num_frames, vlen, sample=sample, fix_start=fix_start
        )
        frames = [frames[i] for i in frame_indices]
    return frames



