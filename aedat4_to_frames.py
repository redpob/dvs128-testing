import dv_processing as dv
import argparse
import pathlib
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def integrate_events_segment_to_frame(x: np.ndarray, y: np.ndarray, p: np.ndarray, H: int, W: int, j_l: int = 0, j_r: int = -1) -> np.ndarray:
    frame = np.zeros(shape=[2, H * W])
    x = x[j_l: j_r].astype(int)  # avoid overflow
    y = y[j_l: j_r].astype(int)
    p = p[j_l: j_r]
    mask = []
    mask.append(p == 0)
    mask.append(np.logical_not(mask[0]))
    for c in range(2):
        position = y[mask[c]] * W + x[mask[c]]
        events_number_per_pos = np.bincount(position)
        frame[c][np.arange(events_number_per_pos.size)] += events_number_per_pos
    return frame.reshape((2, H, W))

parser = argparse.ArgumentParser(description='Convert aedat4 data to frames')

parser.add_argument('-f,--file',
                    default='dvSave-2024_05_31_14_54_35.aedat4',
                    dest='file',
                    type=str,
                    # required=True,
                    # metavar='dv/dvSave-2024_05_31_14_54_35.aedat4',
                    help='Path to an AEDAT4 file')
 
args = parser.parse_args()

# Open the recording file
recording = dv.io.MonoCameraRecording(args.file)

if recording.isEventStreamAvailable():
    frames = []
    print("Converting events to frames...")
    while True:
        events = recording.getNextEventBatch()
        if events is None:
            break
        x = np.array([event.x() for event in events])
        y = np.array([event.y() for event in events])
        p = np.array([event.polarity() for event in events])
        for i in range(len(events)):
            frame = integrate_events_segment_to_frame(x, y, p, 128, 128)
            frame = frame[np.newaxis, np.newaxis, :]
            frames.append(frame)
    print(f"Frame shape: {frame.shape}")
    print(f"Frames number: {len(frames)}")