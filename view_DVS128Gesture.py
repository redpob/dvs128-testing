import torch
import torch.nn.functional as F
import dv_processing as dv
import numpy as np
import cv2 as cv
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
import argparse
from datetime import timedelta

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

def main():
    # PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description='Convert aedat4 data to video')

    parser.add_argument('-f,--file',
                        default='dvSave-2024_06_06_17_04_08.aedat4',
                        dest='file',
                        type=str,
                        help='Path to an AEDAT4 file')
    parser.add_argument('-resume', 
                        default='logs/T16_b8_sgd_lr0.1_c128/checkpoint_max.pth', 
                        type=str, 
                        help='resume from the checkpoint path')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-device', default='cuda:0', help='device')

    args = parser.parse_args()
    
    # OPEN RECORDING
    recording = dv.io.MonoCameraRecording(args.file)

    if not recording.isEventStreamAvailable():
        print("No event stream available")
        exit(0)

    while True:
        events = recording.getNextEventBatch()
        if events is None:
            break
        # NOISE FILTERING FOR BACKGROUND ACTIVITY
        resolution = (128, 128)
        filter = dv.noise.BackgroundActivityNoiseFilter(resolution, backgroundActivityDuration=timedelta(milliseconds=1)) # 1 millisecond activity period
        filter.accept(events) # Pass events to the filter
        filtered = filter.generateEvents() # Call generate events to apply the noise filter

        print(f"Filter reduced number of events by a factor of {filter.getReductionFactor()}")
        visualizer = dv.visualization.EventVisualizer(resolution)
        input = visualizer.generateImage(events)
        output = visualizer.generateImage(filtered)
        preview = cv.hconcat([input, output])

        cv.namedWindow("preview", cv.WINDOW_NORMAL)
        cv.imshow("preview", preview)
        cv.waitKey(20)

if __name__ == '__main__':
    main()