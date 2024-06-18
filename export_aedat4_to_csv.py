import dv_processing as dv
import argparse
import pathlib
import cv2
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Save aedat4 data to csv.')

parser.add_argument('-f,--file',
                    dest='file',
                    type=str,
                    required=True,
                    metavar='path/to/file',
                    help='Path to an AEDAT4 file')

args = parser.parse_args()

file = pathlib.Path(args.file)
file_parent = file.parent
file_stem = file.stem

# Open the recording file
recording = dv.io.MonoCameraRecording(args.file)

if recording.isEventStreamAvailable():
    events_path = file_parent / (file_stem + "_events.csv")
    print("Events will be saved under %s" % events_path)

    print("Saving Events...")
    events_packets = []
    while True:
        events = recording.getNextEventBatch()
        if events is None:
            break
        events_packets.append(pd.DataFrame(events.numpy()))
    print("Done reading events, saving into CSV...")
    events_pandas = pd.concat(events_packets)
    events_pandas.to_csv(events_path)
    print("Saved %d events." % len(events_pandas))