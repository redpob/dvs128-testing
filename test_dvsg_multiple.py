import os
import torch
import torch.nn.functional as F
import dv_processing as dv
import numpy as np
import cv2 as cv
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.datasets import play_frame
import argparse
from datetime import timedelta

print(torch.cuda.is_available())

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

    parser.add_argument('-d', default='recordings', dest='directory',type=str,help='Path to a directory with AEDAT4 files')
    parser.add_argument('-resume', 
                        default='logs/T16_b8_sgd_lr0.1_c128/checkpoint_max.pth', 
                        type=str, 
                        help='resume from the checkpoint path')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-device', default='cuda:0', help='device')
    # parser.add_argument('-device', default='cpu', help='device')

    args = parser.parse_args()
    
    # LOAD MODEL
    net = parametric_lif_net.DVSGestureNet(channels=args.channels, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    functional.set_step_mode(net, 'm')
    net.to(args.device) 
    checkpoint = torch.load(args.resume) # loads the checkpoint file onto the CPU even if it was saved on the GPU
    net.load_state_dict(checkpoint['net']) # loads state dict of 'net' from the loaded checkpoint. The state dict contains the model's weights and other parameters

    # OPEN RECORDING
    file_paths = [os.path.join(args.directory, file_name) for file_name in os.listdir(args.directory) if file_name.endswith('.aedat4')]
    videos = []

    # DEFINE SLICING CALLBACK
    def slicing_callback(events: dv.EventStore):
        accumulator.accept(events)

        # Visualize events
        # visualizer = dv.visualization.EventVisualizer((128, 128))
        # input = visualizer.generateImage(events)
        # cv.imshow('input', input)
        # cv.waitKey(100)

        # Store events
        x = np.array([event.x() for event in events])
        y = np.array([event.y() for event in events])
        p = np.array([event.polarity() for event in events])
        frame = integrate_events_segment_to_frame(x, y, p, 128, 128)
        # print(f"Lowest time: {events.getLowestTime()}, Highest time: {events.getHighestTime()}")
        video.append(frame)

        # print(f"Events in batch: {len(events)}")

    for file_path in file_paths:
        # print(f"Processing {file_path}")
        recording = dv.io.MonoCameraRecording(file_path)
        video = []

        accumulator = dv.Accumulator(recording.getEventResolution())
        accumulator.setMinPotential(0.0)
        accumulator.setMaxPotential(1.0)
        accumulator.setNeutralPotential(0.5)
        accumulator.setEventContribution(0.15)
        accumulator.setDecayFunction(dv.Accumulator.Decay.EXPONENTIAL)
        accumulator.setDecayParam(1e+6)
        accumulator.setIgnorePolarity(False)
        accumulator.setSynchronousDecay(False)

        if not recording.isEventStreamAvailable():
            print("No event stream available")
            exit(0)

        slicer = dv.EventStreamSlicer()
        slicer.doEveryTimeInterval(timedelta(milliseconds=356.8125), slicing_callback)

        total_events = 0

        while True:
            # Receive events
            events = recording.getNextEventBatch()
            if events is None: 
                # print("No more events")
                break

            # Noise filtering for each batch of events
            # resolution = (128, 128)
            # filter = dv.noise.BackgroundActivityNoiseFilter(resolution, backgroundActivityDuration=timedelta(milliseconds=1)) # 1 millisecond activity period
            # filter.accept(events) # Pass events to the filter
            # events = filter.generateEvents() # Call generate events to apply the noise filter
            
            slicer.accept(events)

            total_events += len(events)

        # print(f"Video shape: {np.shape(video)}")
        print(f"{file_path[11:]} processed with {len(video)} frames")
        print(f"Total events: {total_events}")
        
        video = np.array(video)
        video = video[np.newaxis, :]

        videos.append(video)

    labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    test_data = zip(videos, labels)

    # # EVALUATION
    net.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for video, label in test_data:
            # convert video to tensor
            video = torch.tensor(video)
            video = video.float().to(args.device)
            video = video.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            # [batch size, time step, channel, height, width]

            # assign label "2" to the video
            label = torch.tensor(label)
            label = label.to(args.device)
            
            # create one-hot encoded label
            label_onehot = F.one_hot(label, 11)

            # process video through the network
            out_fr = net(video).mean(0)

            # calculate loss
            loss = F.mse_loss(out_fr.view(-1, 11), label_onehot.view(-1, 11))

            # update evaluation metrics
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()
            # print(f"out_fr: {out_fr}")
            # print(f"label.numel(): {label.numel()}")
            print(f"Label: {out_fr.argmax(1)[0]}, out_fr: {[round(elem, 3) for elem in out_fr.tolist()[0]]}")
        # compute average loss and accuracy
    test_loss /= test_samples
    test_acc /= test_samples

    # print(args)
    print(f'test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}')

if __name__ == '__main__':
    main()