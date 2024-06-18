import torch
import torch.nn.functional as F
import dv_processing as dv
import numpy as np
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets import play_frame
import argparse

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
    parser = argparse.ArgumentParser(description='Convert aedat4 data to frames')

    parser.add_argument('-f,--file',
                        default='dvSave-2024_05_31_14_54_35.aedat4',
                        dest='file',
                        type=str,
                        # required=True,
                        # metavar='dv/dvSave-2024_05_31_14_54_35.aedat4',
                        help='Path to an AEDAT4 file')
    parser.add_argument('-resume', 
                        default='logs/T16_b8_sgd_lr0.1_c128/checkpoint_max.pth', 
                        type=str, 
                        help='resume from the checkpoint path')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-data-dir', default = 'DvsGesture', type=str, help='root dir of DVS Gesture dataset')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-b', default=1, type=int, help='batch size')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    
    args = parser.parse_args()
    
    net = parametric_lif_net.DVSGestureNet(channels=args.channels, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)
    functional.set_step_mode(net, 'm')
    net.to(args.device) 
    # creates an instance of the DVS128Gesture dataset for training
    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='time')

    # create data loader
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b, # default 1
        shuffle=True,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )

    # load the checkpoint
    checkpoint = torch.load(args.resume) # loads the checkpoint file onto the CPU even if it was saved on the GPU
    net.load_state_dict(checkpoint['net']) # loads state dict of 'net' from the loaded checkpoint. The state dict contains the model's weights and other parameters
    
    net.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        i = 1
        for video, label in test_data_loader:
            # convert video to tensor
            # video = torch.tensor(video)
            video = video.to(args.device)
            video = video.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            # [batch size, time step, channel, height, width]

            # assign label "10" to each video (random_other_gestures)
            # label = torch.tensor([10])
            label = label.to(args.device)
            
            # create one-hot encoded label
            label_onehot = F.one_hot(label, 11).float()

            # process video through the network
            out_fr = net(video).mean(0)

            # calculate loss
            loss = F.mse_loss(out_fr, label_onehot)

            # update evaluation metrics
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()
            i += 1
            # print(f"Video {i} shape: {video.shape}")
            # print(f"Frames in video {i}: {len(video)}")
            # play_frame(video[0])
            print(f"Total events: {sum(sum(sum(video[0][0])))}")
            # print(f"Label: {out_fr.argmax(1)[0]}, out_fr: {[round(elem, 3) for elem in out_fr.tolist()[0]]}")
        # print(f"Video shape: {video.shape}")
        # print(f"Total videos: {i}")
    # compute average loss and accuracy
    test_loss /= test_samples
    test_acc /= test_samples

    # print(args)
    print(f'test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}')

if __name__ == '__main__':
    main()