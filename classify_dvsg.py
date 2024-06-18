import torch
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse

def main():
    # python -m spikingjelly.activation_based.examples.classify_dvsg -T 16 -device cuda:0 -b 16 -epochs 64 -data-dir /datasets/DVSGesture/ -amp -cupy -opt adam -lr 0.001 -j 8

    # parsing for passsing command-line interfaces (wtf does this mean???)
    # I think i understand what they mean now, they are arguments that the user can pass in to change the parameters used in the file
    parser = argparse.ArgumentParser(description='Classify DVS Gesture')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=1, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, help='root dir of DVS Gesture dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')

    args = parser.parse_args()
    print(args)

    # creates instance of DVSGestureNet SNN model
    # channels: 128 channels (args.channels)
    # spiking_neuron: the Leaky Integrate-and-Fire (LIF) neuron model, which is commonly used
    # surrogate_function: the ATan function is a hyperbolic tangent-based surrogate function that makes it difficult to use backpropagation directly
    # detach_reset: detaching the reset function from the computational graph
    net = parametric_lif_net.DVSGestureNet(channels=args.channels, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)

    # sets the step mode of the SNN model to 'm' which probably stands for multi-step, allowing capture of temporal dynamics in the input data
    functional.set_step_mode(net, 'm')
    if args.cupy:
        functional.set_backend(net, 'cupy', instance=neuron.LIFNode) # sets the backend to CuPy, alternative to NumPy for GPU acceleration
        # CuPy backend could improve performance of the SNN computations on GPUs

    print(net) # prints SNN model to the console

    # moves the neural network model 'net' to the CUDA GPU device
    net.to(args.device) # is set to 'cuda:0', meaning 0 is the index of the CUDA GPU to use

    
    # root: root directory of the dataset
    # train: whether the dataset instance is for training
    # data_type: specifies dataset should return frames as input data
    # frames_number: sets number of frames (default 16) in each input sequence
    # split_by: dataset should be split into training and testing sets by sample number (instead of random split)
    
    # creates an instance of the DVS128Gesture dataset for training
    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='time')


    # dataset: specify the dataset to be used for training or testing - created before
    # batch_size: batch size (default 16) is the number of samples to be preprocessed together in each iteration
    # shuffle: set to True ensures that the samples in the dataset are shuffled before creating batches. improves generalization
    # drop_lst: determines if the last batch is kept or not
    # num_workers: specifies number of worker processes (default 4) for loading data in parallel. This speeds up data loading
    # pin_memory: pins data loader's output tensors to memory. This improves data tranfer speed between CPU and GPU

    # creates a DataLoader object for the testing dataset
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b, # default 1
        shuffle=True,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )
    
    # optimzer provided by user to be either 'sgd' or 'adam'. 
    optimizer = None
    if args.opt == 'sgd': # creates instance of the Stochastic Gradient Descent (SGD) optimizer with a specified learning rate (default 0.1) and momentum (default 0.9)
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam': # creates instance of Adam optimizer with specified learning rate (default 0.1)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else: # if user provided optimzer is not supported, raises error
        raise NotImplementedError(args.opt)


    # creates an instance of 'CosineAnnealingLR' learning rate scheduler from PyTorch
    # 'CosineAnnealingLR' adjust the learning rate following a cosine annealing schedule
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.resume: # checks if the -resume argument is provided (file path to checkpoint file)
        checkpoint = torch.load(args.resume, map_location='cpu') # loads the checkpoint file onto the CPU even if it was saved on the GPU
        net.load_state_dict(checkpoint['net']) # loads state dict of 'net' from the loaded checkpoint. The state dict contains the model's weights and other parameters
        optimizer.load_state_dict(checkpoint['optimizer']) # loads state dict of the optimzer from the checkpoint. This includes variables like momentum buffers and parameter values
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler']) # loads the state dict of the learning rate scheduler from the checkpoint. This includes info about the learning rate and the schedule to adjust it


    net.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for frame, label in test_data_loader:
            frame = frame.to(args.device)
            frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            # [batch size, time step, channel, height, width]
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 11).float()
            out_fr = net(frame).mean(0)
            loss = F.mse_loss(out_fr, label_onehot)
            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out_fr.argmax(1) == label).float().sum().item()
            functional.reset_net(net)
    test_loss /= test_samples
    test_acc /= test_samples

    print(args)
    print(f'test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}')

if __name__ == '__main__':
    main()