import os
from argparse import ArgumentParser

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

from VisionModel import ToyModel
from training import TVTrainer


parser = ArgumentParser(description='DDP TorchVision Training')

parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR',
                    help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help=('number of batches processed locally before '
                          + 'executing allreduce across workers; it multiplies '
                          + 'total batch size.'))
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size per process (default: 256)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print-freq', type=int, default=10,
                    help='output frequency')
parser.add_argument('--sync_bn', action='store_true',
                    help='enabling apex sync BN.')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Mixed precision
parser.add_argument('--amp', '-a', action='store_true')
parser.add_argument('--opt-level', type=str, default="O1")
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)
parser.add_argument('--channels-last', type=bool, default=False)

# Arch
parser.add_argument('--arch', type=str, default="resnet50")

# Deterministic runtime
parser.add_argument('--deterministic', action='store_true')

parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

# Profiling NVTX
parser.add_argument('--prof', default=-1, type=int, help='Only run 10 iterations for profiling.')

# Allow input args for num_nodes, node_id, num_gpus
parser.add_argument('--num_nodes', type=int, default=1,
                    help='Number of available nodes/hosts')
parser.add_argument('--node_id', type=int, default=0,
                    help='Unique ID to identify the current node/host')
parser.add_argument('--num_gpus', type=int, default=1,
                    help='Number of GPUs in each node')


args = parser.parse_args()


print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("GPU count:", torch.cuda.device_count())


def init_distributed():
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    num_gpus = torch.cuda.device_count()

    if local_rank >= num_gpus:
        raise RuntimeError(f"local_rank {local_rank} >= available GPUs {num_gpus}")

    torch.cuda.set_device(local_rank)

    return local_rank, global_rank, world_size



def run(args):
    crop_size = 224
    val_size = 256

    traindir = "training_ultrasounds"
    valdir = "test_ultrasounds"

    if args.global_rank == 0:
        print(f"World size: {args.world_size}")

    # Scale LR
    args.lr = args.lr * (args.batch_size * args.world_size) / 256.0

    model = models.resnet50(num_classes=3).cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank]
    )

    optimizer = optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(val_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ])
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.global_rank
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=args.world_size,
        rank=args.global_rank
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True
    )

    trainer = TVTrainer(
        args,
        train_loader,
        train_sampler,
        val_loader,
        model,
        optimizer
    )

    trainer.run()


def main():
    args.local_rank, args.global_rank, args.world_size = init_distributed()

    args.device = torch.device(f"cuda:{args.local_rank}")
    args.distributed = args.world_size > 1


    args.best_prec1 = 0
    args.total_batch_size = args.batch_size * args.world_size

    cudnn.benchmark = True

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)

    run(args)


if __name__ == "__main__":
    main()
