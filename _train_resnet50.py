import logging
import shlex
import subprocess
import time
import torch

# from torchvision.models import resnet50
from pytorch_model.resnet import resnet50

_logger = logging.getLogger(__name__)


def get_memory_free() -> bool:
    output = subprocess.check_output(
        shlex.split(
            'nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader'
        )
    )

    memory_usage = output.decode().split('\n')
    memory_usage = [int(m) for m in memory_usage if m != '']

    _logger.info('memory usage: %s', memory_usage)
    # 低于500MB，就return True
    return memory_usage[0] < 500


def occupy_gpu():
    _logger.info('start running resnet50')
    model = resnet50().cuda()
    model = torch.nn.DataParallel(model)
    device = torch.device('cuda')
    num_gpus = torch.cuda.device_count()
    while True:
        x = torch.rand(32 * num_gpus, 3, 224, 224, device=device)
        y = model(x)


def main():
    logging.basicConfig(level=logging.INFO)

    gpus_free = False

    while not gpus_free:
        gpus_free = get_memory_free()
        time.sleep(10)

    occupy_gpu()


if __name__ == "__main__":
    main()
