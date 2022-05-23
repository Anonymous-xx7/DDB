from .train import train_segmentor, init_random_seed, set_random_seed
from .test import single_gpu_test, multi_gpu_test

__all__ = ['train_segmentor', 'init_random_seed', 'set_random_seed', 'single_gpu_test', 'multi_gpu_test']
