# Copyright (c) OpenMMLab. All rights reserved.
import torch


class SetTorchThread:

    def __init__(self, num_thread: int = -1) -> None:
        self.prev_num_threads = torch.get_num_threads()
        if num_thread == -1:
            self.num_threads = self.prev_num_threads
        else:
            self.num_threads = num_thread

    def __enter__(self):
        torch.set_num_threads(self.num_threads)
        pass

    def __exit__(self, exc_type, exc_value, tb):
        torch.set_num_threads(self.prev_num_threads)
        pass
