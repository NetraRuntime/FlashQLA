from typing import List, Dict, Tuple

import torch


class CUDATIMING_LOGGER:

    def __init__(self):
        self._timers: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    def start(self, tag: str):
        if tag not in self._timers:
            self._timers[tag] = []
        self._timers[tag].append([torch.cuda.Event(enable_timing=True)])
        self._timers[tag][-1][0].record()

    def end(self, tag: str):
        self._timers[tag][-1].append(torch.cuda.Event(enable_timing=True))
        self._timers[tag][-1][1].record()

    def clear(self):
        self._timers = {}

    def summary(self, num_iters: int):
        result = {
            tag: sum([start.elapsed_time(end) for start, end in timer_list[-num_iters:]]) / num_iters
            for tag, timer_list in self._timers.items()
        }
        self.clear()
        return result

    def __call__(self, tag):
        if tag not in self._timers or len(self._timers[tag][-1]) == 2:
            self.start(tag)
        else:
            self.end(tag)


TIMING_LOGGER = CUDATIMING_LOGGER()


def profile(func, inputs, num_warmups=50, num_iters=50):
    TIMING_LOGGER.clear()
    torch.cuda.synchronize()
    for _ in range(num_warmups):
        func(*inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        func(*inputs)
    end.record()
    torch.cuda.synchronize()
    results = TIMING_LOGGER.summary(num_iters)
    results['total'] = start.elapsed_time(end) / num_iters
    return results
