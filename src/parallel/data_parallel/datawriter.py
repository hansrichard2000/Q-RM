import os
import re

from fairscale.nn.model_parallel.initialize import get_model_parallel_src_rank, get_model_parallel_world_size

from src.parallel.utils import all_gather_object_from_data_parallel_region


class ParallelDataWriter:
    def __init__(self, file: str, mode: str = 'w'):
        self.local_rank = int(os.environ.get("LOCAL_RANK"))
        self.global_rank = int(os.environ.get("RANK"))
        self.model_parallel_src_rank = get_model_parallel_src_rank()
        self.model_parallel_world_size = get_model_parallel_world_size()
        self.worker_id = self.model_parallel_src_rank // self.model_parallel_world_size
        self.file = file
        self.worker_file = self.format_file(file)
        self.writer = None
        if self.global_rank == self.model_parallel_src_rank:
            self.writer = open(self.worker_file, mode=mode, encoding="utf-8")

    def format_file(self, file: str) -> str:
        match = re.search(r".+(\..+)$", file)
        if match:
            return re.sub(rf"{match.group(1)}$", f".worker.{self.worker_id}{match.group(1)}", file)
        return f"{file}.worker.{self.worker_id}"

    def __del__(self):
        self.flush()

        # Gather data from data parallel region
        with open(self.worker_file, mode='r', encoding="utf-8") as reader:
            s = [line.strip() for line in reader]
        ss = all_gather_object_from_data_parallel_region(s)
        if self.local_rank == 0:
            with open(self.file, "w", encoding="utf-8") as writer:
                for s in ss:
                    writer.write(s + '\n')

        if self.writer:
            self.writer.close()

    def flush(self):
        if self.writer:
            self.writer.flush()

    def write(self, s: str, flush: bool = False):
        if self.writer:
            self.writer.write(s)
            if flush:
                self.writer.flush()
