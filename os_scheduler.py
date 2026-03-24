import threading
import time
from collections import deque

class OSScheduler:
    """
    Simulates the Linux Block Layer (blk-mq):
    - Software Staging Queues: Per-thread buffering.
    - Elevator Algorithm (SCAN): Reorders requests by LBA to minimize seek.
    - Hardware Dispatch: Sends optimized batches to the HDD.
    """
    def __init__(self, hdd_model):
        self.hdd_model = hdd_model
        self.staging_queue = []
        self.lock = threading.Lock()
        self.dispatch_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self.dispatch_thread.start()
        self.results = {} # request_id -> result

    def submit_bio(self, lba, size, is_write):
        req_id = f"{lba}-{time.time()}-{random.random()}"
        with self.lock:
            # Add to software staging queue
            self.staging_queue.append({
                "id": req_id,
                "lba": lba,
                "size": size,
                "is_write": is_write,
                "event": threading.Event()
            })
        return req_id

    def wait_for_completion(self, req_id):
        # In a real VFS, this would be an async wait or callback.
        # We'll use a simple polling/event mechanism for the simulation.
        while True:
            with self.lock:
                if req_id in self.results:
                    return self.results.pop(req_id)
            time.sleep(0.001)

    def _dispatch_loop(self):
        while True:
            batch = []
            with self.lock:
                if self.staging_queue:
                    # Elevator Algorithm: Sort by LBA
                    self.staging_queue.sort(key=lambda x: x["lba"])
                    batch = self.staging_queue[:8] # Dispatch in small batches
                    self.staging_queue = self.staging_queue[8:]
            
            if batch:
                for req in batch:
                    # Dispatch to HDD NCQ
                    res = self.hdd_model.submit_request(req["lba"], req["size"], req["is_write"])
                    with self.lock:
                        self.results[req["id"]] = res
            else:
                time.sleep(0.01)

import random # Needed for req_id
