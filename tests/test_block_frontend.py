from __future__ import annotations

import pytest

from clatterdrive.block_frontend import LatencyBlockDevice, SparseMemoryBlockStore
from clatterdrive.hdd import HDDLatencyModel


def test_latency_block_device_round_trips_aligned_data() -> None:
    model = HDDLatencyModel(addressable_blocks=128, latency_scale=0.0, enable_background_scan=False)
    device = LatencyBlockDevice(model, SparseMemoryBlockStore())
    payload = b"A" * model.block_bytes + b"B" * model.block_bytes
    try:
        write_stats = device.write_blocks(10, payload, force_unit_access=True)
        restored, read_stats = device.read_blocks(10, 2)

        assert restored == payload
        assert write_stats.block_count == 2
        assert read_stats.block_count == 2
        assert device.flush().op_type == "FLUSH"
    finally:
        device.close()
        model.stop()


def test_latency_block_device_validates_ranges_and_supports_discard() -> None:
    model = HDDLatencyModel(addressable_blocks=16, latency_scale=0.0, enable_background_scan=False)
    device = LatencyBlockDevice(model, SparseMemoryBlockStore())
    try:
        device.write_blocks(15, b"x" * model.block_bytes)
        assert device.discard_blocks(15, 1).op_type == "DISCARD"
        restored, _ = device.read_blocks(15, 1)
        assert restored == bytes(model.block_bytes)

        with pytest.raises(ValueError, match="addressable range"):
            device.read_blocks(15, 2)
        with pytest.raises(ValueError, match="whole number"):
            device.write_blocks(0, b"unaligned")
    finally:
        device.close()
        model.stop()
