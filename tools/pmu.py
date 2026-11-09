"""Unprivileged per-process ARM64 PMU counters, inherited by new workers.

Counters cover the process after attach, including setup/prefill. Timing
still comes from the runtime's independently measured decode phase.
"""
import ctypes
import errno
import os
import struct


class PMU:
    def __init__(self, pid, profile="memory"):
        self.profile = profile
        self.fds = {}
        self.errors = {}
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        events = {"cycles": (0, 0), "instructions": (0, 1),
                  "l1d_read_access": (3, 0), "l1d_read_miss": (3, 1 << 16),
                  "ll_read_access": (3, 2), "ll_read_miss": (3, 2 | (1 << 16)),
                  "dtlb_read_miss": (3, 3 | (1 << 16))}
        if profile == "control":
            events = {"cycles": (0, 0), "instructions": (0, 1),
                      "branch_instructions": (0, 4), "branch_misses": (0, 5),
                      "l1i_read_miss": (3, 1 | (1 << 16)),
                      "stalled_frontend_cycles": (0, 7),
                      "stalled_backend_cycles": (0, 8)}
        elif profile == "scheduling":
            events = {"cycles": (0, 0), "instructions": (0, 1),
                      "task_clock_ns": (1, 1), "context_switches": (1, 3),
                      "cpu_migrations": (1, 4)}
        elif profile != "memory":
            raise ValueError(profile)
        for name, (kind, config) in events.items():
            attr = ctypes.create_string_buffer(128)
            # read_format = total enabled + total running; inherit workers,
            # hardware excludes kernel/hypervisor; software scheduling counts include
            # kernel activity. Starts immediately when attached.
            flags = (1 << 1) if kind == 1 else (1 << 1) | (1 << 5) | (1 << 6)
            struct.pack_into("<IIQQQQQ", attr, 0, kind, 128, config, 0, 0, 3, flags)
            fd = libc.syscall(241, ctypes.byref(attr), ctypes.c_int(pid), ctypes.c_int(-1), ctypes.c_int(-1), ctypes.c_ulong(8))
            if fd < 0:
                self.errors[name] = os.strerror(ctypes.get_errno())
            else:
                self.fds[name] = fd

    def finish(self):
        result = {"scope": "whole process after launch, inherited workers; startup and prefill included", "profile": self.profile, "errors": self.errors}
        for name, fd in self.fds.items():
            try:
                value, enabled, running = struct.unpack("<QQQ", os.read(fd, 24))
                result[name] = {"raw": value, "enabled_ns": enabled, "running_ns": running,
                                "scaled": value*enabled/running if running else None,
                                "running_fraction": running/enabled if enabled else None}
            finally:
                os.close(fd)
        if all(result.get(k, {}).get("scaled") for k in ["instructions", "l1d_read_miss"]):
            result["l1d_read_mpki"] = result["l1d_read_miss"]["scaled"]*1000/result["instructions"]["scaled"]
        if all(result.get(k, {}).get("scaled") for k in ["branch_instructions", "branch_misses"]):
            result["branch_miss_fraction"] = result["branch_misses"]["scaled"]/result["branch_instructions"]["scaled"]
        return result
