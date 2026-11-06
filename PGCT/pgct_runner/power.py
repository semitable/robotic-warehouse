"""Keep training processes at full speed when their console window is in the background (Windows 11).

Windows 11 runs processes whose window is minimized or in the background with EcoQoS ("power throttling":
efficiency cores, low clocks). Measured on this project's PC (i5-13400, 2026-11-06, E98): the same LBF IPPO job ran
at 197 env steps/s as a child of a foreground app and at 48 steps/s from a minimized PowerShell window, with
rollout and PPO both about 4x slower. A process may opt itself out (no administrator rights, no system setting):
SetProcessInformation(ProcessPowerThrottling) with the EXECUTION_SPEED mechanism turned off ("HighQoS" in the
Windows documentation). Elsewhere this is a no-op. It changes scheduling only, never what a run computes.
"""
from __future__ import annotations

import sys

PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
PROCESS_POWER_THROTTLING_INFORMATION_CLASS = 4   # ProcessPowerThrottling in PROCESS_INFORMATION_CLASS


def disable_power_throttling() -> bool:
    """Opt the current process out of Windows execution-speed throttling; True when the call succeeded."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class ProcessPowerThrottlingState(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessInformation.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
        kernel32.SetProcessInformation.restype = wintypes.BOOL
        state = ProcessPowerThrottlingState(PROCESS_POWER_THROTTLING_CURRENT_VERSION,
                                            PROCESS_POWER_THROTTLING_EXECUTION_SPEED, 0)
        return bool(kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), PROCESS_POWER_THROTTLING_INFORMATION_CLASS,
                                                   ctypes.byref(state), ctypes.sizeof(state)))
    except Exception:  # an unavailable API must never stop a run
        return False
