"""Windows-only suspended launch and kill-on-close Job Object containment."""

import ctypes
import subprocess
import time
from ctypes import wintypes as w


class Limits(ctypes.Structure):
    _fields_ = [
        ("ProcessTime", ctypes.c_int64),
        ("JobTime", ctypes.c_int64),
        ("Flags", w.DWORD),
        ("MinWS", ctypes.c_size_t),
        ("MaxWS", ctypes.c_size_t),
        ("ActiveLimit", w.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("Priority", w.DWORD),
        ("Scheduling", w.DWORD),
    ]


class IO(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64) for name in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")
    ]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("Basic", Limits),
        ("IO", IO),
        ("ProcessMemory", ctypes.c_size_t),
        ("JobMemory", ctypes.c_size_t),
        ("PeakProcess", ctypes.c_size_t),
        ("PeakJob", ctypes.c_size_t),
    ]


class Accounting(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in ("User", "Kernel", "PeriodUser", "PeriodKernel")] + [
        (name, w.DWORD) for name in ("Faults", "Total", "Active", "Terminated")
    ]


class ThreadEntry(ctypes.Structure):
    _fields_ = [
        ("Size", w.DWORD),
        ("Usage", w.DWORD),
        ("ID", w.DWORD),
        ("Owner", w.DWORD),
        ("Priority", w.LONG),
        ("Delta", w.LONG),
        ("Flags", w.DWORD),
    ]


class WindowsTree:
    """No probe instruction can run before successful Job assignment."""

    def __init__(self, argv):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
            "SetInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
            "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
            "TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL),
            "QueryInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p], w.BOOL),
            "CloseHandle": ([w.HANDLE], w.BOOL),
            "CreateToolhelp32Snapshot": ([w.DWORD, w.DWORD], w.HANDLE),
            "Thread32First": ([w.HANDLE, ctypes.POINTER(ThreadEntry)], w.BOOL),
            "Thread32Next": ([w.HANDLE, ctypes.POINTER(ThreadEntry)], w.BOOL),
            "OpenThread": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            "ResumeThread": ([w.HANDLE], w.DWORD),
        }
        for name, (args, result) in signatures.items():
            function = getattr(kernel, name)
            function.argtypes, function.restype = args, result
        self.kernel, self.job = kernel, kernel.CreateJobObjectW(None, None)
        self.process = None
        if not self.job:
            raise RuntimeError("job unavailable")
        try:
            limits = ExtendedLimits()
            limits.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel.SetInformationJobObject(self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise RuntimeError("job limits unavailable")
            self.process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=0x4,
                close_fds=True,
                shell=False,
                bufsize=0,  # CREATE_SUSPENDED
            )
            if not kernel.AssignProcessToJobObject(self.job, w.HANDLE(int(self.process._handle))):
                raise RuntimeError("job assignment unavailable")
            snapshot = kernel.CreateToolhelp32Snapshot(4, 0)  # TH32CS_SNAPTHREAD
            if snapshot == w.HANDLE(-1).value:
                raise RuntimeError("thread snapshot unavailable")
            resumed = False
            try:
                entry = ThreadEntry()
                entry.Size = ctypes.sizeof(entry)
                present = kernel.Thread32First(snapshot, ctypes.byref(entry))
                while present:
                    if entry.Owner == self.process.pid:
                        thread = kernel.OpenThread(2, False, entry.ID)  # THREAD_SUSPEND_RESUME
                        if not thread:
                            raise RuntimeError("probe resume unavailable")
                        try:
                            if kernel.ResumeThread(thread) == 0xFFFFFFFF:
                                raise RuntimeError("probe resume unavailable")
                            resumed = True
                        finally:
                            kernel.CloseHandle(thread)
                    present = kernel.Thread32Next(snapshot, ctypes.byref(entry))
            finally:
                kernel.CloseHandle(snapshot)
            if not resumed:
                raise RuntimeError("probe resume unavailable")
        except BaseException:
            self.cleanup(time.monotonic() + 2)
            raise

    def cleanup(self, deadline):
        kernel = self.kernel
        try:
            if self.job:
                kernel.TerminateJobObject(self.job, 1)
            if self.process is not None and self.process.poll() is None:
                # Also kills an unassigned, still-suspended setup failure.
                self.process.kill()
            while time.monotonic() < deadline:
                info = Accounting()
                if not kernel.QueryInformationJobObject(self.job, 1, ctypes.byref(info), ctypes.sizeof(info), None):
                    return False
                if info.Active == 0 and (self.process is None or self.process.poll() is not None):
                    return True
                time.sleep(0.01)
            return False
        finally:
            if self.job:
                kernel.CloseHandle(self.job)
                self.job = None
