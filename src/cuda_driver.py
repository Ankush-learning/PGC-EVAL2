"""
cuda_driver.py
Lightweight, zero-dependency Python wrapper for the NVIDIA CUDA Driver API (nvcuda.dll).
Supports explicit device memory, page-locked (pinned) host memory, CUDA Unified Memory,
prefetching, CUDA streams, CUDA events, PTX kernel execution, and a GPU-aware memory pool.
"""

import ctypes
import platform
from pathlib import Path

# CUDA Driver API constants
CUDA_SUCCESS = 0
CU_MEM_ATTACH_GLOBAL = 1
CU_CTX_SCHED_AUTO = 0
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89
CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88


class CudaError(RuntimeError):
    pass


class CudaDriver:
    def __init__(self, device_id: int = 0):
        if platform.system() == "Windows":
            self.lib = ctypes.WinDLL("nvcuda.dll")
        else:
            self.lib = ctypes.CDLL("libcuda.so")

        self._setup_prototypes()
        self._check(self.lib.cuInit(0), "cuInit")

        self.device = ctypes.c_int()
        self._check(self.lib.cuDeviceGet(ctypes.byref(self.device), device_id), "cuDeviceGet")

        self.ctx = ctypes.c_void_p()
        self._check(self.lib.cuCtxCreate_v2(ctypes.byref(self.ctx), CU_CTX_SCHED_AUTO, self.device), "cuCtxCreate_v2")

        self.device_id = device_id
        self.modules = []

    def _setup_prototypes(self):
        lib = self.lib
        c_int = ctypes.c_int
        c_uint = ctypes.c_uint
        c_size_t = ctypes.c_size_t
        c_void_p = ctypes.c_void_p
        c_char_p = ctypes.c_char_p
        c_float = ctypes.c_float
        c_ulonglong = ctypes.c_ulonglong

        lib.cuInit.argtypes = [c_uint]
        lib.cuInit.restype = c_int

        lib.cuDriverGetVersion.argtypes = [ctypes.POINTER(c_int)]
        lib.cuDriverGetVersion.restype = c_int

        lib.cuDeviceGet.argtypes = [ctypes.POINTER(c_int), c_int]
        lib.cuDeviceGet.restype = c_int

        lib.cuDeviceGetName.argtypes = [c_char_p, c_int, c_int]
        lib.cuDeviceGetName.restype = c_int

        lib.cuDeviceTotalMem_v2.argtypes = [ctypes.POINTER(c_size_t), c_int]
        lib.cuDeviceTotalMem_v2.restype = c_int

        lib.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(c_int), c_int, c_int]
        lib.cuDeviceGetAttribute.restype = c_int

        lib.cuCtxCreate_v2.argtypes = [ctypes.POINTER(c_void_p), c_uint, c_int]
        lib.cuCtxCreate_v2.restype = c_int

        lib.cuCtxDestroy_v2.argtypes = [c_void_p]
        lib.cuCtxDestroy_v2.restype = c_int

        lib.cuCtxSynchronize.argtypes = []
        lib.cuCtxSynchronize.restype = c_int

        lib.cuMemGetInfo_v2.argtypes = [ctypes.POINTER(c_size_t), ctypes.POINTER(c_size_t)]
        lib.cuMemGetInfo_v2.restype = c_int

        lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(c_ulonglong), c_size_t]
        lib.cuMemAlloc_v2.restype = c_int

        lib.cuMemFree_v2.argtypes = [c_ulonglong]
        lib.cuMemFree_v2.restype = c_int

        lib.cuMemAllocHost_v2.argtypes = [ctypes.POINTER(c_void_p), c_size_t]
        lib.cuMemAllocHost_v2.restype = c_int

        lib.cuMemFreeHost.argtypes = [c_void_p]
        lib.cuMemFreeHost.restype = c_int

        lib.cuMemAllocManaged.argtypes = [ctypes.POINTER(c_ulonglong), c_size_t, c_uint]
        lib.cuMemAllocManaged.restype = c_int

        lib.cuMemPrefetchAsync.argtypes = [c_ulonglong, c_size_t, c_int, c_void_p]
        lib.cuMemPrefetchAsync.restype = c_int

        lib.cuMemcpyHtoD_v2.argtypes = [c_ulonglong, c_void_p, c_size_t]
        lib.cuMemcpyHtoD_v2.restype = c_int

        lib.cuMemcpyDtoH_v2.argtypes = [c_void_p, c_ulonglong, c_size_t]
        lib.cuMemcpyDtoH_v2.restype = c_int

        lib.cuMemcpyHtoDAsync_v2.argtypes = [c_ulonglong, c_void_p, c_size_t, c_void_p]
        lib.cuMemcpyHtoDAsync_v2.restype = c_int

        lib.cuMemcpyDtoHAsync_v2.argtypes = [c_void_p, c_ulonglong, c_size_t, c_void_p]
        lib.cuMemcpyDtoHAsync_v2.restype = c_int

        lib.cuStreamCreate.argtypes = [ctypes.POINTER(c_void_p), c_uint]
        lib.cuStreamCreate.restype = c_int

        lib.cuStreamDestroy_v2.argtypes = [c_void_p]
        lib.cuStreamDestroy_v2.restype = c_int

        lib.cuStreamSynchronize.argtypes = [c_void_p]
        lib.cuStreamSynchronize.restype = c_int

        lib.cuEventCreate.argtypes = [ctypes.POINTER(c_void_p), c_uint]
        lib.cuEventCreate.restype = c_int

        lib.cuEventDestroy_v2.argtypes = [c_void_p]
        lib.cuEventDestroy_v2.restype = c_int

        lib.cuEventRecord.argtypes = [c_void_p, c_void_p]
        lib.cuEventRecord.restype = c_int

        lib.cuEventSynchronize.argtypes = [c_void_p]
        lib.cuEventSynchronize.restype = c_int

        lib.cuEventElapsedTime.argtypes = [ctypes.POINTER(c_float), c_void_p, c_void_p]
        lib.cuEventElapsedTime.restype = c_int

        lib.cuModuleLoadData.argtypes = [ctypes.POINTER(c_void_p), c_char_p]
        lib.cuModuleLoadData.restype = c_int

        lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(c_void_p), c_void_p, c_char_p]
        lib.cuModuleGetFunction.restype = c_int

        lib.cuLaunchKernel.argtypes = [
            c_void_p,
            c_uint, c_uint, c_uint,
            c_uint, c_uint, c_uint,
            c_uint, c_void_p,
            ctypes.POINTER(c_void_p), ctypes.POINTER(c_void_p),
        ]
        lib.cuLaunchKernel.restype = c_int

    def _check(self, status: int, fn_name: str):
        if status != CUDA_SUCCESS:
            raise CudaError(f"{fn_name} failed with CUDA error code {status}")

    def get_device_info(self) -> dict:
        name_buf = ctypes.create_string_buffer(256)
        self._check(self.lib.cuDeviceGetName(name_buf, 256, self.device), "cuDeviceGetName")

        total_mem = ctypes.c_size_t()
        self._check(self.lib.cuDeviceTotalMem_v2(ctypes.byref(total_mem), self.device), "cuDeviceTotalMem_v2")

        drv_ver = ctypes.c_int()
        self._check(self.lib.cuDriverGetVersion(ctypes.byref(drv_ver)), "cuDriverGetVersion")

        major = ctypes.c_int()
        minor = ctypes.c_int()
        sm_count = ctypes.c_int()
        concurrent_managed = ctypes.c_int()

        self.lib.cuDeviceGetAttribute(ctypes.byref(major), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self.device)
        self.lib.cuDeviceGetAttribute(ctypes.byref(minor), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self.device)
        self.lib.cuDeviceGetAttribute(ctypes.byref(sm_count), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, self.device)
        self.lib.cuDeviceGetAttribute(ctypes.byref(concurrent_managed), CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, self.device)

        free_mem, tot_mem = self.mem_info()
        return {
            "gpu_name": name_buf.value.decode("utf-8"),
            "total_vram_mb": round(total_mem.value / (1024 ** 2), 2),
            "free_vram_mb": round(free_mem / (1024 ** 2), 2),
            "compute_capability": f"{major.value}.{minor.value}",
            "sm_count": sm_count.value,
            "cuda_driver_version": f"{drv_ver.value // 1000}.{(drv_ver.value % 1000) // 10}",
            "concurrent_managed_access": bool(concurrent_managed.value),
            "os": f"{platform.system()} {platform.release()}",
            "python_version": platform.python_version(),
        }

    def mem_info(self):
        free_b = ctypes.c_size_t()
        total_b = ctypes.c_size_t()
        self._check(self.lib.cuMemGetInfo_v2(ctypes.byref(free_b), ctypes.byref(total_b)), "cuMemGetInfo_v2")
        return free_b.value, total_b.value

    def mem_alloc(self, nbytes: int) -> int:
        ptr = ctypes.c_ulonglong()
        self._check(self.lib.cuMemAlloc_v2(ctypes.byref(ptr), nbytes), "cuMemAlloc_v2")
        return ptr.value

    def mem_free(self, ptr: int):
        self._check(self.lib.cuMemFree_v2(ptr), "cuMemFree_v2")

    def mem_alloc_host(self, nbytes: int) -> int:
        ptr = ctypes.c_void_p()
        self._check(self.lib.cuMemAllocHost_v2(ctypes.byref(ptr), nbytes), "cuMemAllocHost_v2")
        return ptr.value

    def mem_free_host(self, ptr: int):
        self._check(self.lib.cuMemFreeHost(ctypes.c_void_p(ptr)), "cuMemFreeHost")

    def mem_alloc_managed(self, nbytes: int) -> int:
        ptr = ctypes.c_ulonglong()
        self._check(self.lib.cuMemAllocManaged(ctypes.byref(ptr), nbytes, CU_MEM_ATTACH_GLOBAL), "cuMemAllocManaged")
        return ptr.value

    def mem_prefetch_async(self, ptr: int, nbytes: int, dst_device: int, stream=None):
        st = ctypes.c_void_p(0) if stream is None else stream
        status = self.lib.cuMemPrefetchAsync(ptr, nbytes, dst_device, st)
        return status == CUDA_SUCCESS

    def memcpy_htod(self, dst_dev: int, src_host: int, nbytes: int):
        self._check(self.lib.cuMemcpyHtoD_v2(dst_dev, ctypes.c_void_p(src_host), nbytes), "cuMemcpyHtoD_v2")

    def memcpy_dtoh(self, dst_host: int, src_dev: int, nbytes: int):
        self._check(self.lib.cuMemcpyDtoH_v2(ctypes.c_void_p(dst_host), src_dev, nbytes), "cuMemcpyDtoH_v2")

    def memcpy_htod_async(self, dst_dev: int, src_host: int, nbytes: int, stream):
        self._check(self.lib.cuMemcpyHtoDAsync_v2(dst_dev, ctypes.c_void_p(src_host), nbytes, stream), "cuMemcpyHtoDAsync_v2")

    def memcpy_dtoh_async(self, dst_host: int, src_dev: int, nbytes: int, stream):
        self._check(self.lib.cuMemcpyDtoHAsync_v2(ctypes.c_void_p(dst_host), src_dev, nbytes, stream), "cuMemcpyDtoHAsync_v2")

    def create_stream(self):
        st = ctypes.c_void_p()
        self._check(self.lib.cuStreamCreate(ctypes.byref(st), 0), "cuStreamCreate")
        return st

    def destroy_stream(self, stream):
        self._check(self.lib.cuStreamDestroy_v2(stream), "cuStreamDestroy_v2")

    def stream_synchronize(self, stream):
        self._check(self.lib.cuStreamSynchronize(stream), "cuStreamSynchronize")

    def synchronize(self):
        self._check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")

    def create_event(self):
        ev = ctypes.c_void_p()
        self._check(self.lib.cuEventCreate(ctypes.byref(ev), 0), "cuEventCreate")
        return ev

    def destroy_event(self, ev):
        self._check(self.lib.cuEventDestroy_v2(ev), "cuEventDestroy_v2")

    def record_event(self, ev, stream=None):
        st = ctypes.c_void_p(0) if stream is None else stream
        self._check(self.lib.cuEventRecord(ev, st), "cuEventRecord")

    def event_elapsed_ms(self, start_ev, end_ev) -> float:
        self._check(self.lib.cuEventSynchronize(end_ev), "cuEventSynchronize")
        ms = ctypes.c_float()
        self._check(self.lib.cuEventElapsedTime(ctypes.byref(ms), start_ev, end_ev), "cuEventElapsedTime")
        return float(ms.value)

    def load_ptx(self, ptx_path: Path):
        ptx_bytes = Path(ptx_path).read_bytes() + b"\x00"
        mod = ctypes.c_void_p()
        self._check(self.lib.cuModuleLoadData(ctypes.byref(mod), ptx_bytes), "cuModuleLoadData")
        self.modules.append(mod)
        return mod

    def get_function(self, module, func_name: str):
        fn = ctypes.c_void_p()
        self._check(self.lib.cuModuleGetFunction(ctypes.byref(fn), module, func_name.encode("utf-8")), "cuModuleGetFunction")
        return fn

    def launch_fused_activation(self, fn, d_x: int, d_y: int, alpha: float, beta: float, n: int, block_size: int = 256, stream=None):
        grid_x = (n + block_size - 1) // block_size
        arg_x = ctypes.c_ulonglong(d_x)
        arg_y = ctypes.c_ulonglong(d_y)
        arg_alpha = ctypes.c_float(alpha)
        arg_beta = ctypes.c_float(beta)
        arg_n = ctypes.c_uint(n)

        params = (ctypes.c_void_p * 5)(
            ctypes.addressof(arg_x),
            ctypes.addressof(arg_y),
            ctypes.addressof(arg_alpha),
            ctypes.addressof(arg_beta),
            ctypes.addressof(arg_n),
        )
        st = ctypes.c_void_p(0) if stream is None else stream
        self._check(
            self.lib.cuLaunchKernel(
                fn,
                grid_x, 1, 1,
                block_size, 1, 1,
                0, st,
                params, None,
            ),
            "cuLaunchKernel",
        )

    def close(self):
        if self.ctx:
            self.lib.cuCtxDestroy_v2(self.ctx)
            self.ctx = None


class GpuMemoryPool:
    """
    GPU-Aware Caching Memory Allocator (similar in principle to PyTorch CUDACachingAllocator).
    Reuses freed device blocks by size bucket to eliminate repeated cuMemAlloc/cuMemFree OS/driver calls.
    """
    def __init__(self, drv: CudaDriver):
        self.drv = drv
        self.free_blocks = {}  # nbytes -> list[int]
        self.total_driver_allocs = 0
        self.cache_hits = 0

    def alloc(self, nbytes: int) -> int:
        bucket = self.free_blocks.get(nbytes)
        if bucket:
            self.cache_hits += 1
            return bucket.pop()
        self.total_driver_allocs += 1
        return self.drv.mem_alloc(nbytes)

    def free(self, ptr: int, nbytes: int):
        self.free_blocks.setdefault(nbytes, []).append(ptr)

    def release_all(self):
        for bucket in self.free_blocks.values():
            for ptr in bucket:
                self.drv.mem_free(ptr)
        self.free_blocks.clear()
