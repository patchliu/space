import argparse
import time
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


@dataclass(frozen=True)
class TuneConfig:
    num_core: int
    block_size: int
    block_size_sub: int
    multibuffer: bool


@triton.jit
def silu_mul_split_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    NUM_CORE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_SUB: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)

    block_id = pid
    while block_id < num_blocks:
        block_base = block_id * BLOCK_SIZE
        for sub_offset in tl.static_range(0, BLOCK_SIZE, BLOCK_SIZE_SUB):
            offsets = block_base + sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
            out = x / (1.0 + tl.exp(-x)) * y
            tl.store(out_ptr + offsets, out, mask=mask)

        block_id += NUM_CORE


def require_npu():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("This example needs a torch_npu NPU runtime.")
    return torch.device("npu")


def synchronize():
    torch.npu.synchronize()


def get_num_vector_core(fallback):
    try:
        from triton.runtime import driver

        props = driver.active.utils.get_device_properties(0)
        for key in ("num_vectorcore", "num_aiv", "multi_processor_count"):
            value = props.get(key)
            if value:
                return int(value)
    except Exception:
        pass
    return fallback


def launch_with_optional_multibuffer(kernel, grid, args, meta, multibuffer):
    if not multibuffer:
        kernel[grid](*args, **meta)
        return
    try:
        kernel[grid](*args, **meta, multibuffer=True)
    except TypeError as exc:
        if "multibuffer" not in str(exc):
            raise
        kernel[grid](*args, **meta)


def launch(x, y, out, config):
    n_elements = x.numel()
    num_blocks = triton.cdiv(n_elements, config.block_size)
    grid = (min(num_blocks, config.num_core),)
    launch_with_optional_multibuffer(
        silu_mul_split_kernel,
        grid,
        (x, y, out, n_elements),
        {
            "NUM_CORE": config.num_core,
            "BLOCK_SIZE": config.block_size,
            "BLOCK_SIZE_SUB": config.block_size_sub,
        },
        config.multibuffer,
    )
    return grid


def benchmark(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeat


def parse_ints(text):
    return [int(item) for item in text.split(",") if item]


def make_configs(num_cores, block_sizes, sub_blocks, multibuffer_values):
    configs = []
    for num_core in num_cores:
        for block_size in block_sizes:
            for block_size_sub in sub_blocks:
                if block_size % block_size_sub != 0:
                    continue
                for multibuffer in multibuffer_values:
                    configs.append(
                        TuneConfig(
                            num_core=num_core,
                            block_size=block_size,
                            block_size_sub=block_size_sub,
                            multibuffer=multibuffer,
                        )
                    )
    return configs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-elements", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--num-cores", default="")
    parser.add_argument("--block-sizes", default="2048,4096,8192")
    parser.add_argument("--sub-blocks", default="512,1024")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--include-no-multibuffer", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = require_npu()
    dtype = getattr(torch, args.dtype)
    detected_core = get_num_vector_core(56)
    num_cores = parse_ints(args.num_cores) if args.num_cores else [detected_core]
    block_sizes = parse_ints(args.block_sizes)
    sub_blocks = parse_ints(args.sub_blocks)
    multibuffer_values = [True]
    if args.include_no_multibuffer:
        multibuffer_values.append(False)

    configs = make_configs(num_cores, block_sizes, sub_blocks, multibuffer_values)
    if not configs:
        raise ValueError("No valid configs after filtering block_size % sub_block == 0.")

    torch.manual_seed(0)
    x = torch.randn(args.n_elements, device=device, dtype=dtype)
    y = torch.randn(args.n_elements, device=device, dtype=dtype)
    out = torch.empty_like(x)
    expected = torch.nn.functional.silu(x) * y

    results = []
    for config in configs:
        out.zero_()
        grid = launch(x, y, out, config)
        synchronize()
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
        ms = benchmark(lambda: launch(x, y, out, config), args.warmup, args.repeat)
        results.append((ms, grid, config))
        print(
            f"time_ms={ms:.4f} grid={grid} num_core={config.num_core} "
            f"block={config.block_size} sub_block={config.block_size_sub} "
            f"multibuffer={config.multibuffer}"
        )

    best_ms, best_grid, best_config = min(results, key=lambda item: item[0])
    print(
        "best "
        f"time_ms={best_ms:.4f} grid={best_grid} "
        f"num_core={best_config.num_core} block={best_config.block_size} "
        f"sub_block={best_config.block_size_sub} "
        f"multibuffer={best_config.multibuffer}"
    )


if __name__ == "__main__":
    main()

