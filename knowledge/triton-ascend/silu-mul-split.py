import argparse
import time

import torch
import triton
import triton.language as tl

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


@triton.jit
def silu_mul_naive_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x / (1.0 + tl.exp(-x)) * y
    tl.store(out_ptr + offsets, out, mask=mask)


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


def launch_naive(x, y, out, block_size, multibuffer):
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, block_size),)
    launch_with_optional_multibuffer(
        silu_mul_naive_kernel,
        grid,
        (x, y, out, n_elements),
        {"BLOCK_SIZE": block_size},
        multibuffer,
    )
    return grid


def launch_split(x, y, out, num_core, block_size, block_size_sub, multibuffer):
    n_elements = x.numel()
    num_blocks = triton.cdiv(n_elements, block_size)
    grid = (min(num_blocks, num_core),)
    launch_with_optional_multibuffer(
        silu_mul_split_kernel,
        grid,
        (x, y, out, n_elements),
        {
            "NUM_CORE": num_core,
            "BLOCK_SIZE": block_size,
            "BLOCK_SIZE_SUB": block_size_sub,
        },
        multibuffer,
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("naive", "split", "both"), default="both")
    parser.add_argument("--n-elements", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--num-core", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--block-size-sub", type=int, default=1024)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--no-multibuffer", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = require_npu()
    dtype = getattr(torch, args.dtype)
    num_core = args.num_core or get_num_vector_core(56)
    multibuffer = not args.no_multibuffer

    torch.manual_seed(0)
    x = torch.randn(args.n_elements, device=device, dtype=dtype)
    y = torch.randn(args.n_elements, device=device, dtype=dtype)
    expected = torch.nn.functional.silu(x) * y

    if args.mode in ("naive", "both"):
        out = torch.empty_like(x)
        grid = launch_naive(x, y, out, args.block_size, multibuffer)
        synchronize()
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
        ms = benchmark(
            lambda: launch_naive(x, y, out, args.block_size, multibuffer),
            args.warmup,
            args.repeat,
        )
        print(f"naive grid={grid} block={args.block_size} time_ms={ms:.4f}")

    if args.mode in ("split", "both"):
        out = torch.empty_like(x)
        grid = launch_split(
            x,
            y,
            out,
            num_core,
            args.block_size,
            args.block_size_sub,
            multibuffer,
        )
        synchronize()
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
        ms = benchmark(
            lambda: launch_split(
                x,
                y,
                out,
                num_core,
                args.block_size,
                args.block_size_sub,
                multibuffer,
            ),
            args.warmup,
            args.repeat,
        )
        print(
            "split "
            f"grid={grid} num_core={num_core} block={args.block_size} "
            f"sub_block={args.block_size_sub} multibuffer={multibuffer} "
            f"time_ms={ms:.4f}"
        )


if __name__ == "__main__":
    main()

