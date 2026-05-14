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
def inplace_scale_kernel(
    in_out_ptr,
    scale,
    n_elements,
    NUM_CORE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)

    block_id = pid
    while block_id < num_blocks:
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_out_ptr + offsets, mask=mask, other=0.0)
        out = x * scale
        tl.store(in_out_ptr + offsets, out, mask=mask)

        block_id += NUM_CORE


@triton.jit
def out_of_place_scale_kernel(
    x_ptr,
    out_ptr,
    scale,
    n_elements,
    NUM_CORE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)

    block_id = pid
    while block_id < num_blocks:
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        out = x * scale
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


def launch_inplace(x, scale, num_core, block_size, multibuffer):
    n_elements = x.numel()
    num_blocks = triton.cdiv(n_elements, block_size)
    grid = (min(num_blocks, num_core),)
    launch_with_optional_multibuffer(
        inplace_scale_kernel,
        grid,
        (x, scale, n_elements),
        {"NUM_CORE": num_core, "BLOCK_SIZE": block_size},
        multibuffer,
    )
    return grid


def launch_out_of_place(x, out, scale, num_core, block_size, multibuffer):
    n_elements = x.numel()
    num_blocks = triton.cdiv(n_elements, block_size)
    grid = (min(num_blocks, num_core),)
    launch_with_optional_multibuffer(
        out_of_place_scale_kernel,
        grid,
        (x, out, scale, n_elements),
        {"NUM_CORE": num_core, "BLOCK_SIZE": block_size},
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
    parser.add_argument("--mode", choices=("inplace", "out-of-place", "both"), default="both")
    parser.add_argument("--n-elements", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--num-core", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--scale", type=float, default=1.125)
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
    expected = x * args.scale

    if args.mode in ("inplace", "both"):
        in_out = x.clone()
        grid = launch_inplace(in_out, args.scale, num_core, args.block_size, multibuffer)
        synchronize()
        torch.testing.assert_close(in_out, expected, rtol=1e-2, atol=1e-2)
        bench_buf = x.clone()
        ms = benchmark(
            lambda: launch_inplace(
                bench_buf,
                args.scale,
                num_core,
                args.block_size,
                multibuffer,
            ),
            args.warmup,
            args.repeat,
        )
        print(
            f"inplace grid={grid} num_core={num_core} block={args.block_size} "
            f"multibuffer={multibuffer} time_ms={ms:.4f}"
        )

    if args.mode in ("out-of-place", "both"):
        out = torch.empty_like(x)
        grid = launch_out_of_place(
            x,
            out,
            args.scale,
            num_core,
            args.block_size,
            multibuffer,
        )
        synchronize()
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
        ms = benchmark(
            lambda: launch_out_of_place(
                x,
                out,
                args.scale,
                num_core,
                args.block_size,
                multibuffer,
            ),
            args.warmup,
            args.repeat,
        )
        print(
            "out-of-place "
            f"grid={grid} num_core={num_core} block={args.block_size} "
            f"multibuffer={multibuffer} time_ms={ms:.4f}"
        )


if __name__ == "__main__":
    main()

