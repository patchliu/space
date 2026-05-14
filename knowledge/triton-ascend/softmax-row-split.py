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
def softmax_row_split_kernel(
    x_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    STRIDE_M: tl.constexpr,
    NUM_CORE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    logical_groups = tl.cdiv(M, ROWS_PER_PROGRAM)

    group_id = pid
    while group_id < logical_groups:
        for r in tl.static_range(0, ROWS_PER_PROGRAM):
            row = group_id * ROWS_PER_PROGRAM + r
            mask = (row < M) & (offs_n < N)
            x = tl.load(
                x_ptr + row * STRIDE_M + offs_n,
                mask=mask,
                other=-float("inf"),
            )
            x = x - tl.max(x, axis=0)
            numerator = tl.exp(x)
            denominator = tl.sum(numerator, axis=0)
            out = numerator / denominator
            tl.store(out_ptr + row * STRIDE_M + offs_n, out, mask=mask)

        group_id += NUM_CORE


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


def next_power_of_2(value):
    return 1 << (value - 1).bit_length()


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


def launch(x, out, num_core, block_n, rows_per_program, multibuffer):
    m, n = x.shape
    groups = triton.cdiv(m, rows_per_program)
    grid = (min(groups, num_core),)
    launch_with_optional_multibuffer(
        softmax_row_split_kernel,
        grid,
        (x, out),
        {
            "M": m,
            "N": n,
            "STRIDE_M": x.stride(0),
            "NUM_CORE": num_core,
            "BLOCK_N": block_n,
            "ROWS_PER_PROGRAM": rows_per_program,
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
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--num-core", type=int, default=0)
    parser.add_argument("--block-n", type=int, default=0)
    parser.add_argument("--rows-per-program", type=int, default=1)
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
    block_n = args.block_n or next_power_of_2(args.n)
    multibuffer = not args.no_multibuffer

    torch.manual_seed(0)
    x = torch.randn((args.m, args.n), device=device, dtype=dtype)
    out = torch.empty_like(x)
    expected = torch.softmax(x.float(), dim=1).to(dtype)

    grid = launch(x, out, num_core, block_n, args.rows_per_program, multibuffer)
    synchronize()
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    ms = benchmark(
        lambda: launch(x, out, num_core, block_n, args.rows_per_program, multibuffer),
        args.warmup,
        args.repeat,
    )
    print(
        f"grid={grid} num_core={num_core} m={args.m} n={args.n} "
        f"block_n={block_n} rows_per_program={args.rows_per_program} "
        f"multibuffer={multibuffer} time_ms={ms:.4f}"
    )


if __name__ == "__main__":
    main()

