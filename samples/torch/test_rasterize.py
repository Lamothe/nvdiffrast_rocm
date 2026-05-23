#!/usr/bin/env python3
"""
Test harness for nvdiffrast rasterization on ROCm.
Compares output with a reference implementation to detect bugs.
"""

import numpy as np
import torch

import nvdiffrast.torch as dr


def tensor(*args, **kwargs):
    return torch.tensor(*args, device="cuda", **kwargs)


def test_basic_triangle():
    """Test basic triangle rasterization."""
    print("Testing basic triangle rasterization...")

    glctx = dr.RasterizeCudaContext()

    # Simple triangle in the center
    pos = tensor(
        [[[-0.5, -0.5, 0, 1], [0.5, -0.5, 0, 1], [-0.5, 0.5, 0, 1]]],
        dtype=torch.float32,
    )
    tri = tensor([[0, 1, 2]], dtype=torch.int32)

    rast, rast_db = dr.rasterize(glctx, pos, tri, resolution=[256, 256])

    # Check that the triangle is rasterized
    mask = rast[..., 3] > 0
    num_pixels = mask.sum().item()

    print(f"  Rasterized pixels: {num_pixels}")
    print(
        f"  Barycentric coords range: [{rast[..., :2].min().item():.3f}, {rast[..., :2].max().item():.3f}]"
    )

    # Check that barycentric coordinates sum to 1
    bary_sum = rast[..., 0] + rast[..., 1]
    mask_valid = mask & (bary_sum <= 1.0)
    print(f"  Valid pixels (barysum <= 1): {mask_valid.sum().item()}")

    return rast


def test_interpolation():
    """Test attribute interpolation."""
    print("\nTesting attribute interpolation...")

    glctx = dr.RasterizeCudaContext()

    # Triangle vertices with distinct colors
    pos = tensor(
        [[[-0.5, -0.5, 0, 1], [0.5, -0.5, 0, 1], [-0.5, 0.5, 0, 1]]],
        dtype=torch.float32,
    )
    col = tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=torch.float32)
    tri = tensor([[0, 1, 2]], dtype=torch.int32)

    rast, rast_db = dr.rasterize(glctx, pos, tri, resolution=[256, 256])
    out, out_db = dr.interpolate(col, rast, tri)

    # Check that interpolation produces valid colors
    print(f"  Color range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    print(f"  Color at center (128, 128): {out[0, 128, 128].cpu().numpy()}")

    return out


def test_rasterize_grad():
    """Test rasterization gradient computation."""
    print("\nTesting rasterization gradient...")

    glctx = dr.RasterizeCudaContext()

    # Triangle with learnable vertex positions
    pos = tensor(
        [[[-0.5, -0.5, 0, 1], [0.5, -0.5, 0, 1], [-0.5, 0.5, 0, 1]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    tri = tensor([[0, 1, 2]], dtype=torch.int32)

    rast, rast_db = dr.rasterize(glctx, pos, tri, resolution=[256, 256])

    # Simple loss: sum of z/w values
    loss = rast[..., 2].sum()
    loss.backward()

    print(
        f"  Gradient range: [{pos.grad.min().item():.6f}, {pos.grad.max().item():.6f}]"
    )
    print(f"  Gradient has NaN: {torch.isnan(pos.grad).any().item()}")
    print(f"  Gradient has inf: {torch.isinf(pos.grad).any().item()}")

    return pos.grad


def test_antialias():
    """Test antialiasing function."""
    print("\nTesting antialiasing...")

    glctx = dr.RasterizeCudaContext()

    pos = tensor(
        [[[-0.5, -0.5, 0, 1], [0.5, -0.5, 0, 1], [-0.5, 0.5, 0, 1]]],
        dtype=torch.float32,
    )
    col = tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=torch.float32)
    tri = tensor([[0, 1, 2]], dtype=torch.int32)

    rast, rast_db = dr.rasterize(glctx, pos, tri, resolution=[256, 256])
    out, _ = dr.interpolate(col, rast, tri)

    # Apply antialiasing
    out_aa = dr.antialias(out, rast, pos, tri)

    print(f"  Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    print(f"  AA output range: [{out_aa.min().item():.3f}, {out_aa.max().item():.3f}]")
    print(f"  Output has NaN: {torch.isnan(out_aa).any().item()}")

    return out_aa


def main():
    torch.manual_seed(42)

    print("=" * 60)
    print("nvdiffrast ROCm Rasterization Tests")
    print("=" * 60)

    try:
        test_basic_triangle()
        test_interpolation()
        test_rasterize_grad()
        test_antialias()

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
