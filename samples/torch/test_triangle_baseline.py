#!/usr/bin/env python3
"""
Test harness comparing ROCm nvdiffrast output against the reference tri.png.
Measures MSE and max pixel difference on triangle pixels.
"""

import sys

import numpy as np
import torch
from PIL import Image

import nvdiffrast.torch as dr


def tensor(*args, **kwargs):
    return torch.tensor(*args, device="cuda", **kwargs)


def main():
    print("=" * 60)
    print("nvdiffrast ROCm Triangle Test")
    print("=" * 60)

    # Load reference image (256x256, RGB, 0-255)
    ref_img = np.array(Image.open("docs/img/tri.png")).astype(np.float32) / 255.0

    # Rasterize triangle (same as triangle.py)
    glctx = dr.RasterizeCudaContext()
    pos = tensor(
        [[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]],
        dtype=torch.float32,
    )
    col = tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=torch.float32)
    tri = tensor([[0, 1, 2]], dtype=torch.int32)

    rast, _ = dr.rasterize(glctx, pos, tri, resolution=[256, 256])
    out, _ = dr.interpolate(col, rast, tri)

    # Flip vertically to match reference
    our_img = out.cpu().numpy()[0, ::-1, :, :]

    # Save our output for visual comparison
    our_np = np.clip(np.rint(our_img * 255), 0, 255).astype(np.uint8)
    Image.fromarray(our_np).save("samples/torch/tri_rocm.png")

    # Find triangle bounds in both images
    ref_mask = ref_img.max(axis=2) > 0.01
    our_mask = our_img.max(axis=2) > 0.01

    ref_ys, ref_xs = np.where(ref_mask)
    our_ys, our_xs = np.where(our_mask)

    print(f"\nReference triangle pixels: {ref_mask.sum()}")
    print(
        f"Reference bounds: x=[{ref_xs.min()}, {ref_xs.max()}], y=[{ref_ys.min()}, {ref_ys.max()}]"
    )

    print(f"Our triangle pixels: {our_mask.sum()}")
    print(
        f"Our bounds: x=[{our_xs.min()}, {our_xs.max()}], y=[{our_ys.min()}, {our_ys.max()}]"
    )

    if ref_mask.any() and our_mask.any():
        # Compare on overlapping pixels
        overlap = ref_mask & our_mask
        diff = our_img[overlap] - ref_img[overlap]
        mse = np.mean(diff**2)
        max_diff = np.max(np.abs(diff))

        print(f"\nOverlap pixels: {overlap.sum()}")
        print(f"MSE on overlap pixels: {mse:.8f}")
        print(f"Max diff on overlap pixels: {max_diff:.6f}")

        # Compare on ALL reference triangle pixels (including non-overlap)
        diff_all = our_img[ref_mask] - ref_img[ref_mask]
        mse_all = np.mean(diff_all**2)
        max_diff_all = np.max(np.abs(diff_all))
        print(f"MSE on all ref pixels: {mse_all:.8f}")
        print(f"Max diff on all ref pixels: {max_diff_all:.6f}")

        # Key pixel comparisons at actual triangle locations
        for name, y, x in [
            ("ref-top-left", ref_ys.min(), ref_xs.min()),
            ("ref-bottom-right", ref_ys.max(), ref_xs.max()),
            ("our-top-left", our_ys.min(), our_xs.min()),
            ("our-bottom-right", our_ys.max(), our_xs.max()),
        ]:
            our_c = our_img[y, x]
            ref_c = ref_img[y, x]
            print(f"\n  {name} ({y},{x}):")
            print(f"    Our: R={our_c[0]:.4f} G={our_c[1]:.4f} B={our_c[2]:.4f}")
            print(f"    Ref: R={ref_c[0]:.4f} G={ref_c[1]:.4f} B={ref_c[2]:.4f}")

        # Check barycentric coords near our triangle center
        cy, cx = (
            int((our_ys.min() + our_ys.max()) / 2),
            int((our_xs.min() + our_xs.max()) / 2),
        )
        # rast coords (before flip): y_rast = 255 - cy
        cy_rast = 255 - cy
        u = rast[0, cy_rast, cx, 0].item()
        v = rast[0, cy_rast, cx, 1].item()
        print(
            f"\n  Barycentric at our center (rast={cy_rast},{cx}): u={u:.4f} v={v:.4f} w={1 - u - v:.4f}"
        )

        # Expected: at triangle center, u≈0.33, v≈0.33, w≈0.34
        print(f"  Expected at center: u≈0.33, v≈0.33, w≈0.34")

        # Check if we pass (MSE < 0.001)
        if mse_all < 0.001:
            print(f"\n*** PASS: MSE={mse_all:.8f} < 0.001 ***")
            return 0
        else:
            print(f"\n*** FAIL: MSE={mse_all:.8f} >= 0.001 ***")
            return 1
    else:
        print("ERROR: No triangle pixels found!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
