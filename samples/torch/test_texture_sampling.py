#!/usr/bin/env python3
"""
Test texture sampling with mipmapping on ROCm.
Exercises the exact path: dr.interpolate(uv, rast, tri, rast_db=rast_db, diff_attrs='all')
                          → dr.texture(tex, texc, texd, filter_mode='linear-mipmap-linear')
"""

import sys
import numpy as np
import torch
from PIL import Image

import nvdiffrast.torch as dr


def tensor(*args, **kwargs):
    return torch.tensor(*args, device="cuda", **kwargs)


def create_checkerboard(h, w, size=8):
    """Create a checkerboard texture."""
    tex = torch.zeros(h, w, 3, dtype=torch.float32, device="cuda")
    for y in range(h):
        for x in range(w):
            if ((x // size) + (y // size)) % 2 == 0:
                tex[y, x] = [1.0, 0.0, 0.0]  # red
            else:
                tex[y, x] = [0.0, 1.0, 0.0]  # green
    return tex


def test_uv_interpolation():
    """Test that UV interpolation produces correct results."""
    print("=" * 60)
    print("Test 1: UV Coordinate Interpolation")
    print("=" * 60)

    # Simple triangle covering most of the screen
    pos = tensor([
        [[-0.95, -0.95, 0, 1],
         [ 0.95, -0.95, 0, 1],
         [-0.95,  0.95, 0, 1]]
    ], dtype=torch.float32)

    # UV coords: bottom-left=(0,0), bottom-right=(1,0), top-left=(0,1)
    uv = tensor([
        [[0.0, 0.0],
         [1.0, 0.0],
         [0.0, 1.0]]
    ], dtype=torch.float32)

    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    glctx = dr.RasterizeCudaContext()

    rast, rast_db = dr.rasterize(glctx, pos, tri, [512, 512])

    # Interpolate UVs with derivatives
    texc, texd = dr.interpolate(uv, rast, tri, rast_db=rast_db, diff_attrs='all')

    mask = rast[..., -1:] > 0
    texc_masked = texc[mask]
    texd_masked = texd[mask]

    print(f"  texc range: [{texc_masked.min():.4f}, {texc_masked.max():.4f}]")
    print(f"  texc mean:  [{texc_masked[:, 0].mean():.4f}, {texc_masked[:, 1].mean():.4f}]")
    print(f"  texd range: [{texd_masked.min():.4f}, {texd_masked.max():.4f}]")
    print(f"  texd mean:  [{texd_masked[:, 0].mean():.6f}, {texd_masked[:, 1].mean():.6f}, "
          f"{texd_masked[:, 2].mean():.6f}, {texd_masked[:, 3].mean():.6f}]")
    print(f"  texd NaN: {torch.isnan(texd_masked).any().item()}")
    print(f"  texd Inf: {torch.isinf(texd_masked).any().item()}")

    # For a triangle covering the screen, UV should span [0,1] and mean ~[0.33, 0.33]
    u_mean = texc_masked[:, 0].mean().item()
    v_mean = texc_masked[:, 1].mean().item()
    print(f"  Expected UV mean ~[0.33, 0.33], got [{u_mean:.4f}, {v_mean:.4f}]")

    if abs(u_mean - 0.33) > 0.1 or abs(v_mean - 0.33) > 0.1:
        print("  *** UV means look wrong! ***")

    return texc, texd, mask


def test_texture_sampling(texc, texd, mask):
    """Test texture sampling with mipmapping."""
    print("\n" + "=" * 60)
    print("Test 2: Texture Sampling with Mipmapping")
    print("=" * 60)

    tex = create_checkerboard(256, 256, size=32)

    # Sample with mipmapping (the exact path used by TRELLIS.2)
    out = dr.texture(
        tex.unsqueeze(0),
        texc,
        texd,
        filter_mode='linear-mipmap-linear',
        boundary_mode='clamp'
    )[0]

    out_np = np.clip(out.cpu().numpy() * 255, 0, 255).astype(np.uint8)
    Image.fromarray(out_np).save("samples/torch/texture_sampling_test.png")

    out_masked = out[mask]
    has_nan = torch.isnan(out_masked).any().item()
    has_inf = torch.isinf(out_masked).any().item()

    print(f"  Output range: [{out_masked.min():.4f}, {out_masked.max():.4f}]")
    print(f"  Output mean:  [{out_masked[:, 0].mean():.4f}, {out_masked[:, 1].mean():.4f}, {out_masked[:, 2].mean():.4f}]")
    print(f"  NaN: {has_nan}, Inf: {has_inf}")

    # Check for red and green pixels (checkerboard should have both)
    red = ((out_masked[:, 0] > 0.5) & (out_masked[:, 1] < 0.5)).sum().item()
    green = ((out_masked[:, 0] < 0.5) & (out_masked[:, 1] > 0.5)).sum().item()
    total = out_masked.shape[0]
    print(f"  Red pixels: {red}/{total} ({100*red/total:.1f}%)")
    print(f"  Green pixels: {green}/{total} ({100*green/total:.1f}%)")

    # For a checkerboard with mipmapping, we should see mostly grey (blended red+green)
    # with some red/green near edges. Pure garbage would look random.
    grey = ((out_masked[:, 0] > 0.3) & (out_masked[:, 0] < 0.7) &
            (out_masked[:, 1] > 0.3) & (out_masked[:, 1] < 0.7)).sum().item()
    print(f"  Grey pixels (blended): {grey}/{total} ({100*grey/total:.1f}%)")

    return out


def test_simple_texture():
    """Test texture sampling with a simple known pattern."""
    print("\n" + "=" * 60)
    print("Test 3: Simple Gradient Texture")
    print("=" * 60)

    # Create a simple gradient texture
    h, w = 256, 256
    tex = torch.zeros(h, w, 3, dtype=torch.float32, device="cuda")
    for y in range(h):
        for x in range(w):
            tex[y, x] = [x / (w - 1), y / (h - 1), 0.5]

    # Quad covering the screen
    pos = tensor([
        [[-1, -1, 0, 1],
         [ 1, -1, 0, 1],
         [-1,  1, 0, 1],
         [ 1,  1, 0, 1]]
    ], dtype=torch.float32)

    uv = tensor([
        [[0, 0],
         [1, 0],
         [0, 1],
         [1, 1]]
    ], dtype=torch.float32)

    tri = tensor([[0, 1, 2], [2, 1, 3]], dtype=torch.int32)
    glctx = dr.RasterizeCudaContext()

    rast, rast_db = dr.rasterize(glctx, pos, tri, [512, 512])
    texc, texd = dr.interpolate(uv, rast, tri, rast_db=rast_db, diff_attrs='all')

    # Sample with mipmapping
    out = dr.texture(
        tex.unsqueeze(0),
        texc,
        texd,
        filter_mode='linear-mipmap-linear',
        boundary_mode='clamp'
    )[0]

    out_np = np.clip(out.cpu().numpy() * 255, 0, 255).astype(np.uint8)
    Image.fromarray(out_np).save("samples/torch/gradient_test.png")

    # Check corners: top-left should be ~[0,0,0.5], bottom-right should be ~[1,1,0.5]
    mask = rast[..., -1:] > 0
    tl = out[64, 64].cpu().numpy()  # Top-left area
    br = out[448, 448].cpu().numpy()  # Bottom-right area
    print(f"  Top-left sample:  [{tl[0]:.4f}, {tl[1]:.4f}, {tl[2]:.4f}] (expected ~[0.13, 0.13, 0.5])")
    print(f"  Bottom-right sample: [{br[0]:.4f}, {br[1]:.4f}, {br[2]:.4f}] (expected ~[0.88, 0.88, 0.5])")

    has_nan = torch.isnan(out).any().item()
    has_inf = torch.isinf(out).any().item()
    print(f"  NaN: {has_nan}, Inf: {has_inf}")

    if has_nan or has_inf:
        print("  *** FAIL: Output contains NaN/Inf ***")
        return False

    return True


def test_without_mipmapping():
    """Test texture sampling WITHOUT mipmapping to isolate the issue."""
    print("\n" + "=" * 60)
    print("Test 4: Texture Sampling WITHOUT Mipmapping (linear only)")
    print("=" * 60)

    tex = create_checkerboard(256, 256, size=32)

    pos = tensor([
        [[-0.95, -0.95, 0, 1],
         [ 0.95, -0.95, 0, 1],
         [-0.95,  0.95, 0, 1]]
    ], dtype=torch.float32)

    uv = tensor([
        [[0.0, 0.0],
         [1.0, 0.0],
         [0.0, 1.0]]
    ], dtype=torch.float32)

    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    glctx = dr.RasterizeCudaContext()

    rast, rast_db = dr.rasterize(glctx, pos, tri, [512, 512])
    texc, texd = dr.interpolate(uv, rast, tri, rast_db=rast_db, diff_attrs='all')

    # Sample WITHOUT mipmapping
    out = dr.texture(
        tex.unsqueeze(0),
        texc,
        filter_mode='linear',
        boundary_mode='clamp'
    )[0]

    out_np = np.clip(out.cpu().numpy() * 255, 0, 255).astype(np.uint8)
    Image.fromarray(out_np).save("samples/torch/no_mipmap_test.png")

    mask = rast[..., -1:] > 0
    out_masked = out[mask]
    has_nan = torch.isnan(out_masked).any().item()
    has_inf = torch.isinf(out_masked).any().item()

    print(f"  Output range: [{out_masked.min():.4f}, {out_masked.max():.4f}]")
    print(f"  NaN: {has_nan}, Inf: {has_inf}")

    red = ((out_masked[:, 0] > 0.5) & (out_masked[:, 1] < 0.5)).sum().item()
    green = ((out_masked[:, 0] < 0.5) & (out_masked[:, 1] > 0.5)).sum().item()
    total = out_masked.shape[0]
    print(f"  Red pixels: {red}/{total} ({100*red/total:.1f}%)")
    print(f"  Green pixels: {green}/{total} ({100*green/total:.1f}%)")

    return out


def main():
    texc, texd, mask = test_uv_interpolation()
    test_texture_sampling(texc, texd, mask)
    test_simple_texture()
    test_without_mipmapping()

    print("\n" + "=" * 60)
    print("Saved test images to samples/torch/")
    print("=" * 60)


if __name__ == "__main__":
    main()
