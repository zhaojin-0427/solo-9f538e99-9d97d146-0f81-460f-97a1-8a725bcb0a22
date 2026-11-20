"""最坏规模重复性检验内存压测：12 份 4096x4096 成品。

先生成 12 份带独立 EXPID 的成品（输入与对照数据只用于造数/校验，
不计入 repeatability 管线的内存预算），再在**纯净子进程**里运行
``measure_repeatability_only.py`` 测真实峰值 RSS。
用法： python -I scripts/stress_repeatability.py
"""

from __future__ import annotations

import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.pipeline import build_master_flat

N_PRODUCTS = 12
HEIGHT = WIDTH = 4096
FRAMES_PER_PRODUCT = 3


def generate_products(root: Path) -> tuple[Path, Path, Path]:
    products_dir = root / "products"
    work_dir = root / "work"
    inputs_dir = root / "inputs"
    rng = np.random.default_rng(20261120)
    saturation = 5000.0

    for p in range(N_PRODUCTS):
        frame_paths = []
        for f in range(FRAMES_PER_PRODUCT):
            raw = rng.integers(500, 1500, size=(HEIGHT, WIDTH)).astype(np.int16)
            # 每份顶部死区不同，制造共同有效区域差异。
            raw[: 4 + p, :] = 0
            hdu = fits.PrimaryHDU(data=raw)
            hdu.header["INSTRUME"] = "CAM-STRESS"
            hdu.header["FILTER"] = "g"
            hdu.header["EXPID"] = f"EXP-{p:02d}-{f:02d}"
            frame_path = inputs_dir / f"p{p:02d}" / f"frame_{f:02d}.fits"
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            fits.HDUList([hdu]).writeto(frame_path, overwrite=True)
            frame_paths.append(frame_path)
        product = build_master_flat(
            frame_paths,
            [x.name for x in frame_paths],
            saturation,
            output_dir=products_dir,
            work_dir=work_dir,
        )
        final = products_dir / f"stress_product_{p:02d}.fits"
        product.rename(final)
        print(f"[stress] 成品 {p + 1}/{N_PRODUCTS} 就绪：{final.name}")
    return products_dir, root / "masks", work_dir


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="repeat_stress_"))
    print(f"[stress] 根目录 {root}")
    t0 = time.perf_counter()
    products_dir, masks_dir, work_dir = generate_products(root)
    print(f"[stress] 成品生成耗时 {time.perf_counter() - t0:.1f}s")

    # 生成阶段的大数组已出作用域，强制 GC 归还可释放内存，降低 fork 子进程
    # 继承的 RSS 高水位；页缓存等不可回收部分仍可能抬高子进程绝对峰值，
    # 权威数字请在不持大数据的 shell 中直接运行 measure_repeatability_only.py。
    import gc

    gc.collect()

    entry = Path(__file__).resolve().parent / "measure_repeatability_only.py"
    cmd = [
        sys.executable,
        "-I",
        str(entry),
        str(products_dir),
        str(masks_dir),
        str(work_dir),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - t0
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        return proc.returncode
    print(f"[stress] 子进程墙钟耗时 {elapsed:.1f}s")
    print(
        "[stress] 内存判定：fork 子进程会继承父进程高水位 RSS，peak_mib 仅作"
        "参考；请以独立 shell 运行 measure_repeatability_only.py 的 peak_mib 为准。"
    )

    # 在本进程（含测试夹具，RSS 仅供参考）回读掩膜做正确性抽查。
    mask_path = max(masks_dir.glob("repeatability_mask_*.fits"))
    with fits.open(mask_path) as out:
        mask = np.asarray(out[0].data)
        assert mask.shape == (HEIGHT, WIDTH)
        assert set(np.unique(mask).tolist()) <= {0, 1}
        nvalid = int(out[0].header["NVALID"])
        assert int(mask.sum()) == nvalid
        selected = [out[0].header[f"FILE0{i}"] for i in (1, 2, 3)]

    # 用入选三份成品的 COUNTS 直接求暴力对照，不依赖死区的硬编码假设。
    common = np.ones((HEIGHT, WIDTH), dtype=bool)
    for name in selected:
        with fits.open(products_dir / name) as product:
            common &= np.asarray(product[1].data) >= 3
    expected = int(common.sum())
    assert nvalid == expected, (nvalid, expected)
    assert np.array_equal(mask, common.astype(np.uint8))
    print(
        f"[stress] 校验通过：入选 {selected}，共同有效像元 {nvalid}，"
        f"本进程 RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MiB（含夹具，仅供参考）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
