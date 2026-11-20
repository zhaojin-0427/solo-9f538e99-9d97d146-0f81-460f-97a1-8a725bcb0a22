"""只测 repeatability 管线峰值内存的纯净子进程入口。

不保留任何额外数组；每份成品的 COUNTS 仅顺序扫描一遍。
用法： python -I scripts/measure_repeatability_only.py <成品目录> <掩膜目录> <工作目录>
"""

from __future__ import annotations

import resource
import site
import sys
import time
from pathlib import Path

# 容器内依赖位于系统 site-packages，``-I`` 可直接导入；
# 本地若依赖装在 user-site，``-I`` 会忽略它，这里显式补回以便压测。
for _candidate in site.getusersitepackages(), *site.getsitepackages():
    if _candidate and _candidate not in sys.path:
        sys.path.append(_candidate)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.repeatability import build_repeatability_mask


def main() -> int:
    products_dir, masks_dir, work_dir = (
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        Path(sys.argv[3]),
    )
    # 记录解释器启动后的 RSS 基线。注意：经 fork 继承而来的子进程会带上
    # 父进程的高水位 ru_maxrss，因此绝对 peak_mib 在压测父进程持大数据时
    # 会被抬高；管线自身的真实占用看相对基线的增量 delta_mib。
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    names = sorted(p.name for p in products_dir.glob("*.fits"))
    t0 = time.perf_counter()
    result = build_repeatability_mask(
        filenames=names,
        output_dir=products_dir,
        mask_dir=masks_dir,
        work_dir=work_dir,
    )
    elapsed = time.perf_counter() - t0
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(
        f"RESULT products={len(names)} elapsed={elapsed:.1f}s "
        f"peak_mib={peak_mib:.1f} delta_mib={peak_mib - baseline:.1f} "
        f"baseline_mib={baseline:.1f} "
        f"common_valid={result.common_valid_pixels} mask={result.path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
