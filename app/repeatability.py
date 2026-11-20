"""平场重复性掩膜（repeatability mask）管线。

任务
====

在同一读出模式（相机 ``INSTRUME``、滤镜 ``FILTER``、全幅尺寸一致）下，
从 3~12 份本服务 ``/prepare`` 生成的成品里选出 3 份，要求三份成品的源曝光
号（``EXPID``）集合两两无交集，并使同一位置在三份成品的 ``COUNTS`` 图中
均不小于 3 的共同有效像元最多；并列时取组内文件名排序后字典序最小的一组。

实现要点
========

* 每份成品仅打开一次做头部校验、``COUNTS`` 图仅**顺序扫描一遍**：
  按数据块字节偏移逐行块直读原始字节，把 ``COUNTS >= 3`` 折叠成 1 bit
  累加进逐像元的位码数组（第 i 份成品对应 bit i），之后不再重读任何成品；
  不使用 numpy memmap，避免热页缓存下被映射页计入 RSS 的不确定性。
* 常驻内存：12 份成品时位码图为 ``uint16``，4096² 仅 32 MiB；
  掩膜输出 ``uint8`` 16 MiB；行块读缓冲（uint8 约 2 MiB）很小，
  无论页缓存冷热，峰值都稳在 256 MiB 容器预算内。
* 三份组合在折叠阶段仅生成一次位码直方图，``O(C(n,3))`` 查表求共同有效
  像元数（n<=12 时最多 220 组），不做逐像元组合遍历。
* 任何成品缺相机/滤镜/完整曝光号、文件截断、找不到有共同有效像元的独立
  组合，都按整批可预期失败处理（HTTP 422）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from astropy.io import fits

from .pipeline import FITS_BLOCK, MAX_DIM, PreparationError, _sanitize_ascii

MIN_PRODUCTS = 3
MAX_PRODUCTS = 12
MIN_COUNTS = 3  # 共同有效像元：三份成品在该位置的 COUNTS 均 >= 3
ROW_CHUNK = 512
MIN_COMMON_PIXELS = 1


@dataclass(frozen=True)
class ProductInfo:
    """单份成品的头信息（数据本身不常驻，逐块折叠后即释放）。"""

    name: str
    shape: tuple[int, int]
    instrume: str
    filter_name: str
    nframes: int
    expids: frozenset[str]


@dataclass(frozen=True)
class RepeatabilityResult:
    """重复性掩膜制备结果。"""

    path: Path
    selected: tuple[str, str, str]
    common_valid_pixels: int


def _safe_product_path(output_dir: Path, filename: str) -> Path:
    """只接受 output_dir 下的纯文件名，杜绝路径穿越。"""
    if not filename or filename in (".", ".."):
        raise PreparationError("成品文件名非法（空名或目录名）")
    if "/" in filename or "\\" in filename or os.path.sep in filename:
        raise PreparationError(f"成品文件名非法（不允许路径分隔符）：{filename}")
    candidate = (output_dir / filename).resolve()
    root = output_dir.resolve()
    if candidate.parent != root:
        raise PreparationError(f"成品文件名非法（越出成品目录）：{filename}")
    return candidate


def _require_card(header: fits.Header, key: str, name: str) -> str:
    if key not in header or header[key] is None:
        raise PreparationError(f"成品 {name} 缺少 {key}，无法参与重复性组合")
    value = str(header[key]).strip()
    if not value:
        raise PreparationError(f"成品 {name} 的 {key} 为空，无法参与重复性组合")
    return value


def _open_product(path: Path):
    """打开一份服务成品并完成结构/头部校验，返回 ``(HDUList, ProductInfo)``。

    这里用 ``memmap=False`` 只解析头部（不实体化任何图像），随后立即关闭；
    COUNTS 的像素数据由 :func:`_scan_counts_from_file` 按字节区间分块直读，
    不使用 numpy memmap——避免热页缓存下被映射页全部计入 RSS 的不确定性，
    使最坏规模的堆内存恒定在预算内。
    """
    name = path.name
    try:
        hdul = fits.open(path, do_not_scale_image_data=True, memmap=False)
    except (OSError, ValueError, fits.VerifyError) as exc:
        raise PreparationError(f"成品 {name} 截断或无法解析：{exc}") from exc

    try:
        if len(hdul) != 2 or not isinstance(hdul[0], fits.PrimaryHDU):
            raise PreparationError(
                f"成品 {name} 结构非法或文件截断：须为 PRIMARY + COUNTS 双 HDU 的服务成品"
            )
        if not isinstance(hdul[1], fits.ImageHDU) or hdul[1].name != "COUNTS":
            raise PreparationError(
                f"成品 {name} 结构非法或文件截断：缺少名为 COUNTS 的图像扩展"
            )
        if isinstance(hdul[0], fits.CompImageHDU) or isinstance(
            hdul[1], fits.CompImageHDU
        ):
            raise PreparationError(f"成品 {name} 含压缩 HDU，不是合法服务成品")
        primary_header = hdul[0].header
        if str(primary_header.get("IMAGTYPE", "")).strip() != "MASTERFLAT":
            raise PreparationError(
                f"成品 {name} 不是本服务生成的主平场（IMAGTYPE 不符）"
            )

        counts_hdu = hdul[1]
        counts_header = counts_hdu.header
        # 形状/位深来自头部，不触发图像实体化。
        try:
            naxis = int(primary_header.get("NAXIS", 0))
            if naxis != 2 or int(counts_header.get("NAXIS", 0)) != 2:
                raise PreparationError(f"成品 {name} 的图像不是二维全幅图")
            height = int(counts_header["NAXIS2"])
            width = int(counts_header["NAXIS1"])
        except (TypeError, KeyError, ValueError) as exc:
            raise PreparationError(
                f"成品 {name} 头部尺寸关键字缺失或非法：{exc}"
            ) from exc
        if not (1 <= height <= MAX_DIM and 1 <= width <= MAX_DIM):
            raise PreparationError(
                f"成品 {name} 尺寸 {width}x{height} 非法，宽高均须在 1~{MAX_DIM} 之间"
            )
        bitpix = int(counts_header.get("BITPIX", 0))
        if bitpix not in (8, 16, 32, 64):
            raise PreparationError(f"成品 {name} 的 COUNTS 不是整数图像（BITPIX={bitpix}）")
        # 服务成品的 COUNTS 不做物理量缩放；出现 BSCALE/BZERO 即视为非本服务成品。
        if "BSCALE" in counts_header or "BZERO" in counts_header:
            raise PreparationError(
                f"成品 {name} 的 COUNTS 含 BSCALE/BZERO，不是合法服务成品"
            )

        # 数据块字节偏移（头部解析后即确定），供分块直读；并确认主图尺寸自洽。
        try:
            dat_loc = int(counts_hdu.fileinfo()["datLoc"])
            primary_shape = tuple(hdul[0].shape)
        except (KeyError, OSError, ValueError) as exc:
            raise PreparationError(
                f"成品 {name} 数据块定位失败或文件已截断：{exc}"
            ) from exc
        if primary_shape != (height, width):
            raise PreparationError(
                f"成品 {name} 的 COUNTS 与主图尺寸不一致或文件已截断"
            )

        instrume = _require_card(primary_header, "INSTRUME", name)
        filter_name = _require_card(primary_header, "FILTER", name)
        nframes = primary_header.get("NFRAMES")
        if not isinstance(nframes, (int, np.integer)) or int(nframes) <= 0:
            raise PreparationError(f"成品 {name} 的 NFRAMES 缺失或非法")
        nframes = int(nframes)

        expids: list[str] = []
        for idx in range(1, nframes + 1):
            key = f"EXPID{idx:02d}"
            if key not in primary_header or primary_header[key] is None:
                raise PreparationError(
                    f"成品 {name} 缺少完整曝光号（{key} 缺失），"
                    "无法确认源曝光无交集"
                )
            value = str(primary_header[key]).strip()
            if not value:
                raise PreparationError(
                    f"成品 {name} 的 {key} 为空，完整曝光号不可用"
                )
            expids.append(value)
        info = ProductInfo(
            name=name,
            shape=(int(height), int(width)),
            instrume=instrume,
            filter_name=filter_name,
            nframes=nframes,
            expids=frozenset(expids),
        )
        scan = _CountsScan(path, dat_loc, height, width, bitpix)
        return info, scan
    finally:
        hdul.close()


# COUNTS BITPIX -> (numpy 磁盘 dtype, 解析后用于比较的整数 dtype)
_COUNTS_DTYPE = {
    8: (np.uint8, np.uint8),
    16: (">i2", np.int16),
    32: (">i4", np.int32),
    64: (">i8", np.int64),
}


@dataclass(frozen=True)
class _CountsScan:
    """COUNTS 数据块的字节定位与解码信息。"""

    path: Path
    dat_loc: int
    height: int
    width: int
    bitpix: int


def _scan_counts_from_file(
    scan: _CountsScan, codes: np.ndarray, bit: int
) -> None:
    """对 COUNTS 仅做一遍顺序扫描，把 ``COUNTS >= 3`` 的位置置上 bit。

    直接按 ``datLoc`` 字节偏移逐行块读取原始字节解码，每次只把
    ``ROW_CHUNK`` 行（uint8 时约 2 MiB）读入堆，不整幅载入、不用 memmap，
    因而无论文件页缓存冷热，常驻堆内存都恒定。任何短读都按截断处理。
    """
    disk_dtype, value_dtype = _COUNTS_DTYPE[scan.bitpix]
    itemsize = np.dtype(disk_dtype).itemsize
    row_bytes = scan.width * itemsize
    mask_bit = np.uint16(1 << bit)
    try:
        with scan.path.open("rb") as handle:
            for r0 in range(0, scan.height, ROW_CHUNK):
                r1 = min(r0 + ROW_CHUNK, scan.height)
                n_rows = r1 - r0
                handle.seek(scan.dat_loc + r0 * row_bytes)
                raw = handle.read(n_rows * row_bytes)
                if len(raw) != n_rows * row_bytes:
                    raise PreparationError(
                        f"成品 {scan.path.name} 的 COUNTS 数据块读取不完整，文件已截断"
                    )
                block = np.frombuffer(raw, dtype=disk_dtype)
                block = block.reshape(n_rows, scan.width).astype(value_dtype, copy=False)
                valid = block >= MIN_COUNTS
                codes[r0:r1] = np.bitwise_or(
                    codes[r0:r1], np.where(valid, mask_bit, 0)
                )
    except PreparationError:
        raise
    except (OSError, ValueError) as exc:
        raise PreparationError(
            f"成品 {scan.path.name} 截断或 COUNTS 数据损坏：{exc}"
        ) from exc


def _triple_common_counts(
    codes: np.ndarray, triples: list[tuple[int, int, int]]
) -> np.ndarray:
    """由位码直方图一次性求所有三元组的共同有效像元数。

    位码中同时含三元组三个 bit 的像元即为共同有效；直方图只枚举实际
    出现过的位码取值，避免逐像元遍历 220 组组合。直方图按行块累加，
    规避 ``np.bincount`` 对 uint16 整图提升为 int64 的瞬时大分配。
    """
    histogram = np.zeros(1 << 16, dtype=np.int64)
    height = codes.shape[0]
    for r0 in range(0, height, ROW_CHUNK):
        histogram += np.bincount(
            codes[r0 : r0 + ROW_CHUNK].reshape(-1), minlength=1 << 16
        )
    present = np.nonzero(histogram)[0].astype(np.uint16)
    bits = np.array(
        [[1 << a, 1 << b, 1 << c] for a, b, c in triples], dtype=np.uint16
    )  # (T, 3)
    required = np.bitwise_or.reduce(bits, axis=1)  # (T,)
    # present: (P,)；逐位码判断是否覆盖每个三元组所需的三个 bit。
    covered = (present[None, :] & required[:, None]) == required[:, None]
    return covered.astype(np.int64) @ histogram[present].astype(np.int64)


def _write_mask_fits(
    mask: np.ndarray,
    header: fits.Header,
    tmp_path: Path,
) -> None:
    """单 HDU uint8 掩膜流式直写，数据块补齐到 2880 字节。"""
    block = header.tostring(sep="", padding=True).encode("ascii")
    with tmp_path.open("wb") as handle:
        handle.write(block)
        np.ascontiguousarray(mask, dtype=np.uint8).tofile(handle)
        pad = (FITS_BLOCK - mask.nbytes % FITS_BLOCK) % FITS_BLOCK
        if pad:
            handle.write(b"\0" * pad)
        handle.flush()
        os.fsync(handle.fileno())


def build_repeatability_mask(
    filenames: list[str],
    output_dir: Path,
    mask_dir: Path,
    work_dir: Path,
) -> RepeatabilityResult:
    """选出三份独立成品并写出共同有效像元掩膜，返回结果描述。

    成功返回 :class:`RepeatabilityResult`；任何输入/数据问题抛
    :class:`PreparationError`（HTTP 422），不留下半成品文件。
    """
    if not (MIN_PRODUCTS <= len(filenames) <= MAX_PRODUCTS):
        raise PreparationError(
            f"成品数量须在 {MIN_PRODUCTS}~{MAX_PRODUCTS} 份之间，"
            f"收到 {len(filenames)} 份"
        )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PreparationError(f"工作目录不可用：{exc}", status_code=500) from exc

    # 候选按文件名排序：三元组按下标字典序枚举，首个最大值即并列时
    # “组内文件名排序后字典序最小”的一组。
    ordered_names = sorted(filenames)
    products: list[ProductInfo] = []
    codes: np.ndarray | None = None

    for index, filename in enumerate(ordered_names):
        path = _safe_product_path(output_dir, filename)
        if not path.is_file():
            raise PreparationError(f"成品不存在或不是普通文件：{filename}")
        info, scan = _open_product(path)
        if codes is None:
            height, width = info.shape
            codes = np.zeros((height, width), dtype=np.uint16)
            common_shape = info.shape
            common_instrume = info.instrume
            common_filter = info.filter_name
        else:
            if info.shape != common_shape:
                raise PreparationError(
                    f"成品 {info.name} 尺寸 {info.shape[1]}x{info.shape[0]} 与首份 "
                    f"{common_shape[1]}x{common_shape[0]} 不一致，候选尺寸须一致"
                )
            if info.instrume != common_instrume:
                raise PreparationError(
                    f"成品 {info.name} 的相机 {info.instrume!r} 与首份 "
                    f"{common_instrume!r} 不一致，候选相机须一致"
                )
            if info.filter_name != common_filter:
                raise PreparationError(
                    f"成品 {info.name} 的滤镜 {info.filter_name!r} 与首份 "
                    f"{common_filter!r} 不一致，候选滤镜须一致"
                )

        if index >= 16:  # 位码图为 uint16，理论上界由 MAX_PRODUCTS=12 提前拦住
            raise PreparationError("候选成品数超出位码容量")
        # 仅顺序扫描一遍 COUNTS（按字节区间分块直读），文件不保持打开。
        _scan_counts_from_file(scan, codes, index)
        products.append(info)

    assert codes is not None and products
    height, width = common_shape  # type: ignore[name-defined]

    # 枚举 EXPID 集合两两无交集的三元组（按下标字典序）。
    triples: list[tuple[int, int, int]] = []
    n = len(products)
    for a in range(n):
        for b in range(a + 1, n):
            if products[a].expids & products[b].expids:
                continue
            for c in range(b + 1, n):
                if products[a].expids & products[c].expids:
                    continue
                if products[b].expids & products[c].expids:
                    continue
                triples.append((a, b, c))

    if not triples:
        raise PreparationError(
            "找不到三份 EXPID 集合两两无交集的独立成品，重复性组合为空"
        )

    common_counts = _triple_common_counts(codes, triples)
    best_pos = int(np.argmax(common_counts))
    best_score = int(common_counts[best_pos])
    if best_score < MIN_COMMON_PIXELS:
        raise PreparationError(
            "所有独立组合的共同有效像元数均为 0（无同一位置三份 COUNTS 均 >= 3）"
        )

    a, b, c = triples[best_pos]
    selected = (products[a].name, products[b].name, products[c].name)
    required_bits = np.uint16((1 << a) | (1 << b) | (1 << c))

    # 逐行块由位码图生成 uint8 掩膜：共同有效填 1，其余填 0。
    mask = np.zeros((height, width), dtype=np.uint8)
    for r0 in range(0, height, ROW_CHUNK):
        r1 = min(r0 + ROW_CHUNK, height)
        valid = (codes[r0:r1] & required_bits) == required_bits
        mask[r0:r1] = valid.astype(np.uint8)
    del codes

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    token = os.urandom(3).hex()
    final_name = f"repeatability_mask_{timestamp}_{token}.fits"
    tmp_path = work_dir / f".{final_name}.tmp"
    final_path = mask_dir / final_name

    primary = fits.PrimaryHDU(data=mask)
    header = primary.header
    header["IMAGTYPE"] = ("REPEATMASK", "Flat repeatability common-valid pixel mask")
    header["INSTRUME"] = (common_instrume, "Camera shared by the selected products")
    header["FILTER"] = (common_filter, "Filter shared by the selected products")
    header["NVALID"] = (best_score, "Common valid pixels (all three COUNTS >= 3)")
    header["CREATED"] = (
        datetime.now(UTC).isoformat(timespec="seconds"),
        "Creation time (UTC)",
    )
    for slot, name in enumerate(selected, start=1):
        header[f"FILE{slot:02d}"] = (
            _sanitize_ascii(name),
            "Selected product filename",
        )
        header[f"EXP{slot:02d}CNT"] = (
            len(products[(a, b, c)[slot - 1]].expids),
            f"Distinct EXPID count of product {slot}",
        )
    header.add_history(
        "Mask pixel = 1 where each selected product's COUNTS extension is >= 3; "
        "the three products have pairwise disjoint source EXPID sets; "
        "the combination maximizes the common valid pixel count, with the "
        "lexicographically smallest sorted filename set on ties."
    )
    header_bytes = header.copy()
    del primary, header

    try:
        _write_mask_fits(mask, header_bytes, tmp_path)
        # 回读校验走 memmap，分块统计，避免再实体化一份 4096² 掩膜。
        with fits.open(tmp_path, memmap=True) as verify:
            if len(verify) != 1 or verify[0].shape != (height, width):
                raise PreparationError("重复性掩膜 FITS 结构校验失败", status_code=500)
            verified = np.asarray(verify[0].data)
            if verified.dtype != np.uint8:
                raise PreparationError("重复性掩膜数据类型校验失败", status_code=500)
            checked_sum = 0
            for r0 in range(0, height, ROW_CHUNK):
                block = np.asarray(verified[r0 : r0 + ROW_CHUNK])
                if np.any(block > 1):
                    raise PreparationError(
                        "重复性掩膜取值校验失败", status_code=500
                    )
                checked_sum += int(block.sum())
            if checked_sum != best_score:
                raise PreparationError("重复性掩膜像元计数校验失败", status_code=500)
        os.replace(tmp_path, final_path)
    except PreparationError:
        tmp_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise PreparationError(f"重复性掩膜写出失败：{exc}", status_code=500) from exc

    return RepeatabilityResult(
        path=final_path,
        selected=selected,
        common_valid_pixels=best_score,
    )
