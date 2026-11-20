"""repeatability 管线层测试：选组规则、掩膜正确性与各类整批失败。"""

from __future__ import annotations

import io
import warnings

import numpy as np
import pytest
from astropy.io import fits

from app.repeatability import (
    MIN_COUNTS,
    PreparationError,
    build_repeatability_mask,
)


def _run(tmp_path, products_dir, names):
    return build_repeatability_mask(
        filenames=names,
        output_dir=products_dir,
        mask_dir=tmp_path / "masks",
        work_dir=tmp_path / "work",
    )


def _expected_common(products_dir, names, shape):
    """对照实现：逐份读取 COUNTS，直接求所有独立三元组的共同有效像元。"""
    valid_masks = {}
    expid_sets = {}
    for name in names:
        with fits.open(products_dir / name) as hdul:
            counts = np.asarray(hdul[1].data)
            valid_masks[name] = counts >= MIN_COUNTS
            nframes = hdul[0].header["NFRAMES"]
            expid_sets[name] = {
                str(hdul[0].header[f"EXPID{i:02d}"]).strip()
                for i in range(1, nframes + 1)
            }
    best, best_names = 0, None
    ordered = sorted(names)
    for a in range(len(ordered)):
        for b in range(a + 1, len(ordered)):
            if expid_sets[ordered[a]] & expid_sets[ordered[b]]:
                continue
            for c in range(b + 1, len(ordered)):
                trio = (ordered[a], ordered[b], ordered[c])
                if expid_sets[trio[0]] & expid_sets[trio[2]]:
                    continue
                if expid_sets[trio[1]] & expid_sets[trio[2]]:
                    continue
                common = (
                    valid_masks[trio[0]]
                    & valid_masks[trio[1]]
                    & valid_masks[trio[2]]
                )
                score = int(common.sum())
                if score > best:
                    best, best_names = score, trio
    return best, best_names


def test_happy_path_selects_disjoint_trio_and_writes_mask(
    tmp_path, product_factory
):
    products_dir = tmp_path / "products"
    p1 = product_factory(
        "p1.fits", ["E1", "E2", "E3"], dead_rows=2, products_dir=products_dir
    )
    p2 = product_factory(
        "p2.fits", ["E4", "E5", "E6"], dead_rows=4, products_dir=products_dir
    )
    p3 = product_factory(
        "p3.fits", ["E7", "E8", "E9"], dead_rows=6, products_dir=products_dir
    )
    names = [p1.name, p2.name, p3.name]

    result = _run(tmp_path, products_dir, names)

    assert result.selected == tuple(sorted(names))
    with fits.open(result.path) as out:
        assert len(out) == 1
        mask = np.asarray(out[0].data)
        h = out[0].header
        assert mask.dtype == np.uint8
        assert set(np.unique(mask).tolist()) <= {0, 1}
        assert h["IMAGTYPE"] == "REPEATMASK"
        assert h["NVALID"] == result.common_valid_pixels
        assert h["FILE01"] == result.selected[0]
        assert h["FILE02"] == result.selected[1]
        assert h["FILE03"] == result.selected[2]
        assert h["INSTRUME"] == "CAM-A"
        assert h["FILTER"] == "g"
    expected_score, expected_trio = _expected_common(
        products_dir, names, mask.shape
    )
    assert result.common_valid_pixels == expected_score
    assert result.selected == expected_trio
    assert int(mask.sum()) == expected_score
    # 顶部 6 行在 p3 全无效：掩膜必须为 0；其余行全为 1。
    assert mask[:6, :].sum() == 0
    assert mask[6:, :].sum() == (40 - 6) * 32


def test_picks_maximum_common_pixel_combination(tmp_path, product_factory):
    products_dir = tmp_path / "products"
    # p4 的死区最大；最优组合应避开它。
    p1 = product_factory(
        "p1.fits", ["A1", "A2", "A3"], dead_rows=2, products_dir=products_dir
    )
    p2 = product_factory(
        "p2.fits", ["B1", "B2", "B3"], dead_rows=2, products_dir=products_dir
    )
    p3 = product_factory(
        "p3.fits", ["C1", "C2", "C3"], dead_rows=2, products_dir=products_dir
    )
    p4 = product_factory(
        "p4.fits", ["D1", "D2", "D3"], dead_rows=10, products_dir=products_dir
    )
    names = [p4.name, p2.name, p1.name, p3.name]  # 故意乱序提交

    result = _run(tmp_path, products_dir, names)

    assert "p4.fits" not in result.selected
    assert result.selected == ("p1.fits", "p2.fits", "p3.fits")
    expected_score, expected_trio = _expected_common(
        products_dir, names, (40, 32)
    )
    assert result.selected == expected_trio
    assert result.common_valid_pixels == expected_score


def test_tie_breaks_by_lexicographically_smallest_sorted_filenames(
    tmp_path, product_factory
):
    products_dir = tmp_path / "products"
    # 四份完全同构（同死区、同 EXPID 数量、互不相交）：所有三元组同分。
    names = []
    for name, prefix in [
        ("zeta.fits", "Z"),
        ("alpha.fits", "A"),
        ("mid.fits", "M"),
        ("beta.fits", "B"),
    ]:
        p = product_factory(
            name,
            [f"{prefix}1", f"{prefix}2", f"{prefix}3"],
            dead_rows=3,
            products_dir=products_dir,
        )
        names.append(p.name)

    result = _run(tmp_path, products_dir, names)

    assert result.selected == ("alpha.fits", "beta.fits", "mid.fits")


def test_shared_expid_combination_is_excluded(tmp_path, product_factory):
    products_dir = tmp_path / "products"
    p1 = product_factory(
        "p1.fits", ["E1", "E2", "E3"], products_dir=products_dir
    )
    p2 = product_factory(
        "p2.fits", ["E3", "E4", "E5"], products_dir=products_dir
    )  # 与 p1 共用 E3
    p3 = product_factory(
        "p3.fits", ["E6", "E7", "E8"], products_dir=products_dir
    )
    with pytest.raises(PreparationError, match="无交集"):
        _run(tmp_path, products_dir, [p1.name, p2.name, p3.name])


def test_partial_disjoint_set_still_finds_working_trio(tmp_path, product_factory):
    products_dir = tmp_path / "products"
    # p1/p2 相交（共用 E2），但 p1/p3/p4 构成独立三元组。
    p1 = product_factory(
        "p1.fits", ["E1", "E2", "E10"], dead_rows=2, products_dir=products_dir
    )
    p2 = product_factory(
        "p2.fits", ["E2", "E9", "E11"], dead_rows=2, products_dir=products_dir
    )
    p3 = product_factory(
        "p3.fits", ["E3", "E4", "E12"], dead_rows=2, products_dir=products_dir
    )
    p4 = product_factory(
        "p4.fits", ["E5", "E6", "E13"], dead_rows=2, products_dir=products_dir
    )
    result = _run(tmp_path, products_dir, [p1.name, p2.name, p3.name, p4.name])
    assert "p2.fits" not in result.selected
    assert result.selected == ("p1.fits", "p3.fits", "p4.fits")


def test_missing_expid_product_rejected(tmp_path, product_factory):
    products_dir = tmp_path / "products"
    p1 = product_factory(
        "p1.fits", ["E1", "E2", "E3"], products_dir=products_dir
    )
    p2 = product_factory(
        "p2.fits", ["E4", "E5", "E6"], products_dir=products_dir
    )
    p3 = product_factory(
        "p3.fits", [None, "E7", "E8"], products_dir=products_dir
    )  # 首帧缺 EXPID
    with pytest.raises(PreparationError, match="EXPID01|曝光号"):
        _run(tmp_path, products_dir, [p1.name, p2.name, p3.name])


def test_missing_instrume_rejected(tmp_path, product_factory):
    products_dir = tmp_path / "products"
    p1 = product_factory(
        "p1.fits", ["E1", "E2", "E3"], products_dir=products_dir
    )
    p2 = product_factory(
        "p2.fits", ["E4", "E5", "E6"], products_dir=products_dir
    )
    p3 = product_factory(
        "p3.fits", ["E7", "E8", "E9"], products_dir=products_dir
    )
    with fits.open(p1, mode="update") as hdul:
        del hdul[0].header["INSTRUME"]
    with pytest.raises(PreparationError, match="INSTRUME|相机"):
        _run(tmp_path, products_dir, [p1.name, p2.name, p3.name])


def test_missing_filter_rejected(tmp_path, product_factory):
    products_dir = tmp_path / "products"
    p1 = product_factory(
        "p1.fits", ["E1", "E2", "E3"], products_dir=products_dir
    )
    p2 = product_factory(
        "p2.fits", ["E4", "E5", "E6"], products_dir=products_dir
    )
    p3 = product_factory(
        "p3.fits", ["E7", "E8", "E9"], products_dir=products_dir
    )
    with fits.open(p2, mode="update") as hdul:
        del hdul[0].header["FILTER"]
    with pytest.raises(PreparationError, match="FILTER|滤镜"):
        _run(tmp_path, products_dir, [p1.name, p2.name, p3.name])


def test_instrument_mismatch_rejected(tmp_path, product_factory):
    products_dir = tmp_path / "products"
    p1 = product_factory(
        "p1.fits", ["E1", "E2", "E3"], products_dir=products_dir
    )
    p2 = product_factory(
        "p2.fits",
        ["E4", "E5", "E6"],
        instrume="CAM-B",
        products_dir=products_dir,
    )
    p3 = product_factory(
        "p3.fits", ["E7", "E8", "E9"], products_dir=products_dir
    )
    with pytest.raises(PreparationError, match="相机"):
        _run(tmp_path, products_dir, [p1.name, p2.name, p3.name])


def test_filter_mismatch_rejected(tmp_path, product_factory):
    products_dir = tmp_path / "products"
    p1 = product_factory(
        "p1.fits", ["E1", "E2", "E3"], products_dir=products_dir
    )
    p2 = product_factory(
        "p2.fits",
        ["E4", "E5", "E6"],
        filter_name="r",
        products_dir=products_dir,
    )
    p3 = product_factory(
        "p3.fits", ["E7", "E8", "E9"], products_dir=products_dir
    )
    with pytest.raises(PreparationError, match="滤镜"):
        _run(tmp_path, products_dir, [p1.name, p2.name, p3.name])


def test_shape_mismatch_rejected(tmp_path, product_factory):
    products_dir = tmp_path / "products"
    p1 = product_factory(
        "p1.fits",
        ["E1", "E2", "E3"],
        height=40,
        width=32,
        products_dir=products_dir,
    )
    p2 = product_factory(
        "p2.fits",
        ["E4", "E5", "E6"],
        height=24,
        width=32,
        products_dir=products_dir,
    )
    p3 = product_factory(
        "p3.fits",
        ["E7", "E8", "E9"],
        height=40,
        width=32,
        products_dir=products_dir,
    )
    with pytest.raises(PreparationError, match="尺寸"):
        _run(tmp_path, products_dir, [p1.name, p2.name, p3.name])


def test_truncated_product_rejected(tmp_path, product_factory):
    products_dir = tmp_path / "products"
    p1 = product_factory(
        "p1.fits", ["E1", "E2", "E3"], products_dir=products_dir
    )
    p2 = product_factory(
        "p2.fits", ["E4", "E5", "E6"], products_dir=products_dir
    )
    p3 = product_factory(
        "p3.fits", ["E7", "E8", "E9"], products_dir=products_dir
    )
    blob = p2.read_bytes()
    p2.write_bytes(blob[: len(blob) // 2])
    with pytest.raises(PreparationError, match="截断|损坏|无法解析|结构非法"):
        _run(tmp_path, products_dir, [p1.name, p2.name, p3.name])


def test_missing_file_rejected(tmp_path, product_factory):
    products_dir = tmp_path / "products"
    p1 = product_factory(
        "p1.fits", ["E1", "E2", "E3"], products_dir=products_dir
    )
    p2 = product_factory(
        "p2.fits", ["E4", "E5", "E6"], products_dir=products_dir
    )
    with pytest.raises(PreparationError, match="不存在"):
        _run(tmp_path, products_dir, [p1.name, p2.name, "ghost.fits"])


def test_path_traversal_rejected(tmp_path, product_factory):
    products_dir = tmp_path / "products"
    p1 = product_factory(
        "p1.fits", ["E1", "E2", "E3"], products_dir=products_dir
    )
    p2 = product_factory(
        "p2.fits", ["E4", "E5", "E6"], products_dir=products_dir
    )
    for evil in ["../p1.fits", "/etc/passwd", "sub/p1.fits", "..\\p1.fits"]:
        with pytest.raises(PreparationError, match="文件名非法"):
            _run(tmp_path, products_dir, [p1.name, p2.name, evil])


def test_no_common_valid_pixels_rejected(tmp_path, product_factory):
    products_dir = tmp_path / "products"
    # 每份 39/40 行为死区（COUNTS=0），再把各自唯一的有效行放到不同位置：
    # 三份有效行互不重叠，任何组合共同有效像元都是 0。
    paths = []
    for i in range(3):
        p = product_factory(
            f"p{i + 1}.fits",
            [f"E{i * 3 + 1}", f"E{i * 3 + 2}", f"E{i * 3 + 3}"],
            dead_rows=39,
            products_dir=products_dir,
        )
        paths.append(p)
    valid_row = {0: 39, 1: 38, 2: 37}
    for i, p in enumerate(paths):
        with fits.open(p, mode="update") as hdul:
            counts = hdul[1].data
            counts[:] = 0
            counts[valid_row[i], :] = 3  # 每份只在互不相同的一行有效
    with pytest.raises(PreparationError, match="共同有效像元"):
        _run(tmp_path, products_dir, [p.name for p in paths])


def test_non_service_product_rejected(tmp_path, product_factory):
    products_dir = tmp_path / "products"
    p1 = product_factory(
        "p1.fits", ["E1", "E2", "E3"], products_dir=products_dir
    )
    p2 = product_factory(
        "p2.fits", ["E4", "E5", "E6"], products_dir=products_dir
    )
    p3 = product_factory(
        "p3.fits", ["E7", "E8", "E9"], products_dir=products_dir
    )
    with fits.open(p1, mode="update") as hdul:
        hdul[0].header["IMAGTYPE"] = "SOMETHING_ELSE"
    with pytest.raises(PreparationError, match="本服务"):
        _run(tmp_path, products_dir, [p1.name, p2.name, p3.name])


def test_count_limits(tmp_path, product_factory):
    products_dir = tmp_path / "products"
    names = []
    for i in range(2):
        p = product_factory(
            f"p{i}.fits",
            [f"E{i * 3 + 1}", f"E{i * 3 + 2}", f"E{i * 3 + 3}"],
            products_dir=products_dir,
        )
        names.append(p.name)
    with pytest.raises(PreparationError, match="成品数量"):
        _run(tmp_path, products_dir, names)


def test_prepare_still_works_without_expid(tmp_path, product_factory):
    # /prepare 原规则：缺 EXPID 仍可制备，只是头部不写 EXPID01。
    p = product_factory("noexp.fits", [None, None, None])
    with fits.open(p) as out:
        assert out[0].header["IMAGTYPE"] == "MASTERFLAT"
        assert "EXPID01" not in out[0].header
        assert out[0].header["INSTRUME"] == "CAM-A"
        assert out[0].header["FILTER"] == "g"


def test_prepare_propagates_instrume_filter_and_expids(tmp_path, product_factory):
    p = product_factory("meta.fits", ["EXP-1", "EXP-2", "EXP-3"])
    with fits.open(p) as out:
        h = out[0].header
        assert h["INSTRUME"] == "CAM-A"
        assert h["FILTER"] == "g"
        assert h["EXPID01"] == "EXP-1"
        assert h["EXPID02"] == "EXP-2"
        assert h["EXPID03"] == "EXP-3"
        assert "EXPID04" not in h


def test_mask_bytes_roundtrip_through_buffer(tmp_path, product_factory):
    products_dir = tmp_path / "products"
    # 使用真实长度（37 字符）的服务成品文件名，覆盖头卡长度边界。
    long_names = [
        "master_flat_20260101T000000_ab12cd.fits",
        "master_flat_20260101T001000_ef34ab.fits",
        "master_flat_20260101T002000_56cd78.fits",
    ]
    paths = []
    for i, long_name in enumerate(long_names):
        p = product_factory(
            f"short_{i}.fits",
            [f"E{i * 3 + 1}", f"E{i * 3 + 2}", f"E{i * 3 + 3}"],
            dead_rows=i,
            products_dir=products_dir,
        )
        renamed = p.parent / long_name
        p.rename(renamed)
        paths.append(renamed)
    # 长文件名写入 FILE01..03 卡：序列化头部不得触发卡截断警告。
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = _run(tmp_path, products_dir, [p.name for p in paths])
    raw = result.path.read_bytes()
    with fits.open(io.BytesIO(raw)) as out:
        mask = np.asarray(out[0].data)
        assert int(mask.sum()) == result.common_valid_pixels
        assert out[0].header["NVALID"] == result.common_valid_pixels
        assert out[0].header["FILE01"] == long_names[0]
