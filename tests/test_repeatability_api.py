"""/repeatability 接口测试。"""

from __future__ import annotations

import io

import numpy as np
from astropy.io import fits
from fastapi.testclient import TestClient

from app import main


def _client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path / "products")
    monkeypatch.setattr(main, "MASK_DIR", tmp_path / "masks")
    for sub in ("work", "products", "masks"):
        (tmp_path / sub).mkdir()
    return TestClient(main.app)


def _make_products(tmp_path, product_factory, n=4):
    products_dir = tmp_path / "products"
    names = []
    for i in range(n):
        p = product_factory(
            f"prod_{i:02d}.fits",
            [f"E{i * 3 + 1}", f"E{i * 3 + 2}", f"E{i * 3 + 3}"],
            dead_rows=i,
            products_dir=products_dir,
        )
        names.append(p.name)
    return products_dir, names


def test_repeatability_endpoint_returns_mask(tmp_path, monkeypatch, product_factory):
    client = _client(tmp_path, monkeypatch)
    _, names = _make_products(tmp_path, product_factory)
    resp = client.post("/repeatability", json={"filenames": names})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/fits"
    nvalid = int(resp.headers["x-common-valid-pixels"])
    assert nvalid > 0
    with fits.open(io.BytesIO(resp.content)) as out:
        mask = np.asarray(out[0].data)
        assert mask.dtype == np.uint8
        assert int(mask.sum()) == nvalid
        assert out[0].header["NVALID"] == nvalid
        assert out[0].header["FILE01"] in names
    # 掩膜已落盘到 masks 目录。
    assert list((tmp_path / "masks").glob("repeatability_mask_*.fits"))


def test_repeatability_docs_and_openapi_include_endpoint(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    openapi = client.get("/openapi.json").json()
    assert "/repeatability" in openapi["paths"]
    assert client.get("/docs").status_code == 200


def test_repeatability_too_few_filenames_422(tmp_path, monkeypatch, product_factory):
    client = _client(tmp_path, monkeypatch)
    _, names = _make_products(tmp_path, product_factory, n=3)
    resp = client.post("/repeatability", json={"filenames": names[:2]})
    assert resp.status_code == 422


def test_repeatability_too_many_filenames_422(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    resp = client.post(
        "/repeatability", json={"filenames": [f"p{i}.fits" for i in range(13)]}
    )
    assert resp.status_code == 422


def test_repeatability_missing_field_422(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    resp = client.post("/repeatability", json={})
    assert resp.status_code == 422


def test_repeatability_bad_product_returns_422(tmp_path, monkeypatch, product_factory):
    client = _client(tmp_path, monkeypatch)
    _, names = _make_products(tmp_path, product_factory)
    names.append("does_not_exist.fits")
    resp = client.post("/repeatability", json={"filenames": names})
    assert resp.status_code == 422
    assert "不存在" in resp.json()["detail"]


def test_repeatability_shared_expid_returns_422(tmp_path, monkeypatch, product_factory):
    client = _client(tmp_path, monkeypatch)
    products_dir = tmp_path / "products"
    p1 = product_factory(
        "a.fits", ["X1", "X2", "X3"], products_dir=products_dir
    )
    p2 = product_factory(
        "b.fits", ["X3", "X4", "X5"], products_dir=products_dir
    )
    p3 = product_factory(
        "c.fits", ["X6", "X7", "X8"], products_dir=products_dir
    )
    resp = client.post(
        "/repeatability", json={"filenames": [p1.name, p2.name, p3.name]}
    )
    assert resp.status_code == 422
    assert "无交集" in resp.json()["detail"]


def test_repeatability_path_traversal_returns_422(tmp_path, monkeypatch, product_factory):
    client = _client(tmp_path, monkeypatch)
    _, names = _make_products(tmp_path, product_factory, n=3)
    resp = client.post(
        "/repeatability", json={"filenames": names[:2] + ["../secret.fits"]}
    )
    assert resp.status_code == 422
    assert "文件名非法" in resp.json()["detail"]


def test_repeatability_selected_files_header(tmp_path, monkeypatch, product_factory):
    client = _client(tmp_path, monkeypatch)
    _, names = _make_products(tmp_path, product_factory, n=3)
    resp = client.post("/repeatability", json={"filenames": names})
    assert resp.status_code == 200, resp.text
    listed = resp.headers["x-selected-files"].split("; ")
    assert sorted(listed) == sorted(names)
