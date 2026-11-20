# 天文测光主平场制备与重复性检验服务

基于 **FastAPI + Astropy** 的主平场（master flat）制备接口。上传 3～12 张已扣偏置的
平场 FITS，服务按统一规则聚合，输出可直接用于平场校正的成品 FITS；并可对多份成品做
平场重复性检验，输出共同有效像元掩膜。

## 处理规则

### `/prepare` 主平场制备

1. **输入约束**
   - 帧数 3～12 张；每张恰好一幅**无压缩、二维、16 位整数**主图（`BITPIX=16`）。
   - 宽高均不超过 4096；同批帧尺寸必须一致。
   - 同批相机（`INSTRUME`）与滤镜（`FILTER`）标识必须一致（关键字缺失则不参与比对）。
   - 禁止压缩 HDU（`CompImageHDU`）与表 HDU。
2. **物理量换算**：保留原始整数读入，自行按 `BSCALE` / `BZERO` 换算
   （`value = raw * BSCALE + BZERO`），不依赖 Astropy 的自动缩放。
3. **有效像元**
   - 按 `BLANK` 关键字排除缺测像元；
   - 仅保留换算后 `0 < value < saturation`（严格大于零、严格低于饱和阈值）的像元；
   - 饱和阈值必须为正且有限。
4. **两阶段中位数聚合**
   - 每帧的有效像元除以**该帧有效像元全图中位数**做归一；
   - 对同一位置跨帧的有效归一值取**中位数**；
   - 某位置参与帧数少于 3 时输出 `NaN`。
5. **输出 FITS（双 HDU）**
   - 主图（PRIMARY，`float32`）：校正系数；
   - 扩展图（`COUNTS`，`uint8`）：每个像元实际参与的帧数；
   - 头部另存原帧提供的 `INSTRUME`、`FILTER`，以及各帧曝光号
     `EXPID01`…`EXPIDnn`（某帧缺 `EXPID` 时跳过该卡，**仍可正常制备**）；
   - 头部同时记录帧数、饱和阈值、各帧中位数、输入清单与处理历史。
6. **整批失败语义**：文件截断、格式不符、任一帧无有效像元、写出失败均整批报错，
   成品先写临时文件、回读校验通过后才原子改名，**只提供完整成品，不留半成品**。

### `/repeatability` 平场重复性检验

在 `/docs` 以 `POST /repeatability` 提交**同一读出模式**下 3～12 份本服务生成的
全幅成品文件名（仅文件名，JSON body `{"filenames": [...]}`）。规则：

1. **候选一致性**：所有候选成品的相机（`INSTRUME`）、滤镜（`FILTER`）与全幅尺寸
   必须一致；任一成品缺相机、滤镜或完整曝光号（`EXPID01`…`EXPIDnn` 缺一不可）、
   文件截断或不是本服务成品，整批返回错误。
2. **独立组合**：从候选中选 3 份，三份成品的源 `EXPID` 集合须**两两无交集**
   （没有共用原曝光）。
3. **共同有效像元最大化**：共同有效像元指同一位置在三份成品的 `COUNTS` 图中
   **均不少于 3**；取共同有效像元数最多的一组。并列时把各组内文件名排序，
   取**字典序最小**的一组。
4. **输出掩膜 FITS（单 HDU，`uint8`）**：共同有效像元填 `1`，其余填 `0`；
   头部 `FILE01`/`FILE02`/`FILE03` 列出入选文件名，`NVALID` 给出共同有效像元数，
   并记录相机、滤镜与处理历史；响应头另带 `X-Common-Valid-Pixels` 与
   `X-Selected-Files`。
5. **失败语义**：找不到任何有共同有效像元（≥1）的独立组合、文件名非法/越界、
   成品不存在等，均只返回 `422` 错误（`{"detail": "..."}`），不产生半成品。

## 内存设计（256 MiB 容器预算）

`/prepare` 最坏情形（12 帧 × 4096×4096）实测峰值约 **238 MiB**，余量约 18 MiB：

- 第一趟用 65536 桶的原始值**直方图**精确求每帧中位数，不缓存整帧有效值；
- 第二趟按**自适应列条带**聚合，条带宽度由 `STRIP_BUDGET_BYTES`（默认 8 MiB）反推；
- 帧轴中位数用**原地排序 + 索引选择**，避免 `np.nanmedian` 的整份条带复制与类型提升；
- 输出用 `ndarray.tofile` **流式直写** FITS 数据块，规避 `HDUList.writeto` 的大块内部缓冲。

`/repeatability` 最坏情形（12 份 × 4096×4096）独立进程实测峰值约 **94 MiB**：

- 每份成品只打开一次做头部校验；`COUNTS` 按数据块字节偏移（`datLoc`）
  **逐行块直读原始字节**（不使用 numpy memmap，避免热页缓存下被映射页计入
  RSS 的不确定性），把 `COUNTS >= 3` 折叠成 1 bit 累加进 `uint16` 位码图
  （4096² 仅 32 MiB），每份成品的 `COUNTS` 只顺序扫描一遍，之后不再重读；
- 位码直方图按行块累加，规避 `np.bincount` 把 `uint16` 整图提升为 `int64`
  的瞬时大分配；三元组（最多 220 组）只在直方图上查表计数，不做逐像元组合遍历；
- 掩膜按行块由位码图生成，`uint8` 输出 16 MiB；回读校验同样走 memmap 分块，
  不再实体化第二份整图。

服务全程串行处理（全局锁），避免两个大请求叠加内存。

## 启动方式

### Docker Compose（推荐）

```bash
docker compose up --build
```

容器内存硬限制为 256 MiB（`mem_limit` / `memswap_limit`）；成品、重复性掩膜与
工作目录分别使用命名卷（`flat_products` / `flat_masks` / `flat_work`）。
启动后访问：

- 交互文档（可直接上传/提交）：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/health>

### 本地直接运行

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 接口用法

### `POST /prepare`（`multipart/form-data`）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `files` | File[] | 3～12 张平场 FITS |
| `saturation` | number | 正且有限的饱和阈值（与换算后数据同单位） |

成功返回 `application/fits` 成品附件；失败返回 `422`（输入/数据问题）或
`500`（写出问题），响应体 `{"detail": "..."}`。

```bash
curl -X POST http://localhost:8000/prepare \
  -F "saturation=4000" \
  -F "files=@flat_0.fits" \
  -F "files=@flat_1.fits" \
  -F "files=@flat_2.fits" \
  -o master_flat.fits
```

### `POST /repeatability`（`application/json`）

请求体为同读出模式下 3～12 份**本服务成品文件名**（不含目录）：

```bash
curl -X POST http://localhost:8000/repeatability \
  -H "Content-Type: application/json" \
  -d '{"filenames": ["master_flat_20260101T000000_ab12cd.fits",
                     "master_flat_20260101T001000_ef34ab.fits",
                     "master_flat_20260101T002000_56cd78.fits"]}' \
  -o repeatability_mask.fits
```

成功返回 `application/fits` 掩膜附件，响应头含 `X-Common-Valid-Pixels` 与
`X-Selected-Files`；任何候选不合格或无可用独立组合时返回 `422`。

## 测试

```bash
pip install pytest httpx
PYTHONPATH=. python -m pytest tests/ -q
```

- `tests/test_pipeline.py`、`tests/test_api.py`：`/prepare` 单元/接口测试，
  含与 numpy 全量对照实现的数值比对、BLANK/BSCALE/BZERO、<3 帧 NaN、
  饱和/非正过滤、相机/滤镜与尺寸一致性、截断/格式/帧数/阈值等失败路径。
- `tests/test_repeatability.py`、`tests/test_repeatability_api.py`：
  `/repeatability` 选组（最大化、并列字典序）、EXPID 无交集、掩膜数值与
  暴力枚举对照、缺相机/滤镜/曝光号、截断、尺寸/相机/滤镜不一致、路径穿越、
  无共同有效像元等失败路径，以及 `/prepare` 头部透传与缺 EXPID 仍可制备。
- `tests/scripts/stress_memory.py`：12 帧 4096² 端到端数值压测。
- `tests/scripts/measure_pipeline_only.py`：`/prepare` 纯净子进程峰值内存测量。
- `tests/scripts/stress_repeatability.py`：12 份 4096² 成品的重复性压测，
  调用 `measure_repeatability_only.py` 在纯净子进程测峰值并做正确性抽查。
- `tests/scripts/matrix_sizes.py`：多尺寸/帧数矩阵（独立进程测峰值）。

## 目录结构

```
app/
  main.py             # FastAPI 路由、上传隔离、串行锁
  pipeline.py         # /prepare：校验、直方图中位数、条带聚合、低内存 FITS 写出
  repeatability.py    # /repeatability：memmap 位码折叠、独立三元组选优、掩膜写出
tests/                # 单元/接口测试与压测脚本
Dockerfile
docker-compose.yml
requirements.txt
```
