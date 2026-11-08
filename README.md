# PV–BESS–EV Execution Correction & Conditional Terminal Recovery

**Battery execution correction for PV–BESS–EV charging stations: Inventory costs and conditional terminal recovery** (`v1.0.0`)

**English** | **简体中文**（Parallel Bilingual Documentation / 中英双语说明）

---

## 1. Overview & Architecture | 研究概览与控制架构

**EN:** This repository provides the core control algorithms, simulation suites, model inputs, and numerical result tables for feeder-limited photovoltaic–battery–electric-vehicle (PV–BESS–EV) fast-charging stations. It studies fast feeder compliance as an intertemporal energy service: an ordered **reserve-bounded execution filter (REF)** mitigates real-time feeder import exceedances by curtailing scheduled charging before drawing reserve-bounded discharge, paired-trajectory accounting quantifies the resulting terminal storage inventory deficit and critical valuation threshold, and conditional terminal recovery (via the Midpoint Bridge Theorem and event-driven reference governing) restores designated target inventory subject to physical ramp, lag, and zero-export demand absorption limits.

**中文：** 本仓库提供受馈线容量限制的光储充（PV–BESS–EV）快充场站核心控制算法、仿真实验程序、模型输入与数值结果表。本研究将快速馈线合规建模为跨期能量服务：执行层采用有序**备用有界执行滤波器（REF）**，优先撤回计划充电功率，再在保护可交付备用余量的前提下补充放电；配对轨迹核算定量评估由此产生的终端储能库存差与临界估值阈值；在终端恢复阶段，基于中点桥接定理（Midpoint Bridge Theorem）与事件驱动参考调节器（Reference Governor），在变流器爬坡速率、一阶滞后及防逆流本地消纳约束下条件恢复目标储能电量。

```text
Upstream Public Datasets (EV charging sessions, PVOD, ACN-Data, OPSD, SOFIE, StoreNet, ...)
  │  code/fetch_inputs.py · code/prepare_station_inputs.py · experiments/station/inputs.py
  ▼
Causal History / Calibration / Test Windows   (source_time < decision_time, UTC -> UTC+8)
  │  24-h persistence forecast + split / adaptive conformal intervals [lower, point, upper]
  ▼
Supervisory Rolling-Horizon MILP Scheduler    (3 scenarios, non-anticipative t=0 action, HiGHS)
  │  c0_kw >= 0, d0_kw >= 0, pre-step deliverable reserve floor R_req (kWh)
  ▼
PV-Aware Zero-Export Physical Realization     (discharge <= max(EV_actual - PV_actual + charge, 0))
  │  metered (ev_actual_kw, pv_actual_kw) observed at execution time
  ▼
Ordered Reserve-Bounded Execution Filter (REF) (1) curtail charge delta_c -> (2) reserve-bounded discharge delta_d
  │
  ├──► Paired Inventory Valuation             delta_E_term = E_T(arm) - E_T(GS_F0 / F0)
  └──► Conditional Terminal Recovery          two-hold ramp-before-lag Midpoint Bridge / Reference Governor
```

### Repository Structure | 目录结构

| Path / 路径 | EN Description | 中文说明 |
| :--- | :--- | :--- |
| `code/` | Core Python package (`pv_bess_ev`: standalone REF operator, analytical Midpoint Bridge solver, paired inventory valuation, event-driven reference governor, and ONAN transformer thermal model) plus upstream data preparation scripts (`fetch_inputs.py`, `prepare_station_inputs.py`). | 核心算法包（`pv_bess_ev`：独立 REF 算子、解析中点桥接求解器、配对库存估值、事件驱动参考调节器及油浸式配变热模型）与上游公开数据获取及预处理脚本。 |
| `experiments/` | Self-contained runnable experiment suites (`station`, `closed_loop`, `recovery`, `recovery_full`, `acn`, `seven_station`, `dynamic_network`, `thermal`, `fpga`, `benchmarks`). | 10 组完整可复现实验套件，涵盖单站多季节与压力网格、双进程闭环、终端恢复、ACN 三站点、IEEE 33 节点配电网、配变电热耦合、Artix-7 FPGA 定点实现及外部控制基准算例。 |
| `inputs/` | Upstream dataset catalog (`sources.json`) and derived model input series (`inputs/model_inputs/`, `inputs/series/`). | 上游公开数据集目录（`sources.json`）及可直接用于实验复现的派生模型输入序列。 |
| `results/` | Topic-organized numerical result tables and trajectory series (`station/`, `closed_loop/`, `recovery/`, `acn/`, `network/`, `thermal/`, `fpga/`, `external_sites/`, `sensitivity/`, `benchmarks/`, `series/`). | 按研究主题分类组织的全部论文与补充材料数值结果表及典型运行时间序列。 |

---

## 2. Physical Conventions & Methodological Summary | 物理符号约定与核心机制

### 2.1 Power Sign Conventions & Units | 功率符号与单位约定

* **Station scheduling & REF (`code/pv_bess_ev/ref.py`, `experiments/station/`):**
  * **EN:** Charging (`charge_kw`), discharging (`discharge_kw`), EV load (`ev_actual_kw`), PV output (`pv_actual_kw`), contractual feeder limit (`feeder_safe_kw`), and solar curtailment (`pv_curtail_kw`) are **non-negative scalars in `kW`**. Energies and reserves are in **`kWh`**, and `SOC` lies in `[0.10, 0.90]` (base station: `600 kWh`, `250 kW / 250 kW`, $\eta_c = \eta_d = 0.95$). Converter exclusivity forbids simultaneous charging and discharging. Net grid import before filtering is $g = \text{EV}_{\text{actual}} - \text{PV}_{\text{actual}} + p_c - p_d + p_{\text{curt}} \ge 0$.
  * **中文：** 场站调度与 REF 模块中，充电（`charge_kw`）、放电（`discharge_kw`）、EV 负荷（`ev_actual_kw`）、光伏出力（`pv_actual_kw`）、馈线限额（`feeder_safe_kw`）及弃光功率均为**非负千瓦量（`kW`）**；电量与备用单位为 **`kWh`**，`SOC` 范围为 `[0.10, 0.90]`（基准场站：`600 kWh`、`250 kW`、$\eta_c = \eta_d = 0.95$）。变流器物理互斥禁止同时充放电。
* **Actuator dynamics & terminal recovery (`code/pv_bess_ev/midpoint_bridge.py`, `experiments/recovery/`, `experiments/recovery_full/`):**
  * **EN:** Sub-minute actuator trajectories use **signed battery power $p = p_{\text{dis}} - p_{\text{ch}}$ (`kW`)**, with **$p > 0$ for discharging** and **$p < 0$ for charging**. At 1-second resolution ($\Delta t = 1/3600\text{ h}$), ramp limiting precedes first-order lag: $p_k = p_{k-1} + a \cdot \text{clip}(u - p_{k-1}, -r, r)$ where $a = 1 - \exp(-1/\tau_{\text{lag}})$. Stored energy evolves piecewise-affinely across $p = 0$: $E_k = E_{k-1} - \Delta t \, p_k / \eta_d$ for $p_k \ge 0$ and $E_k = E_{k-1} - \Delta t \, \eta_c p_k$ for $p_k < 0$.
  * **中文：** 秒级执行器动态与终端恢复模块采用**有符号电池功率 $p = p_{\text{dis}} - p_{\text{ch}}$（`kW`）**，即 **$p > 0$ 为放电、$p < 0$ 为充电**。离散状态按“先斜率限制、后一阶滞后”推进：$p_k = p_{k-1} + a \cdot \text{clip}(u - p_{k-1}, -r, r)$，$a = 1 - \exp(-1/\tau_{\text{lag}})$；储能电量在 $p=0$ 处按充放电效率分段仿射积分。

### 2.2 Causal Observation Timing | 因果观测时序

* **EN:** At decision step $t$, persistence forecasts and conformal calibration bands access only observations strictly prior to $t$ (`source_time < decision_time`). The supervisory 3-scenario MILP enforces first-stage non-anticipativity across favorable, nominal, and adverse scenarios before realized step disturbances arrive. Realized `(ev_actual_kw, pv_actual_kw)` enter only during physical PV-aware realization and execution-layer REF evaluation, and join historical calibration pools only after step completion. In `experiments/closed_loop/`, controller and plant run as separate processes communicating over asynchronous pipes.
* **中文：** 在决策步 $t$，持续性预测与保形区间校准仅读取严格早于 $t$ 的历史记录（`source_time < decision_time`）。上层三场景 MILP 在未知当步真实扰动前施加首步非预期性耦合约束；真实负荷与光伏仅在步内光伏感知防逆流实现与执行层 REF 计算阶段可见，并在步末更新至历史序列。`experiments/closed_loop/` 将控制器与受控站分离为两个独立进程，通过管道按固定控制时隙异步交互。

### 2.3 Ordered Reserve-Bounded Execution Filter (REF) | 有序备用有界执行滤波器（REF）

* **EN:** After applying the common PV-aware zero-export mapping $p_d = \min(d_0, \max(\text{EV} - \text{PV} + p_c, 0))$, any feeder excursion $\xi_0 = \max(g - P_{\text{cap}}, 0)$ is mitigated in strict priority order:
  1. **Curtail scheduled charging:** $\delta_c = \min(p_c, \xi_0)$, relieving feeder import 1-for-1 without depleting stored energy.
  2. **Supplement reserve-bounded discharge:** $\delta_d = \min(\bar{\delta}_d, \xi_0 - \delta_c)$ only after charging reaches zero, where $\bar{\delta}_d = \min\bigl(P_{d,\max} - p_d,\, \max(0, E_{\text{deliv}} - R_{\text{req}} - p_d \Delta t)/\Delta t\bigr)$ and $E_{\text{deliv}} = \eta_d E_{\text{cap}}(\text{SOC} - \text{SOC}_{\min})$.
* **中文：** 在执行统一的光伏感知防逆流映射后，针对馈线越限量 $\xi_0 = \max(g - P_{\text{cap}}, 0)$，REF 按严格优先级依次执行：
  1. **优先撤回计划充电：** $\delta_c = \min(p_c, \xi_0)$，1:1 压降馈线受电且不消耗储能库存；
  2. **按备用余量补充放电：** 仅在充电撤回至零后启用额外放电 $\delta_d = \min(\bar{\delta}_d, \xi_0 - \delta_c)$，放电上限同时受变流器功率余量与扣除保护备用 $R_{\text{req}}$ 后的可交付能量约束。

### 2.4 Paired Terminal Inventory Valuation | 配对终端库存估值

* **EN:** Paired evaluations compare arms initialized from identical states ($E_0$) under identical exogenous streams against the uncorrected baseline (`GS_F0` / `F0`):
  $$\Delta E_{\text{term}} = E_T(\text{arm}) - E_T(\text{baseline}), \qquad \Delta C_{\text{adj}}(v) = \Delta C_{\text{op}} - v \, \Delta E_{\text{term}}$$
  Whenever corrective support reduces operating expenditure ($\Delta C_{\text{op}} < 0$) while incurring a terminal deficit ($\Delta E_{\text{term}} < 0$), the critical valuation $v^* = \Delta C_{\text{op}} / \Delta E_{\text{term}}$ marks the exact threshold above which storage depletion outweighs immediate electricity-cost savings.
* **中文：** 配对评估在同一初始状态（$E_0$）与同一外部扰动序列下，以无执行修正的基准臂（`GS_F0` / `F0`）为对照计算终端库存差 $\Delta E_{\text{term}}$ 与计及库存估值的成本差 $\Delta C_{\text{adj}}(v) = \Delta C_{\text{op}} - v \, \Delta E_{\text{term}}$；当执行修正降低直接运行成本（$\Delta C_{\text{op}} < 0$）但造成终端库存赤字（$\Delta E_{\text{term}} < 0$）时，$v^* = \Delta C_{\text{op}} / \Delta E_{\text{term}}$ 即为成本排序反转的临界储能单价。

---

## 3. Data & Results Organization | 数据输入与结果表说明

| Category / 类别 | Path / 路径 | Description / 说明 |
| :--- | :--- | :--- |
| **Derived Model Inputs**<br>**派生模型输入** | `inputs/model_inputs/`<br>`inputs/series/`<br>`experiments/*/inputs/` | Redistributable model input packets and operating series included directly in this repository. Suites `closed_loop`, `recovery`, `recovery_full`, `seven_station`, `dynamic_network`, `thermal`, `fpga`, and `benchmarks` run directly without external downloads. / 直接包含在仓库中的派生模型输入，上述 8 组实验无需外部下载即可直接运行。 |
| **Upstream Public Datasets**<br>**上游公开原始数据集** | `inputs/sources.json`<br>`code/fetch_inputs.py`<br>`code/prepare_station_inputs.py` | Public raw datasets obtained from their original publishers (Chinese urban EV charging transactions on Figshare, ScienceDB `PVOD v1.0`, Caltech `ACN-Data`, OPSD, SOFIE, Dingle, StoreNet, RAPT, Estonia, and municipal EV records). URLs and SHA-256 checksums are listed in `inputs/sources.json`. / 遵循原发布方条款从公开渠道获取的原始数据集，下载链接与 SHA-256 校验值见 `inputs/sources.json`。 |
| **Numerical Result Tables**<br>**数值结果汇总与序列** | `results/` | Complete CSV summary tables and compressed `.npz` / `.csv.gz` operating trajectories organized into 11 topic folders (`station`, `closed_loop`, `recovery`, `acn`, `network`, `thermal`, `fpga`, `external_sites`, `sensitivity`, `benchmarks`, `series`). / 按 11 个主题目录完整收录的论文正文与补充材料数值结果表及运行序列。 |

```bash
# List registered upstream datasets in inputs/sources.json | 查看上游公开数据集清单
python -B code/fetch_inputs.py list
```

---

## 4. Quickstart: Single-Station Benchmark | 快速开始：单场站闭环实验示例

Run all commands from the **repository root** (`pv-bess-ev-execution-recovery/`):  
以下命令均在**仓库根目录**下执行：

```bash
# 1. Install dependencies | 安装依赖
python -m pip install -r experiments/station/requirements.txt

# 2. Prepare station inputs from PVOD.zip and Figshare Dataset.zip | 由上游数据包生成场站输入
python -B experiments/station/inputs.py \
  --output input \
  --pvod-archive /path/to/PVOD.zip \
  --china-archive /path/to/Dataset.zip

# 3. Run paired GS_F0 (uncorrected) vs GS_F1 (REF) on SPRING_30D stress case | 运行配对闭环仿真
python -B experiments/station/run.py \
  --data input \
  --output outputs/stress_spring \
  --split SPRING_30D \
  --condition PRESS_E1P6_PV0P6 \
  --arm paired
```

* Use `--condition BASE` with `--split WINTER_30D`, `SPRING_30D`, `SUMMER_CLEAN_30D`, or `LONG_194D` (`4,656` hourly steps) for nominal seasonal or 194-day evaluations, or pass `--steps 48` for a short test run.
* 若评估常规倍率多季节或 194 天连续序列，可将 `--condition` 设为 `BASE` 并选择 `WINTER_30D`、`SPRING_30D`、`SUMMER_CLEAN_30D` 或 `LONG_194D`；附加 `--steps 48` 可快速测试小步数运行。

---

## 5. Experiment Suites & Result Mapping | 实验套件与结果目录对照表

All entrypoints run from the repository root with `python -B` (`seven_station`, `dynamic_network`, and `thermal` additionally use `pandapower==3.4.0`):  
所有实验均在仓库根目录下通过 `python -B` 启动（配电网与电热实验需安装 `pandapower==3.4.0`）：

| # | Suite / 实验套件 | Entrypoint & Example Command / 运行命令示例 | Corresponding Results in `results/` / 对应结果目录 |
| :- | :--- | :--- | :--- |
| **1** | **Single-Station Scheduling & REF**<br>单场站滚动调度与 REF | `python -B experiments/station/run.py --data input --output outputs/station --split SPRING_30D --condition BASE --arm paired` | `results/station/pv_aware_seasonal_comparison.csv`<br>`results/station/pv_aware_194d_longstream_comparison.csv`<br>`results/station/stress_grid_6x6_108_arms.csv`<br>`results/station/information_sets_15min_stress.csv` |
| **2** | **High-Frequency Software Closed Loop**<br>双进程高频异步闭环（OPSD / SOFIE） | `python -B experiments/closed_loop/run.py --case PVAWARE_OPSD24H --arm both --output outputs/closed_loop_opsd24h` | `results/closed_loop/sofie_dec4_60s_paired_arms.csv`<br>`results/closed_loop/opsd_67h_terminal_inventory_values.csv`<br>`results/closed_loop/closed_loop_121_paired_results.csv` |
| **3** | **Equal-Hold Midpoint Bridge Recovery**<br>等保持中点桥接终端恢复 | `python -B experiments/recovery/run.py --branch analytic --output outputs/recovery_analytic` | `results/recovery/equal_hold_bridge_arm_results.csv`<br>`results/recovery/midpoint_bridge_catalog_results.csv`<br>`results/series/recovery/equal_hold_bridge_steps.csv` |
| **4** | **Rolling Preparation & Terminal Recovery**<br>提前备储与滚动终端恢复 | `python -B experiments/recovery_full/run.py --case MAY18 --arm SIMPLE_PREP_CAUSAL_BRAKE --commands plan --output outputs/rec_may18.json` | `results/recovery/full_horizon_10_arm_comparison.csv`<br>`results/recovery/rolling_preparation_6_arm_comparison.csv`<br>`results/series/recovery/full_recovery_curve_data.csv` |
| **5** | **ACN Three-Site Reference Governor**<br>ACN 三站点事件驱动参考调节器 | `python -B experiments/acn/run.py --case JPL_EVENT_REFERENCE_GOVERNOR --raw-pages raw/jpl --report outputs/acn_jpl.json` | `results/acn/reference_governor_3_sites.csv`<br>`results/acn/three_site_paired_contrasts.csv`<br>`results/series/acn/acn_daily_summary.csv` |
| **6** | **IEEE 33-Bus Seven-Station Placement**<br>IEEE 33 节点七场站接入位置对比 | `python -B experiments/seven_station/run.py --case FAR --output outputs/seven_far` | `results/network/ieee33_seven_station_arm_endpoints.csv`<br>`results/network/ieee33_seven_station_paired_differences.csv` |
| **7** | **IEEE 33-Bus 1-s Dynamic Network**<br>IEEE 33 节点逐秒动态协调对比 | `python -B experiments/dynamic_network/run.py --arm all --output outputs/dynamic_network` | `results/network/ieee33_3h_paired_summary.csv`<br>`results/series/network/ieee33_dynamic_3h_seconds.csv` |
| **8** | **Transformer Electro-Thermal Control**<br>250 kVA 配电变压器电热耦合控制 | `python -B experiments/thermal/run.py --case all --arm all --report outputs/thermal_summary.json` | `results/thermal/electrothermal_coupling_16_arms.csv`<br>`results/thermal/electrothermal_coupling_8_pairs.csv` |
| **9** | **Artix-7 FPGA Q16.16 Fixed-Point & HIL**<br>Artix-7 FPGA 定点硬件核与串口 HIL | `python -B experiments/fpga/run.py --count 5000 --seed 20260706` (and `experiments/fpga/vivado/build.tcl` for Vivado synthesis) | `results/fpga/fpga_hil_479305_vectors_summary.csv` |
| **10** | **Published Control Benchmarks**<br>外部公开控制基准（BOPTEST & IEEE-14） | `python -B experiments/benchmarks/run.py --output-dir outputs/benchmarks` | `results/benchmarks/boptest_and_ieee14_summary.csv` |

Additional cross-city, household, and parameter sensitivity tables referenced in the manuscript and Supplementary Material are located in `results/external_sites/` and `results/sensitivity/`.

---

## 6. Reproducibility, Licensing & Citation | 确定性复现、开源许可与引用说明

* **Numerical reproducibility (`数值确定性配置`):** Experiment runners fix single-thread BLAS/OpenMP (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`), `PYTHONHASHSEED=20260706`, and single-thread HiGHS (`threads=1`, `parallel="off"`, `random_seed=20260706`, `mip_rel_gap=1e-4`, feasibility tolerance `1e-7`).
* **Licenses & Attribution (`开源许可与商用/使用署名要求`):**
  * **EN:** Software (`code/`, `experiments/`) is licensed under **Apache License 2.0 (`Apache-2.0`)**, and derived model inputs, numerical result tables, and documentation (`inputs/`, `results/`, `README.md`) are licensed under **Creative Commons Attribution 4.0 International (`CC-BY-4.0`)** (see `LICENSE.txt` and `LICENSES/`). Commercial and non-commercial use, adaptation, and redistribution are permitted; **in any commercial product/service, software or data redistribution, derivative work, technical report, or academic publication that uses material from this repository, please credit the authors (`Bailong Ma, Lingzhao Meng, Yunjie Peng, Xinhua Wang, Jinyuan Zhou, Renqian Tao`) and include the repository link (`https://github.com/dongfangluo31/pv-bess-ev-execution-recovery`)**, as detailed in `NOTICE.txt` and `CITATION.cff`.
  * **中文：** 本仓库软件代码（`code/`、`experiments/`）采用 **`Apache-2.0`** 开源协议，派生模型输入、数值结果表与技术文档（`inputs/`、`results/`、`README.md`）采用 **`CC-BY-4.0`** 协议（详见 `LICENSE.txt` 与 `LICENSES/`）。允许商业及非商业使用、修改与再分发；**在商业应用、产品集成、二次分发、技术报告或学术发表等任何使用过程中，凡使用了本仓库中的代码、模型输入或数值结果，均须按 `NOTICE.txt` 与 `CITATION.cff` 提及全体作者（`Bailong Ma, Lingzhao Meng, Yunjie Peng, Xinhua Wang, Jinyuan Zhou, Renqian Tao`）并附上本仓库链接（`https://github.com/dongfangluo31/pv-bess-ev-execution-recovery`）**。

---

## 7. Declaration of Generative AI Use | 生成式人工智能使用声明

* **EN:** During the preparation of this open-source release, the authors used generative artificial intelligence (Generative AI) tools to assist in organizing the repository structure, standardizing code comments, and drafting bilingual documentation.
* **中文：** 在本开源仓库的整理与发布过程中，作者使用了生成式人工智能（Generative AI）工具辅助梳理仓库目录结构、规范代码注释以及编写中英双语说明文档。
