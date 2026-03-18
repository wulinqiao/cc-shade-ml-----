"""
run_ccshademl.py — CC-SHADE-ML 算法主运行模块 / CC-SHADE-ML Algorithm Main Runner.

此模块实现 CC-SHADE-ML 算法与 demo 基准测试框架的集成，
提供单次试验运行和多次独立重复实验的完整流程。
This module integrates the CC-SHADE-ML algorithm with the demo benchmark
framework, providing single-trial and multi-run experiment workflows.

主要组件 / Main Components:
    - call_fun():      目标函数调用适配器 / Objective function call adapter
    - run_one_trial(): 单次独立实验 / Single independent trial
    - main():          命令行入口，支持多函数多次运行 / CLI entry point

使用示例 / Usage Example:
    # 快速测试 / Quick test
    python run_ccshademl.py --id 1 --runs 1 --fev 50000

    # 完整实验 / Full experiment
    python run_ccshademl.py --id 1 --runs 25 --fev 3000000

    # 全部函数 / All functions
    python run_ccshademl.py --runs 25 --fev 3000000

参考 / References:
    - Vakhnin & Sopov, CC-SHADE-ML, Algorithms 2022, 15, 451
    - IEEE LSGO CEC'2013 benchmark suite
"""

import os
import sys
import time
import math
import logging
import argparse
import numpy as np
import numpy.typing as npt
from typing import Callable, Dict, List, Optional, Tuple

# ── 路径修复 / Path Fix ───────────────────────────────────────────────────────
# 保证无论从哪个目录运行，都能找到 benchmark/ 数据文件
# Ensures benchmark/ data files are found regardless of working directory
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

from benchmark.cec2013lsgo.cec2013 import Benchmark
import constants as C
from header import (
    initializePopulation, initializeHistory,
    findBestIndex, generation_CR, generation_F,
    chooseCrossoverIndecies, Algorithm_1,
    check_out_borders, updateArchive,
    find_best_part_index, find_best_fitness_value,
    rnd_indecies, indecesSuccession, randperm,
    random_performance, max_number,
    mean_stat, min_stat, max_stat, median_stat, stddev_stat,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 算法参数常量 / Algorithm Parameter Constants
# ══════════════════════════════════════════════════════════════════════════════

# 问题参数 / Problem Parameters
DIMENSION        = 1000             # 问题维度 / Problem dimension

# 算法参数（与原 main.cpp 一致）/ Algorithm Parameters (consistent with main.cpp)
FEV_GLOBAL       = int(3e6)         # 函数评估预算 / Function evaluation budget
R_RUNS           = 25               # 独立运行次数 / Number of independent runs
HISTORY_LENGTH   = 6                # SHADE 历史记忆长度 / SHADE history length
PIECE            = 0.1              # pbest 比例 / Fraction for pbest selection
POWER            = 7.0              # Boltzmann 选择温度（论文推荐值）/ Boltzmann temperature (paper recommended)

# 多层次候选池（tuned 版本，来自论文 Section 3.3）
# Multi-level candidate pools (tuned version from paper Section 3.3)
POP_SIZE_POOL       = [25, 50, 100]     # 种群大小候选 / Population size candidates
SUBCOMPONENTS_POOL  = [5, 10, 20, 50]  # 子分量数候选 / Subcomponent count candidates

# 派生常量 / Derived Constants
POP_SIZE_MAX  = max(POP_SIZE_POOL)      # 最大种群大小 / Maximum population size: 100
ARCHIVE_SIZE  = POP_SIZE_MAX            # 档案大小 / Archive size: 100
M_MAX         = max(SUBCOMPONENTS_POOL) # 最大子分量数 / Maximum subcomponents: 50


def call_fun(
    fun: Callable[[npt.NDArray[np.float64]], float],
    x: npt.NDArray[np.float64]
) -> float:
    """
    目标函数调用适配器 / Objective function call adapter.

    demo 框架的目标函数会将 1D 输入升维为 (1, D) 并返回 shape(1,) 数组，
    本函数将结果统一转为 Python float，并传入副本防止 numba JIT 原地修改。
    The demo framework auto-promotes 1D input to (1, D) and returns shape(1,)
    array; this function converts result to Python float and passes a copy
    to prevent in-place modification by numba JIT.

    Args:
        fun (Callable): demo 框架提供的目标函数 / Objective function from demo framework
        x (ndarray):    待评估的解向量，shape (N,) / Solution vector to evaluate, shape (N,)

    Returns:
        float: 目标函数值 / Objective function value
    """
    result = fun(x.copy())
    if hasattr(result, '__len__'):
        return float(result[0])
    return float(result)


def run_one_trial(
    fun: Callable[[npt.NDArray[np.float64]], float],
    info: Dict,
    FEV_global: int,
    seed: Optional[int] = None
) -> Tuple[List[float], float]:
    """
    运行一次 CC-SHADE-ML 独立实验 / Run one independent CC-SHADE-ML trial.

    实现论文 Algorithm 2 的完整流程：外循环自适应选择 M 和 pop_size，
    内循环用 SHADE 优化各子分量，结束后更新性能分数。
    Implements the full flow of paper Algorithm 2: outer loop adaptively
    selects M and pop_size, inner loop optimizes each subcomponent with
    SHADE, then updates performance scores.

    Args:
        fun (Callable): demo 框架提供的目标函数 / Objective function from demo framework
        info (Dict):    函数信息字典，包含 'lower'、'upper'、'dimension'
                        / Function info dict with 'lower', 'upper', 'dimension'
        FEV_global (int): 函数评估总预算 / Total function evaluation budget
        seed (int, optional): 随机种子，None 表示不设置 / Random seed; None means not set

    Returns:
        Tuple[List[float], float]:
            - fitness_record: 100 个等间隔检查点的 best-so-far 值
                              / 100 evenly-spaced best-so-far values
            - elapsed: 运行时间（秒）/ Wall-clock time in seconds

    Raises:
        KeyError: 当 info 缺少必要字段时 / When info is missing required fields
    """
    for field in ('lower', 'upper', 'dimension'):
        if field not in info:
            raise KeyError(f"info 缺少必要字段 / info missing required field: '{field}'")

    if seed is not None:
        C.set_seed(seed)
        logger.debug(f"随机种子已设置 / Random seed set: {seed}")

    a_bound  = info['lower']
    b_bound  = info['upper']
    krantost = FEV_global // 100

    # ── 分配工作数组 / Allocate working arrays ─────────────────────────────
    population     = np.zeros((POP_SIZE_MAX, DIMENSION))
    population_new = np.zeros((POP_SIZE_MAX, DIMENSION))
    archive        = np.zeros((ARCHIVE_SIZE, DIMENSION))
    solution       = np.zeros(DIMENSION)
    u              = np.zeros((POP_SIZE_MAX, DIMENSION))

    history_f  = [[0.0] * HISTORY_LENGTH for _ in range(M_MAX)]
    history_cr = [[0.0] * HISTORY_LENGTH for _ in range(M_MAX)]

    f_arr    = [0.0] * POP_SIZE_MAX
    cr_arr   = [0.0] * POP_SIZE_MAX
    s_cr     = [0.0] * POP_SIZE_MAX
    s_f      = [0.0] * POP_SIZE_MAX
    delta_f  = [0.0] * POP_SIZE_MAX
    w_arr    = [0.0] * POP_SIZE_MAX

    cc_best_individual_index = [0] * M_MAX
    fitness_cc     = [[0.0] * POP_SIZE_MAX for _ in range(M_MAX)]
    fitness_cc_new = [[0.0] * POP_SIZE_MAX for _ in range(M_MAX)]

    range_arr = [0] * (M_MAX + 1)
    k         = [0] * M_MAX
    A         = [0] * M_MAX
    indeces   = list(range(DIMENSION))

    performance_cc       = [1.0] * len(SUBCOMPONENTS_POOL)
    performance_pop_size = [1.0] * len(POP_SIZE_POOL)

    # ── 初始化种群 / Initialize population ────────────────────────────────
    FEV            = FEV_global
    best_solution  = 1e300
    fitness_record: List[float] = []
    trigger        = 0

    initializePopulation(population, population_new, POP_SIZE_MAX, DIMENSION, a_bound, b_bound)

    t0 = time.time()

    # ══════════════════════════════════════════════════════════════════════
    # 外循环：自适应选择 M 和 pop_size / Outer loop: adaptive M and pop_size
    # 对应论文 Algorithm 2 第4-17行 / Corresponds to Algorithm 2 lines 4-17
    # ══════════════════════════════════════════════════════════════════════
    while FEV > 0:
        # 论文公式(3)：Boltzmann 选择 / Eq.(3): Boltzmann selection
        cc_index       = random_performance(performance_cc,       len(SUBCOMPONENTS_POOL), POWER)
        pop_size_index = random_performance(performance_pop_size, len(POP_SIZE_POOL),       POWER)

        pop_size  = POP_SIZE_POOL[pop_size_index]
        M         = SUBCOMPONENTS_POOL[cc_index]
        S         = DIMENSION // M
        piece_int = max(1, int(pop_size * PIECE))

        # 建立子分量边界 / Build subcomponent boundaries
        range_arr[0] = 0
        range_arr[M] = DIMENSION
        for i in range(1, M):
            range_arr[i] = range_arr[i - 1] + S

        # 随机置换变量索引（随机分组）/ Random variable permutation (random grouping)
        indecesSuccession(indeces, DIMENSION)
        randperm(indeces, DIMENSION)

        for i in range(M):
            A[i] = 0
            k[i] = 0
        for i in range(pop_size):
            s_cr[i] = s_f[i] = delta_f[i] = 0.0

        initializeHistory(history_f, history_cr, HISTORY_LENGTH, M)
        rnd_indecies(cc_best_individual_index, pop_size, M)

        # ── 初始化各子分量适应度（构建上下文向量）
        # Initialize subcomponent fitness (build context vector)
        for p in range(M):
            for i in range(pop_size):
                for j in range(range_arr[p], range_arr[p + 1]):
                    solution[indeces[j]] = population[i][indeces[j]]
                for p_cc in range(M):
                    if p != p_cc:
                        bi = cc_best_individual_index[p_cc]
                        for j in range(range_arr[p_cc], range_arr[p_cc + 1]):
                            solution[indeces[j]] = population[bi][indeces[j]]

                val = call_fun(fun, solution)
                fitness_cc[p][i] = val
                fitness_cc_new[p][i] = val
                FEV -= 1

                if FEV % krantost == 0 and trigger < 100:
                    fitness_record.append(best_solution)
                    trigger += 1

                if val < best_solution:
                    best_solution = val

            find_best_part_index(cc_best_individual_index, fitness_cc, p, pop_size)

        best_v = find_best_fitness_value(fitness_cc, M, pop_size)
        if best_v < best_solution:
            best_solution = best_v

        # ── 内循环：SHADE 优化 / Inner loop: SHADE optimization ────────────
        fev_cycle           = FEV_global // 50   # 每次内循环预算：60,000 FEV
        best_fitness_before = best_solution

        while fev_cycle > 0 and FEV > 0:
            for p in range(M):
                success = 0

                # 变异与交叉 / Mutation and crossover
                for i in range(pop_size):
                    pbest = findBestIndex(fitness_cc, pop_size, piece_int, p)
                    r_idx = int(C.RANDOM() * (HISTORY_LENGTH - 1))
                    cr_arr[i] = generation_CR(history_cr, r_idx, p)
                    f_arr[i]  = generation_F(history_f,  r_idx, p)
                    r1, r2 = chooseCrossoverIndecies(pbest, pop_size, A, p)

                    # current-to-pbest/1 变异 / current-to-pbest/1 mutation
                    if r2 < pop_size:
                        for j in range(range_arr[p], range_arr[p + 1]):
                            idx = indeces[j]
                            u[i][idx] = (population[i][idx]
                                         + f_arr[i] * (population[pbest][idx] - population[i][idx])
                                         + f_arr[i] * (population[r1][idx]    - population[r2][idx]))
                    else:
                        r2 -= pop_size
                        for j in range(range_arr[p], range_arr[p + 1]):
                            idx = indeces[j]
                            u[i][idx] = (population[i][idx]
                                         + f_arr[i] * (population[pbest][idx] - population[i][idx])
                                         + f_arr[i] * (population[r1][idx]    - archive[r2][idx]))

                    # 二项式交叉 / Binomial crossover
                    jrand = int(C.RANDOM() * ((range_arr[p+1] - range_arr[p]) + range_arr[p]))
                    for j in range(range_arr[p], range_arr[p + 1]):
                        idx = indeces[j]
                        if not (C.RANDOM() <= cr_arr[i] or j == jrand):
                            u[i][idx] = population[i][idx]

                    check_out_borders(u, population, i, a_bound, b_bound, range_arr, p, indeces)

                # 评估与选择 / Evaluation and selection
                for i in range(pop_size):
                    for j in range(range_arr[p], range_arr[p + 1]):
                        solution[indeces[j]] = u[i][indeces[j]]
                    for p_cc in range(M):
                        if p != p_cc:
                            bi = cc_best_individual_index[p_cc]
                            for j in range(range_arr[p_cc], range_arr[p_cc + 1]):
                                solution[indeces[j]] = population[bi][indeces[j]]

                    test_val = call_fun(fun, solution)
                    FEV      -= 1
                    fev_cycle -= 1

                    if FEV % krantost == 0 and trigger < 100:
                        fitness_record.append(best_solution)
                        trigger += 1

                    # 贪心选择 / Greedy selection
                    if test_val < fitness_cc[p][i]:
                        for j in range(range_arr[p], range_arr[p + 1]):
                            population_new[i][indeces[j]] = u[i][indeces[j]]
                        updateArchive(archive, population, i, ARCHIVE_SIZE, A, range_arr, p, indeces)
                        fitness_cc_new[p][i] = test_val
                        delta_f[success]     = abs(test_val - fitness_cc[p][i])
                        s_f[success]         = f_arr[i]
                        s_cr[success]        = cr_arr[i]
                        success             += 1

                # 更新历史记忆 / Update history memory
                Algorithm_1(delta_f, w_arr, s_cr, s_f, history_cr, history_f, k, success, HISTORY_LENGTH, p)
                success = 0

                for i in range(pop_size):
                    for j in range(range_arr[p], range_arr[p + 1]):
                        population[i][indeces[j]] = population_new[i][indeces[j]]
                    fitness_cc[p][i] = fitness_cc_new[p][i]

                min_pop_f = find_best_fitness_value(fitness_cc, M, pop_size)
                if min_pop_f < best_solution:
                    best_solution = min_pop_f

                find_best_part_index(cc_best_individual_index, fitness_cc, p, pop_size)

        # ── 更新性能分数（论文公式2）/ Update performance scores (paper Eq.2)
        best_fitness_after = best_solution
        if best_fitness_after != 0.0:
            perf = (best_fitness_before - best_fitness_after) / abs(best_fitness_after)
        else:
            perf = 0.0

        if math.isinf(perf) or math.isnan(perf):
            perf = 1e4
        perf = min(perf, 100.0)   # 防止 exp(7*perf) 溢出 / Prevent exp overflow
        if perf < 1e-4:
            perf = 1e-4

        performance_cc[cc_index]             = perf
        performance_pop_size[pop_size_index] = perf

    elapsed = time.time() - t0

    # 补齐到恰好 100 个检查点 / Pad to exactly 100 checkpoints
    while len(fitness_record) < 100:
        fitness_record.append(best_solution)

    logger.debug(f"试验完成 / Trial finished: best={best_solution:.4e}, time={elapsed:.1f}s")
    return fitness_record, elapsed


def main() -> None:
    """
    命令行入口：运行 CC-SHADE-ML 并保存结果 / CLI entry point: run CC-SHADE-ML and save results.

    支持通过参数指定函数 ID、运行次数和评估预算。
    Supports specifying function ID, number of runs, and evaluation budget via arguments.

    Usage:
        python run_ccshademl.py --id 1 --runs 25 --fev 3000000
    """
    parser = argparse.ArgumentParser(description="CC-SHADE-ML on LSGO CEC'2013")
    parser.add_argument("--id",   type=int, default=None,      help="函数 ID 1-15，默认全跑 / Function ID 1-15, default all")
    parser.add_argument("--runs", type=int, default=R_RUNS,    help="独立运行次数 / Number of independent runs")
    parser.add_argument("--fev",  type=int, default=FEV_GLOBAL,help="函数评估预算 / Function evaluation budget")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    benchmark = Benchmark()
    func_ids  = list(range(1, 16)) if args.id is None else [args.id]

    logger.info(f"CC-SHADE-ML 启动 / Starting: func_ids={func_ids}, runs={args.runs}, fev={args.fev:,}")
    print("=" * 60)
    print("CC-SHADE-ML  ×  demo benchmark framework")
    print(f"  函数 / Functions: {func_ids}  |  runs={args.runs}  |  FEV={args.fev:,}")
    print("=" * 60)

    for fun_id in func_ids:
        fun  = benchmark.get_function(fun_id)
        info = benchmark.get_info(fun_id)
        print(f"\n▶ f{fun_id}  bounds=[{info['lower']}, {info['upper']}]")

        all_records: List[List[float]] = []
        all_times:   List[float]       = []

        for z in range(args.runs):
            seed   = 42 + z * 1000   # 保证各次运行独立可复现 / Ensures independent reproducible runs
            record, elapsed = run_one_trial(fun, info, args.fev, seed=seed)
            all_records.append(record)
            all_times.append(elapsed)
            print(f"  Run {z+1:2d}/{args.runs}: best={record[-1]:.4e}  time={elapsed:.1f}s")

        # ── 统计汇总 / Statistical summary ───────────────────────────────
        finals = [r[-1] for r in all_records]
        print(f"\n  [f{fun_id} 汇总 / Summary]")
        print(f"  BEST  : {min(finals):.6e}")
        print(f"  MEAN  : {np.mean(finals):.6e}")
        print(f"  MEDIAN: {np.median(finals):.6e}")
        if args.runs > 1:
            print(f"  STD   : {np.std(finals, ddof=1):.6e}")
        print(f"  WORST : {max(finals):.6e}")
        print(f"  平均时间 / Avg time: {np.mean(all_times):.1f}s / run")

        # ── 保存结果 / Save results ───────────────────────────────────────
        out_path = f"results/f{fun_id}_ccshademl.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"CC-SHADE-ML  f{fun_id}  runs={args.runs}  FEV={args.fev}\n\n")
            f.write("每次运行 100 个记录点的 best_so_far / 100 best-so-far checkpoints per run:\n")
            for z, rec in enumerate(all_records):
                f.write(f"Run {z+1:2d}: " + ", ".join(f"{v:.4e}" for v in rec) + "\n")
            f.write(f"\nBEST  : {min(finals):.6e}\n")
            f.write(f"MEAN  : {np.mean(finals):.6e}\n")
            f.write(f"MEDIAN: {np.median(finals):.6e}\n")
            if args.runs > 1:
                f.write(f"STD   : {np.std(finals, ddof=1):.6e}\n")
            f.write(f"WORST : {max(finals):.6e}\n")

        print(f"  结果已保存 / Results saved: {out_path}")
        logger.info(f"f{fun_id} 完成 / f{fun_id} done: mean={np.mean(finals):.4e}")


if __name__ == "__main__":
    main()
