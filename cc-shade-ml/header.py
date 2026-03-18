"""
header.py — 算法核心工具函数模块 / Algorithm Core Utility Functions Module.

此模块直接翻译自 C++ Header.h，包含 CC-SHADE-ML 算法所需的全部工具函数。
This module is a direct translation of C++ Header.h, containing all utility
functions required by the CC-SHADE-ML algorithm.

主要组件 / Main Components:
    - 排序工具 / Sorting utilities: quickSort, bubble_sort, bubble_sort_indecies
    - 初始化 / Initialization: initializePopulation, initializeHistory
    - SHADE 核心 / SHADE core: generation_CR, generation_F, findBestIndex,
                                chooseCrossoverIndecies, Algorithm_1
    - CC 工具 / CC utilities: find_best_part_index, find_best_fitness_value
    - 随机工具 / Random utilities: rnd_indecies, indecesSuccession, randperm
    - 统计 / Statistics: mean_stat, min_stat, max_stat, median_stat, stddev_stat
    - 自适应选择 / Adaptive selection: random_performance

使用示例 / Usage Example:
    >>> from header import initializePopulation, random_performance
    >>> import numpy as np
    >>> pop = np.zeros((100, 1000))
    >>> pop_new = np.zeros((100, 1000))
    >>> initializePopulation(pop, pop_new, 100, 1000, -100, 100)

参考 / References:
    - Vakhnin & Sopov, CC-SHADE-ML, Algorithms 2022, 15, 451
    - Tanabe & Fukunaga, SHADE, CEC 2013
"""

from __future__ import annotations
import math
import logging
import numpy as np
import numpy.typing as npt
from typing import List, Tuple

from constants import RANDOM, randn, randc

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 排序工具 / Sorting Utilities
# ══════════════════════════════════════════════════════════════════════════════

def quickSort(arr: List[float], left: int, right: int) -> None:
    """
    原地快速排序（升序）/ In-place quicksort (ascending order).

    对应 C++ quickSort，用于排序适应度数组。
    Corresponds to C++ quickSort; used for sorting fitness arrays.

    Args:
        arr (List[float]): 待排序的列表 / List to sort in-place
        left (int):  左边界索引 / Left boundary index
        right (int): 右边界索引 / Right boundary index
    """
    i, j = left, right
    pivot = arr[(left + right) // 2]
    while i <= j:
        while arr[i] < pivot:
            i += 1
        while arr[j] > pivot:
            j -= 1
        if i <= j:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
            j -= 1
    if left < j:
        quickSort(arr, left, j)
    if i < right:
        quickSort(arr, i, right)


def bubble_sort(a: List[float], length: int) -> None:
    """
    冒泡排序（升序）/ Bubble sort (ascending order).

    对应 C++ bubble_sort，用于小数组的稳定排序。
    Corresponds to C++ bubble_sort; used for stable sorting of small arrays.

    Args:
        a (List[float]): 待排序的列表 / List to sort in-place
        length (int):    有效元素数量 / Number of valid elements
    """
    for j in range(length - 1):
        for i in range(length - j - 1):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]


def bubble_sort_indecies(a: List[float], index: List[int], length: int) -> None:
    """
    带索引跟踪的冒泡排序 / Bubble sort with index tracking.

    对应 C++ bubble_sort_indecies，排序值数组的同时保持索引对应。
    Corresponds to C++ bubble_sort_indecies; keeps index array synchronized.

    Args:
        a (List[float]):  待排序的值列表 / List of values to sort in-place
        index (List[int]): 与 a 对应的索引列表 / Index list synchronized with a
        length (int):      有效元素数量 / Number of valid elements
    """
    for j in range(length - 1):
        for i in range(length - j - 1):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                index[i], index[i + 1] = index[i + 1], index[i]


# ══════════════════════════════════════════════════════════════════════════════
# 初始化 / Initialization
# ══════════════════════════════════════════════════════════════════════════════

def initializePopulation(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    pop_size: int,
    N: int,
    a: float,
    b: float
) -> None:
    """
    随机初始化种群 / Randomly initialize the population.

    对应 C++ initializePopulation：用 [a, b] 均匀随机数同步填充 x 和 y。
    Corresponds to C++ initializePopulation: fills x and y with uniform
    random values in [a, b].

    Args:
        x (ndarray): 主种群数组，形状 (pop_size_max, N) / Main population, shape (pop_size_max, N)
        y (ndarray): 种群副本，形状 (pop_size_max, N) / Population copy, shape (pop_size_max, N)
        pop_size (int): 当前种群大小 / Current population size
        N (int):        问题维度 / Problem dimension
        a (float):      搜索空间下界 / Lower bound of search space
        b (float):      搜索空间上界 / Upper bound of search space
    """
    ba = b - a
    for i in range(pop_size):
        for j in range(N):
            val = RANDOM() * ba + a
            x[i][j] = val
            y[i][j] = val


def initializeHistory(
    HISTORY_F: List[List[float]],
    HISTORY_CR: List[List[float]],
    H: int,
    M: int
) -> None:
    """
    初始化 SHADE 历史记忆为 0.5 / Initialize SHADE history memory to 0.5.

    对应 C++ initializeHistory，将所有历史 F 和 CR 设为初始值 0.5。
    Corresponds to C++ initializeHistory; sets all history F and CR to 0.5.

    Args:
        HISTORY_F  (List[List[float]]): F 历史记忆，形状 (M_max, H) / F history, shape (M_max, H)
        HISTORY_CR (List[List[float]]): CR 历史记忆，形状 (M_max, H) / CR history, shape (M_max, H)
        H (int): 历史记忆长度 / History memory length
        M (int): 子分量数量 / Number of subcomponents
    """
    for i in range(M):
        for j in range(H):
            HISTORY_F[i][j] = 0.5
            HISTORY_CR[i][j] = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# SHADE 核心 / SHADE Core
# ══════════════════════════════════════════════════════════════════════════════

def findBestIndex(
    fitness_cc: List[List[float]],
    pop_size: int,
    piece_int: int,
    p: int
) -> int:
    """
    从前 piece_int 名中随机选择 pbest 索引 / Randomly select pbest index from top piece_int.

    对应 C++ findBestIndex：实现 current-to-pbest/1 变异策略中 pbest 的选取。
    Corresponds to C++ findBestIndex; implements pbest selection for
    current-to-pbest/1 mutation strategy.

    Args:
        fitness_cc (List[List[float]]): 各子分量的适应度矩阵 / Fitness matrix per subcomponent
        pop_size (int):  当前种群大小 / Current population size
        piece_int (int): pbest 候选数量（top p%）/ Number of pbest candidates (top p%)
        p (int):         当前子分量索引 / Current subcomponent index

    Returns:
        int: 选中的 pbest 个体索引 / Selected pbest individual index
    """
    f_sort = list(fitness_cc[p][:pop_size])
    bubble_sort(f_sort, pop_size)

    sorted_indices = []
    for i in range(pop_size):
        for j in range(pop_size):
            if f_sort[i] == fitness_cc[p][j]:
                sorted_indices.append(j)
                break

    k = max(1, piece_int)
    return sorted_indices[int(RANDOM() * k)]


def generation_CR(HISTORY_CR: List[List[float]], r: int, p: int) -> float:
    """
    从历史记忆采样交叉率 CR / Sample crossover rate CR from history memory.

    对应 C++ generation_CR：从 N(H_CR[p][r], 0.1) 采样并截断至 [0, 1]。
    Corresponds to C++ generation_CR; samples from N(H_CR[p][r], 0.1)
    and clips to [0, 1].

    Args:
        HISTORY_CR (List[List[float]]): CR 历史记忆 / CR history memory
        r (int): 历史记忆索引 / History memory index
        p (int): 子分量索引 / Subcomponent index

    Returns:
        float: 采样得到的 CR 值，范围 [0, 1] / Sampled CR value in [0, 1]
    """
    CR = randn(HISTORY_CR[p][r], 0.1)
    while CR > 1.0:
        CR = randn(HISTORY_CR[p][r], 0.1)
    if CR < 0.0:
        CR = 0.0
    return CR


def generation_F(HISTORY_F: List[List[float]], r: int, p: int) -> float:
    """
    从历史记忆采样缩放因子 F / Sample scale factor F from history memory.

    对应 C++ generation_F：从 Cauchy(H_F[p][r], 0.1) 采样并截断至 (0, 1]。
    Corresponds to C++ generation_F; samples from Cauchy(H_F[p][r], 0.1)
    and clips to (0, 1].

    Args:
        HISTORY_F (List[List[float]]): F 历史记忆 / F history memory
        r (int): 历史记忆索引 / History memory index
        p (int): 子分量索引 / Subcomponent index

    Returns:
        float: 采样得到的 F 值，范围 (0, 1] / Sampled F value in (0, 1]
    """
    F = randc(HISTORY_F[p][r], 0.1)
    while F < 0.0 or F > 1.0:
        F = randc(HISTORY_F[p][r], 0.1)
    return F


def chooseCrossoverIndecies(
    pbest: int,
    pop_size: int,
    A: List[int],
    p: int
) -> Tuple[int, int]:
    """
    选择变异向量的差分索引 r1, r2 / Select difference vector indices r1, r2.

    对应 C++ chooseCrossoverIndecies：保证 r1 ≠ r2 ≠ pbest，
    r2 可来自种群或外部档案。
    Corresponds to C++ chooseCrossoverIndecies; ensures r1 ≠ r2 ≠ pbest,
    r2 may come from population or external archive.

    Args:
        pbest (int):    当前 pbest 索引 / Current pbest index
        pop_size (int): 种群大小 / Population size
        A (List[int]):  各子分量的档案计数 / Archive count per subcomponent
        p (int):        子分量索引 / Subcomponent index

    Returns:
        Tuple[int, int]: (r1, r2) 差分向量索引对 / Difference vector index pair
    """
    r1 = int(RANDOM() * (pop_size - 1))
    r2 = int(RANDOM() * (pop_size - 1 + A[p]))
    while r1 == r2 or r1 == pbest or r2 == pbest:
        r1 = int(RANDOM() * (pop_size - 1))
        r2 = int(RANDOM() * (pop_size - 1 + A[p]))
    return r1, r2


def reset_k(k: List[int], H: int, p: int) -> None:
    """
    历史记忆指针回环 / Wrap history memory pointer.

    对应 C++ reset_k：实现环形缓冲区机制，k[p] >= H 时归零。
    Corresponds to C++ reset_k; implements ring buffer, resets k[p] when >= H.

    Args:
        k (List[int]): 各子分量的历史写指针 / History write pointer per subcomponent
        H (int):       历史记忆长度 / History memory length
        p (int):       子分量索引 / Subcomponent index
    """
    if k[p] >= H:
        k[p] = 0


def Algorithm_1(
    delta_f: List[float],
    w: List[float],
    S_CR: List[float],
    S_F: List[float],
    HISTORY_CR: List[List[float]],
    HISTORY_F: List[List[float]],
    k: List[int],
    success: int,
    H: int,
    p: int
) -> int:
    """
    用成功参数更新 SHADE 历史记忆 / Update SHADE history memory with successful parameters.

    对应 C++ Algorithm_1：使用本代成功的 S_F、S_CR，通过 Lehmer 加权均值
    更新历史记忆。无论是否有成功，k 指针每轮都前进（环形缓冲区机制）。
    Corresponds to C++ Algorithm_1; updates history memory using successful
    S_F and S_CR via weighted Lehmer mean. The k pointer always advances
    each round regardless of success (ring buffer mechanism).

    Args:
        delta_f (List[float]):     各成功个体的适应度改进量 / Fitness improvement per success
        w (List[float]):           权重缓冲区（会被本函数写入）/ Weight buffer (written by this function)
        S_CR (List[float]):        成功的 CR 值列表 / List of successful CR values
        S_F (List[float]):         成功的 F 值列表 / List of successful F values
        HISTORY_CR (List[List]):   CR 历史记忆 / CR history memory
        HISTORY_F (List[List]):    F 历史记忆 / F history memory
        k (List[int]):             历史写指针 / History write pointer
        success (int):             本代成功次数 / Number of successes this generation
        H (int):                   历史记忆长度 / History memory length
        p (int):                   子分量索引 / Subcomponent index

    Returns:
        int: 0（与 C++ 接口保持一致）/ 0 (consistent with C++ interface)
    """
    if success == 0:
        k[p] += 1
        reset_k(k, H, p)
        return 0

    sum1 = sum(delta_f[i] for i in range(success))
    if sum1 == 0.0:
        k[p] += 1
        reset_k(k, H, p)
        return 0

    sum_scr = sum(S_CR[i] for i in range(success))
    sum_sf  = sum(S_F[i]  for i in range(success))

    if sum_scr == 0.0 and sum_sf == 0.0:
        k[p] += 1
        reset_k(k, H, p)
        return 0

    total_df = sum1
    for i in range(success):
        w[i] = delta_f[i] / total_df

    if sum_scr == 0.0 and sum_sf != 0.0:
        # 只更新 F / Update F only
        wf2 = sum(w[i] * S_F[i] * S_F[i] for i in range(success))
        wf  = sum(w[i] * S_F[i]           for i in range(success))
        mean_f = (wf2 / wf) if wf != 0.0 else 0.5
        if mean_f != mean_f:  # NaN 检查 / NaN check
            mean_f = 0.5
        HISTORY_F[p][k[p]]  = mean_f
        HISTORY_CR[p][k[p]] = 0.0
        k[p] += 1
        reset_k(k, H, p)
        return 0

    if sum_scr != 0.0 and sum_sf == 0.0:
        # 只更新 CR / Update CR only
        mean_cr = sum(w[i] * S_CR[i] for i in range(success))
        if mean_cr != mean_cr:
            mean_cr = 0.5
        HISTORY_CR[p][k[p]] = mean_cr
        HISTORY_F[p][k[p]]  = 0.0
        k[p] += 1
        reset_k(k, H, p)
        return 0

    # 同时更新 F 和 CR / Update both F and CR
    wf2 = sum(w[i] * S_F[i] * S_F[i] for i in range(success))
    wf  = sum(w[i] * S_F[i]           for i in range(success))
    mean_f = (wf2 / wf) if wf != 0.0 else 0.5
    if mean_f != mean_f:
        mean_f = 0.5

    mean_cr = sum(w[i] * S_CR[i] for i in range(success))
    if mean_cr != mean_cr:
        mean_cr = 0.5

    HISTORY_F[p][k[p]]  = mean_f
    HISTORY_CR[p][k[p]] = mean_cr
    k[p] += 1
    reset_k(k, H, p)
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# 越界修复 / Border Repair
# ══════════════════════════════════════════════════════════════════════════════

def check_out_borders(
    u: npt.NDArray[np.float64],
    population: npt.NDArray[np.float64],
    i: int,
    a: float,
    b: float,
    range_arr: List[int],
    p: int,
    indeces: List[int]
) -> None:
    """
    修复越界的变异向量 / Repair mutant vector that exceeds search bounds.

    对应 C++ check_out_borders：超界则取父代与边界中点，直到合法为止。
    Corresponds to C++ check_out_borders; uses midpoint of parent and boundary
    until value is within bounds.

    Args:
        u (ndarray):          变异向量数组 / Mutant vector array, shape (pop_size_max, N)
        population (ndarray): 当前种群 / Current population, shape (pop_size_max, N)
        i (int):              个体索引 / Individual index
        a (float):            搜索空间下界 / Lower bound
        b (float):            搜索空间上界 / Upper bound
        range_arr (List[int]): 子分量边界数组 / Subcomponent boundary array
        p (int):              子分量索引 / Subcomponent index
        indeces (List[int]):  随机置换后的变量索引 / Permuted variable indices
    """
    for j in range(range_arr[p], range_arr[p + 1]):
        idx = indeces[j]
        while u[i][idx] < a:
            u[i][idx] = (a + population[i][idx]) / 2.0
        while u[i][idx] > b:
            u[i][idx] = (b + population[i][idx]) / 2.0


# ══════════════════════════════════════════════════════════════════════════════
# 档案更新 / Archive Update
# ══════════════════════════════════════════════════════════════════════════════

def updateArchive(
    archive: npt.NDArray[np.float64],
    population: npt.NDArray[np.float64],
    i: int,
    archive_size: int,
    A: List[int],
    range_arr: List[int],
    p: int,
    indeces: List[int]
) -> None:
    """
    将被替换个体加入外部档案 / Add replaced individual to external archive.

    对应 C++ updateArchive：档案未满时顺序添加，满时随机替换。
    档案用于增加种群多样性，变异时可作为差分向量来源。
    Corresponds to C++ updateArchive; appends when not full, randomly
    replaces when full. Archive maintains diversity for mutation.

    Args:
        archive (ndarray):    外部档案数组 / External archive array
        population (ndarray): 当前种群 / Current population
        i (int):              被替换个体的索引 / Index of replaced individual
        archive_size (int):   档案最大容量 / Maximum archive capacity
        A (List[int]):        各子分量的当前档案大小 / Current archive size per subcomponent
        range_arr (List[int]): 子分量边界数组 / Subcomponent boundary array
        p (int):              子分量索引 / Subcomponent index
        indeces (List[int]):  随机置换后的变量索引 / Permuted variable indices
    """
    if A[p] >= archive_size:
        idx = int(RANDOM() * (archive_size - 1))
        for j in range(range_arr[p], range_arr[p + 1]):
            archive[idx][indeces[j]] = population[i][indeces[j]]
    else:
        for j in range(range_arr[p], range_arr[p + 1]):
            archive[A[p]][indeces[j]] = population[i][indeces[j]]
        A[p] += 1


# ══════════════════════════════════════════════════════════════════════════════
# CC 辅助 / Cooperative Coevolution Utilities
# ══════════════════════════════════════════════════════════════════════════════

def find_best_part_index(
    cc_best_individual_index: List[int],
    fitness_cc: List[List[float]],
    p: int,
    pop_size: int
) -> None:
    """
    找到子分量 p 中适应度最优的个体索引 / Find the best individual index in subcomponent p.

    对应 C++ find_best_part_index，更新上下文向量中子分量 p 的最优代表。
    Corresponds to C++ find_best_part_index; updates the best representative
    for subcomponent p in the context vector.

    Args:
        cc_best_individual_index (List[int]): 各子分量最优个体索引 / Best index per subcomponent
        fitness_cc (List[List[float]]): 各子分量的适应度矩阵 / Fitness matrix per subcomponent
        p (int):        子分量索引 / Subcomponent index
        pop_size (int): 当前种群大小 / Current population size
    """
    best = 0
    min_f = fitness_cc[p][0]
    for j in range(pop_size):
        if fitness_cc[p][j] < min_f:
            min_f = fitness_cc[p][j]
            best = j
    cc_best_individual_index[p] = best


def find_best_fitness_value(
    fitness_cc: List[List[float]],
    M: int,
    pop_size: int
) -> float:
    """
    获取所有子分量中的全局最优适应度值 / Get global best fitness across all subcomponents.

    对应 C++ find_best_fitness_value，遍历所有子分量和个体找最小值。
    Corresponds to C++ find_best_fitness_value; finds minimum across all
    subcomponents and individuals.

    Args:
        fitness_cc (List[List[float]]): 各子分量的适应度矩阵 / Fitness matrix per subcomponent
        M (int):        子分量数量 / Number of subcomponents
        pop_size (int): 当前种群大小 / Current population size

    Returns:
        float: 全局最优适应度值 / Global best fitness value
    """
    minn = fitness_cc[0][0]
    for i in range(M):
        for j in range(pop_size):
            if fitness_cc[i][j] < minn:
                minn = fitness_cc[i][j]
    return minn


# ══════════════════════════════════════════════════════════════════════════════
# 随机工具 / Random Utilities
# ══════════════════════════════════════════════════════════════════════════════

def rnd_indecies(vec: List[int], pop: int, M: int) -> None:
    """
    用随机整数初始化最优个体索引数组 / Initialize best individual index array with random integers.

    对应 C++ rnd_indecies，为各子分量随机分配初始最优个体索引。
    Corresponds to C++ rnd_indecies; randomly assigns initial best individual
    index for each subcomponent.

    Args:
        vec (List[int]): 待填充的索引数组 / Index array to fill
        pop (int):       种群大小 / Population size
        M (int):         子分量数量 / Number of subcomponents
    """
    for i in range(M):
        vec[i] = int(RANDOM() * (pop - 1))


def indecesSuccession(x: List[int], N: int) -> None:
    """
    将索引数组初始化为 [0, 1, ..., N-1] / Initialize index array to [0, 1, ..., N-1].

    对应 C++ indecesSuccession，用于 randperm 之前的初始化。
    Corresponds to C++ indecesSuccession; called before randperm.

    Args:
        x (List[int]): 待初始化的索引数组 / Index array to initialize
        N (int):       数组长度（问题维度）/ Array length (problem dimension)
    """
    for i in range(N):
        x[i] = i


def randperm(x: List[int], N: int) -> None:
    """
    对索引数组进行随机置换 / Randomly permute the index array in-place.

    对应 C++ randperm：每个位置与一个不同位置交换，实现随机分组。
    Corresponds to C++ randperm; each position is swapped with a different
    random position to achieve random grouping.

    Args:
        x (List[int]): 待置换的索引数组（原地修改）/ Index array to permute in-place
        N (int):       数组长度 / Array length
    """
    for i in range(N):
        a = int(RANDOM() * (N - 1))
        while a < 0 or a > (N - 1) or a == i or a == N:
            a = int(RANDOM() * (N - 1))
        x[i], x[a] = x[a], x[i]


# ══════════════════════════════════════════════════════════════════════════════
# 统计函数 / Statistics Functions
# ══════════════════════════════════════════════════════════════════════════════

def mean_stat(x: List[List[float]], gener: int, R: int) -> float:
    """
    计算第 gener 代所有运行的均值 / Compute mean across all runs at generation gener.

    Args:
        x (List[List[float]]): 收敛数据，形状 (R, generations) / Convergence data
        gener (int): 代数索引 / Generation index
        R (int):     独立运行次数 / Number of independent runs

    Returns:
        float: 均值 / Mean value
    """
    return sum(x[i][gener] for i in range(R)) / R


def min_stat(x: List[List[float]], gener: int, R: int) -> float:
    """
    计算第 gener 代所有运行的最小值 / Compute minimum across all runs at generation gener.

    Args:
        x (List[List[float]]): 收敛数据 / Convergence data
        gener (int): 代数索引 / Generation index
        R (int):     独立运行次数 / Number of independent runs

    Returns:
        float: 最小值 / Minimum value
    """
    return min(x[i][gener] for i in range(R))


def max_stat(x: List[List[float]], gener: int, R: int) -> float:
    """
    计算第 gener 代所有运行的最大值 / Compute maximum across all runs at generation gener.

    Args:
        x (List[List[float]]): 收敛数据 / Convergence data
        gener (int): 代数索引 / Generation index
        R (int):     独立运行次数 / Number of independent runs

    Returns:
        float: 最大值 / Maximum value
    """
    return max(x[i][gener] for i in range(R))


def median_stat(x: List[List[float]], gener: int, R: int) -> float:
    """
    计算第 gener 代所有运行的中位数 / Compute median across all runs at generation gener.

    Args:
        x (List[List[float]]): 收敛数据 / Convergence data
        gener (int): 代数索引 / Generation index
        R (int):     独立运行次数 / Number of independent runs

    Returns:
        float: 中位数 / Median value
    """
    fitness = sorted(x[i][gener] for i in range(R))
    return fitness[(R - 1) // 2]


def stddev_stat(x: List[List[float]], gener: int, R: int, M: float) -> float:
    """
    计算第 gener 代所有运行的标准差 / Compute standard deviation across all runs.

    Args:
        x (List[List[float]]): 收敛数据 / Convergence data
        gener (int): 代数索引 / Generation index
        R (int):     独立运行次数 / Number of independent runs
        M (float):   均值（由调用方提供）/ Mean value (provided by caller)

    Returns:
        float: 标准差 / Standard deviation
    """
    s = sum((M - x[i][gener]) ** 2 for i in range(R))
    return math.sqrt(s / (R - 1)) if R > 1 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 其他工具 / Other Utilities
# ══════════════════════════════════════════════════════════════════════════════

def max_number(x: List[int], length: int) -> int:
    """
    返回整数数组的最大值 / Return the maximum value of an integer array.

    对应 C++ max_number，用于确定最大子分量数 M_MAX。
    Corresponds to C++ max_number; used to determine M_MAX.

    Args:
        x (List[int]): 整数数组 / Integer array
        length (int):  有效元素数量 / Number of valid elements

    Returns:
        int: 最大值 / Maximum value
    """
    return max(x[:length])


# ══════════════════════════════════════════════════════════════════════════════
# 自适应选择 / Adaptive Selection
# ══════════════════════════════════════════════════════════════════════════════

def random_performance(
    performance: List[float],
    N: int,
    power: float
) -> int:
    """
    基于 Boltzmann 分布的自适应配置选择 / Adaptive configuration selection via Boltzmann distribution.

    对应 C++ random_performance：计算每个配置的 softmax 概率，用轮盘赌选择。
    性能越高的配置被选中的概率指数级增加，power 控制选择压力（论文推荐 7.0）。
    Corresponds to C++ random_performance; computes softmax probabilities and
    uses roulette wheel selection. Higher performance exponentially increases
    selection probability; power controls selection pressure (paper recommends 7.0).

    Args:
        performance (List[float]): 各配置的历史性能分数 / Historical performance scores
        N (int):                   配置数量 / Number of configurations
        power (float):             Boltzmann 选择温度参数 / Boltzmann selection temperature

    Returns:
        int: 被选中的配置索引 / Selected configuration index
    """
    rnd = RANDOM()
    performance2 = []

    for i in range(N):
        exponent = power * performance[i]
        if exponent > 709.0:       # 防止 math.exp 溢出 / Prevent math.exp overflow
            val = 1e300
        elif exponent < -709.0:
            val = 0.0
        else:
            val = math.exp(exponent)
        if math.isinf(val):
            val = 1e300
        performance2.append(val)

    total = sum(performance2)
    for i in range(N):
        performance2[i] /= total

    # 累积概率（轮盘赌）/ Cumulative probability (roulette wheel)
    for i in range(1, N):
        performance2[i] += performance2[i - 1]

    index = -1
    for i in range(N - 1):
        if rnd > performance2[i] and rnd <= performance2[i + 1]:
            index = i + 1

    if rnd >= 0.0 and rnd <= performance2[0]:
        index = 0
    if rnd >= performance2[N - 1]:
        index = N - 1

    return index
