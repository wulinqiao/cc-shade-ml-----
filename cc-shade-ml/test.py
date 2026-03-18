"""
CC-SHADE-ML 算法单元测试 / Unit Tests for CC-SHADE-ML Algorithm.

此模块对 CC-SHADE-ML 算法的各核心组件进行单元测试。
This module performs unit tests on the core components of the CC-SHADE-ML algorithm.

主要测试组件 / Main Test Components:
    - TestRandomGeneration:   随机数生成 / Random number generation
    - TestPopulationInit:     种群初始化 / Population initialization
    - TestVariableGrouping:   变量随机分组 / Variable random grouping
    - TestSHADEParameters:    SHADE 参数采样 / SHADE parameter sampling
    - TestBorderCheck:        越界修复 / Border violation repair
    - TestBoltzmannSelection: Boltzmann 自适应选择 / Boltzmann adaptive selection
    - TestHistoryUpdate:      历史记忆更新 / History memory update
    - TestBenchmarkInterface: 基准函数接口 / Benchmark function interface
    - TestSmokeRun:           完整算法冒烟测试 / Full algorithm smoke test

使用示例 / Usage Example:
    python test.py
    python -m unittest test -v

参考 / References:
    - Vakhnin & Sopov, CC-SHADE-ML, Algorithms 2022, 15, 451
"""

import os
import sys
import math
import unittest
import numpy as np
import numpy.typing as npt
from typing import List

# ── 路径修复 / Path Fix ───────────────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

import constants as C
from header import (
    initializePopulation, initializeHistory,
    generation_CR, generation_F,
    check_out_borders, Algorithm_1,
    indecesSuccession, randperm,
    random_performance,
)
from benchmark.cec2013lsgo.cec2013 import Benchmark

# ── 算法参数常量 / Algorithm Parameter Constants ──────────────────────────────
DIMENSION        = 1000
POPULATION_SIZE  = 100
HISTORY_LENGTH   = 6
MAX_SUBCOMPS     = 50
LOWER_BOUND      = -100
UPPER_BOUND      = 100
DEFAULT_SEED     = 42


class TestRandomGeneration(unittest.TestCase):
    """
    随机数生成模块测试 / Tests for random number generation module.

    验证 RANDOM()、randn() 的范围、分布和可复现性。
    Validates range, distribution, and reproducibility of RANDOM() and randn().
    """

    def setUp(self) -> None:
        """每个测试前重置种子 / Reset seed before each test."""
        C.set_seed(DEFAULT_SEED)

    def test_random_range(self) -> None:
        """测试 RANDOM() 输出范围在 (0,1] / Test RANDOM() output range is in (0,1]."""
        samples: List[float] = [C.RANDOM() for _ in range(10000)]
        self.assertTrue(all(0 < x <= 1 for x in samples),
                        "RANDOM() 存在超出 (0,1] 范围的值 / Value out of (0,1] range")

    def test_random_uniformity(self) -> None:
        """测试 RANDOM() 均值接近 0.5 / Test RANDOM() mean is close to 0.5."""
        samples: List[float] = [C.RANDOM() for _ in range(10000)]
        self.assertAlmostEqual(np.mean(samples), 0.5, delta=0.02,
                               msg=f"均值偏差过大 / Mean too far from 0.5: {np.mean(samples):.4f}")

    def test_seed_reproducibility(self) -> None:
        """测试相同种子产生相同序列 / Test same seed produces identical sequence."""
        C.set_seed(99)
        seq_a: List[float] = [C.RANDOM() for _ in range(5)]
        C.set_seed(99)
        seq_b: List[float] = [C.RANDOM() for _ in range(5)]
        self.assertEqual(seq_a, seq_b,
                         "相同种子应产生相同序列 / Same seed should produce identical sequence")

    def test_different_seeds_differ(self) -> None:
        """测试不同种子产生不同序列 / Test different seeds produce different sequences."""
        C.set_seed(1)
        seq_a: List[float] = [C.RANDOM() for _ in range(5)]
        C.set_seed(2)
        seq_b: List[float] = [C.RANDOM() for _ in range(5)]
        self.assertNotEqual(seq_a, seq_b,
                            "不同种子应产生不同序列 / Different seeds should produce different sequences")

    def test_randn_mean(self) -> None:
        """测试 randn(0.5, 0.1) 均值接近 0.5 / Test randn(0.5, 0.1) mean is close to 0.5."""
        samples: List[float] = [C.randn(0.5, 0.1) for _ in range(5000)]
        self.assertAlmostEqual(np.mean(samples), 0.5, delta=0.01,
                               msg=f"randn 均值偏差 / randn mean deviation: {np.mean(samples):.4f}")


class TestPopulationInit(unittest.TestCase):
    """
    种群初始化测试 / Tests for population initialization.

    验证种群形状、边界合法性和多样性。
    Validates population shape, boundary legality, and diversity.
    """

    def setUp(self) -> None:
        """初始化测试用种群 / Initialize test population."""
        C.set_seed(DEFAULT_SEED)
        self.pop     = np.zeros((POPULATION_SIZE, DIMENSION))
        self.pop_new = np.zeros((POPULATION_SIZE, DIMENSION))
        initializePopulation(self.pop, self.pop_new, POPULATION_SIZE, DIMENSION, LOWER_BOUND, UPPER_BOUND)

    def test_shape(self) -> None:
        """测试种群形状正确 / Test population shape is correct."""
        self.assertEqual(self.pop.shape, (POPULATION_SIZE, DIMENSION))

    def test_within_bounds(self) -> None:
        """测试所有个体在搜索边界内 / Test all individuals are within search bounds."""
        self.assertTrue(np.all(self.pop >= LOWER_BOUND) and np.all(self.pop <= UPPER_BOUND),
                        "存在超出边界的个体 / Individual out of bounds")

    def test_diversity(self) -> None:
        """测试种群具有多样性 / Test population has diversity."""
        self.assertGreater(np.std(self.pop), 10.0,
                           f"种群多样性不足 / Insufficient diversity: std={np.std(self.pop):.2f}")

    def test_pop_new_equals_pop(self) -> None:
        """测试 population_new 初始与 population 相同 / Test population_new equals population initially."""
        np.testing.assert_array_equal(self.pop, self.pop_new)


class TestVariableGrouping(unittest.TestCase):
    """
    变量随机分组测试 / Tests for variable random grouping.

    验证 randperm 产生合法置换，保证 CC 分组机制的正确性。
    Validates randperm produces a valid permutation for the CC grouping mechanism.
    """

    def setUp(self) -> None:
        """初始化索引数组 / Initialize index array."""
        self.indeces: List[int] = list(range(DIMENSION))

    def test_succession_initialization(self) -> None:
        """测试 indecesSuccession 生成 0~999 / Test indecesSuccession generates 0~999."""
        indecesSuccession(self.indeces, DIMENSION)
        self.assertEqual(self.indeces, list(range(DIMENSION)))

    def test_randperm_completeness(self) -> None:
        """测试 randperm 不丢失也不重复任何索引 / Test randperm has no missing or duplicate indices."""
        indecesSuccession(self.indeces, DIMENSION)
        C.set_seed(DEFAULT_SEED)
        randperm(self.indeces, DIMENSION)
        self.assertEqual(sorted(self.indeces), list(range(DIMENSION)),
                         "randperm 后索引不完整 / Indices incomplete after randperm")

    def test_randperm_shuffles(self) -> None:
        """测试 randperm 确实打乱了顺序 / Test randperm actually shuffles the order."""
        indecesSuccession(self.indeces, DIMENSION)
        C.set_seed(DEFAULT_SEED)
        randperm(self.indeces, DIMENSION)
        self.assertNotEqual(self.indeces, list(range(DIMENSION)),
                            "randperm 未改变顺序 / randperm did not shuffle")


class TestSHADEParameters(unittest.TestCase):
    """
    SHADE 参数采样测试 / Tests for SHADE parameter sampling.

    验证 F 和 CR 采样范围合法，历史记忆初始化正确。
    Validates F and CR sampling ranges and history memory initialization.
    """

    def setUp(self) -> None:
        """初始化历史记忆 / Initialize history memory."""
        self.hist_f  = [[0.0] * HISTORY_LENGTH for _ in range(MAX_SUBCOMPS)]
        self.hist_cr = [[0.0] * HISTORY_LENGTH for _ in range(MAX_SUBCOMPS)]
        initializeHistory(self.hist_f, self.hist_cr, HISTORY_LENGTH, MAX_SUBCOMPS)
        C.set_seed(DEFAULT_SEED)

    def test_history_initialization(self) -> None:
        """测试历史记忆初始化为 0.5 / Test history memory initialized to 0.5."""
        self.assertTrue(all(
            self.hist_f[p][r] == 0.5
            for p in range(MAX_SUBCOMPS) for r in range(HISTORY_LENGTH)
        ), "历史记忆初始值应为 0.5 / History should be initialized to 0.5")

    def test_cr_range(self) -> None:
        """测试 CR 采样值在 [0,1] / Test CR sampling values are in [0,1]."""
        cr_vals: List[float] = [generation_CR(self.hist_cr, 0, 0) for _ in range(1000)]
        self.assertTrue(all(0 <= v <= 1 for v in cr_vals),
                        "CR 存在超出 [0,1] 的值 / CR value out of [0,1]")

    def test_f_range(self) -> None:
        """测试 F 采样值在 (0,1] / Test F sampling values are in (0,1]."""
        f_vals: List[float] = [generation_F(self.hist_f, 0, 0) for _ in range(1000)]
        self.assertTrue(all(0 < v <= 1 for v in f_vals),
                        "F 存在超出 (0,1] 的值 / F value out of (0,1]")


class TestBorderCheck(unittest.TestCase):
    """
    越界修复测试 / Tests for border violation repair.

    验证变异后超界的个体被正确修复到搜索空间内。
    Validates that mutant vectors exceeding bounds are correctly repaired.
    """

    def test_repair_within_bounds(self) -> None:
        """测试修复后所有值在边界内 / Test all values are within bounds after repair."""
        pop_size = 10
        u_test   = np.random.uniform(-300, 300, (pop_size, DIMENSION))
        pop_test = np.random.uniform(LOWER_BOUND, UPPER_BOUND, (pop_size, DIMENSION))
        indeces  = list(range(DIMENSION))
        range_t  = [0, DIMENSION]

        for i in range(pop_size):
            check_out_borders(u_test, pop_test, i, LOWER_BOUND, UPPER_BOUND, range_t, 0, indeces)

        self.assertTrue(
            np.all(u_test >= LOWER_BOUND) and np.all(u_test <= UPPER_BOUND),
            "越界修复后仍有值超出边界 / Values still out of bounds after repair"
        )


class TestBoltzmannSelection(unittest.TestCase):
    """
    Boltzmann 自适应选择测试 / Tests for Boltzmann adaptive selection.

    验证高性能配置被选中的频率更高，均匀性能时选择近似均匀。
    Validates high-performance configs are selected more often and
    equal performance yields near-uniform selection.
    """

    def test_high_performance_preferred(self) -> None:
        """测试高性能配置被优先选中 / Test high-performance config is preferentially selected."""
        performance = [0.001, 0.001, 10.0, 0.001]
        C.set_seed(DEFAULT_SEED)
        counts = [0, 0, 0, 0]
        for _ in range(2000):
            counts[random_performance(performance, 4, 7.0)] += 1
        self.assertEqual(counts[2], max(counts),
                         f"高性能配置未被最多选中 / High-perf not most selected: {counts}")

    def test_all_configs_reachable(self) -> None:
        """
        测试性能相同时所有配置都能被选中 / Test all configs reachable when performance is equal.

        注意：性能差异极大时（如0.001 vs 10.0），Boltzmann以接近100%概率选最优配置，
        其他配置不可达是正确行为，故本测试使用相同性能值来验证可达性。
        Note: With extreme differences, near-100% selection of best config is correct.
        This test uses equal performance to verify reachability.
        """
        performance = [1.0, 1.0, 1.0, 1.0]
        C.set_seed(DEFAULT_SEED)
        counts = [0, 0, 0, 0]
        for _ in range(4000):
            counts[random_performance(performance, 4, 7.0)] += 1
        self.assertTrue(all(c > 0 for c in counts),
                        f"性能相同时所有配置应可达 / All configs should be reachable: {counts}")

    def test_equal_performance_uniform(self) -> None:
        """测试性能相同时选择近似均匀 / Test near-uniform selection when performance is equal."""
        performance = [1.0, 1.0, 1.0, 1.0]
        C.set_seed(DEFAULT_SEED)
        counts = [0, 0, 0, 0]
        for _ in range(4000):
            counts[random_performance(performance, 4, 7.0)] += 1
        for i, c in enumerate(counts):
            self.assertGreater(c / 4000, 0.15,
                               f"配置{i}选中比例过低 / Config {i} ratio too low: {c/4000:.3f}")
            self.assertLess(c / 4000, 0.35,
                            f"配置{i}选中比例过高 / Config {i} ratio too high: {c/4000:.3f}")


class TestHistoryUpdate(unittest.TestCase):
    """
    Algorithm_1 历史记忆更新测试 / Tests for Algorithm_1 history memory update.

    验证有成功时记忆被正确更新，无成功时记忆不变。
    Validates memory is updated on success and unchanged on no success.
    """

    def setUp(self) -> None:
        """初始化测试用历史记忆 / Initialize test history memory."""
        self.hist_f  = [[0.5] * HISTORY_LENGTH for _ in range(5)]
        self.hist_cr = [[0.5] * HISTORY_LENGTH for _ in range(5)]
        self.k       = [0] * 5
        self.w       = [0.0] * POPULATION_SIZE

    def test_update_on_success(self) -> None:
        """测试有成功时历史F被更新 / Test history F is updated when success > 0."""
        success = 3
        delta_f = [1.0, 2.0, 3.0] + [0.0] * (POPULATION_SIZE - 3)
        s_f     = [0.8, 0.6, 0.7] + [0.0] * (POPULATION_SIZE - 3)
        s_cr    = [0.9, 0.4, 0.6] + [0.0] * (POPULATION_SIZE - 3)
        old_val = self.hist_f[0][0]
        Algorithm_1(delta_f, self.w, s_cr, s_f, self.hist_cr, self.hist_f, self.k, success, HISTORY_LENGTH, 0)
        self.assertNotEqual(self.hist_f[0][0], old_val,
                            "有成功时历史F应被更新 / History F should be updated on success")
        self.assertGreater(self.hist_f[0][0], 0)
        self.assertLessEqual(self.hist_f[0][0], 1)

    def test_k_pointer_advances(self) -> None:
        """测试有成功时 k 指针前进 / Test k pointer advances on success."""
        success = 2
        delta_f = [1.0, 2.0] + [0.0] * (POPULATION_SIZE - 2)
        s_f     = [0.5, 0.6] + [0.0] * (POPULATION_SIZE - 2)
        s_cr    = [0.5, 0.4] + [0.0] * (POPULATION_SIZE - 2)
        Algorithm_1(delta_f, self.w, s_cr, s_f, self.hist_cr, self.hist_f, self.k, success, HISTORY_LENGTH, 0)
        self.assertEqual(self.k[0], 1,
                         "有成功时 k 应前进1 / k should advance by 1 on success")

    def test_no_update_on_no_success(self) -> None:
        """
        测试无成功时历史值不变但 k 指针仍前进 / Test history value unchanged but k advances on no success.

        Algorithm_1 的设计：无论是否有成功，k 指针每轮都前进（环形缓冲区机制）。
        无成功时只跳过历史值的更新，但不跳过指针移动。
        Design: k always advances each round (ring buffer mechanism).
        On no success, history values are NOT updated, but k still moves.
        """
        Algorithm_1(
            [0.0]*POPULATION_SIZE, self.w,
            [0.0]*POPULATION_SIZE, [0.0]*POPULATION_SIZE,
            self.hist_cr, self.hist_f, self.k, 0, HISTORY_LENGTH, 0
        )
        # 历史值不应改变 / History values should NOT change
        self.assertEqual(self.hist_f[0][0], 0.5,
                         "无成功时历史F不应改变 / History F should not change on no success")
        # k 指针应前进（这是正确的环形缓冲区行为）/ k should still advance (correct ring buffer behavior)
        self.assertEqual(self.k[0], 1,
                         "k 指针每轮都应前进，无论是否有成功 / k should always advance each round")


class TestBenchmarkInterface(unittest.TestCase):
    """
    基准函数接口测试 / Tests for benchmark function interface.

    验证 Benchmark 类能正确提供目标函数和问题信息。
    Validates Benchmark class correctly provides objective functions and problem info.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """初始化 Benchmark（只加载一次）/ Initialize Benchmark once."""
        cls.benchmark = Benchmark()
        cls.fun       = cls.benchmark.get_function(1)
        cls.info      = cls.benchmark.get_info(1)

    def test_info_fields(self) -> None:
        """测试 get_info 包含必要字段 / Test get_info contains required fields."""
        for field in ['lower', 'upper', 'dimension']:
            self.assertIn(field, self.info, f"缺少字段 / Missing field: {field}")

    def test_dimension(self) -> None:
        """测试问题维度为 1000 / Test problem dimension is 1000."""
        self.assertEqual(self.info['dimension'], DIMENSION)

    def test_valid_bounds(self) -> None:
        """测试边界合法 / Test bounds are valid."""
        self.assertLess(self.info['lower'], self.info['upper'])

    def test_function_returns_finite(self) -> None:
        """测试函数返回有限正数 / Test function returns a finite positive value."""
        x = np.zeros(DIMENSION)
        result = self.fun(x)
        val = float(result[0]) if hasattr(result, '__len__') else float(result)
        self.assertTrue(math.isfinite(val) and val > 0,
                        f"f(zeros) 应返回有限正数 / Should be finite positive: {val:.4e}")


class TestSmokeRun(unittest.TestCase):
    """
    完整算法冒烟测试 / Full algorithm smoke test.

    以极少的 FEV 跑一次完整流程，验证算法端到端可运行。
    Runs one complete trial with minimal FEV to verify end-to-end execution.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """运行一次短实验 / Run one short trial."""
        from run_ccshademl import run_one_trial
        benchmark = Benchmark()
        fun  = benchmark.get_function(1)
        info = benchmark.get_info(1)
        cls.record, cls.elapsed = run_one_trial(fun, info, FEV_global=10000, seed=DEFAULT_SEED)

    def test_record_length(self) -> None:
        """测试返回恰好 100 个记录点 / Test exactly 100 record points are returned."""
        self.assertEqual(len(self.record), 100,
                         f"应返回100个点 / Should return 100 points, got {len(self.record)}")

    def test_monotone_decreasing(self) -> None:
        """测试收敛曲线单调不增 / Test convergence curve is monotonically non-increasing."""
        violations = [i for i in range(len(self.record)-1) if self.record[i] < self.record[i+1]]
        self.assertEqual(len(violations), 0,
                         f"收敛曲线在以下位置上升 / Curve rises at: {violations}")

    def test_runtime_reasonable(self) -> None:
        """测试运行时间合理（<120s）/ Test runtime is reasonable (<120s)."""
        self.assertLess(self.elapsed, 120,
                        f"运行时间过长 / Runtime too long: {self.elapsed:.1f}s")

    def test_best_value_finite_positive(self) -> None:
        """测试最终最优值有限且为正 / Test final best value is finite and positive."""
        best = self.record[-1]
        self.assertTrue(math.isfinite(best) and best > 0,
                        f"最优值应为有限正数 / Should be finite positive: {best:.4e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
