"""
constants.py — 随机数生成模块 / Random Number Generation Module.

此模块直接翻译自 C++ Constants.h，提供算法所需的随机数生成函数。
This module is a direct translation of C++ Constants.h, providing random
number generation functions required by the algorithm.

主要组件 / Main Components:
    - RANDOM(): 均匀分布随机数 / Uniform random number in (0, 1]
    - randn():  正态分布随机数 / Normal distribution random number
    - randc():  柯西分布随机数 / Cauchy distribution random number
    - set_seed(): 设置随机种子 / Set random seed for reproducibility

使用示例 / Usage Example:
    >>> import constants as C
    >>> C.set_seed(42)
    >>> val = C.RANDOM()   # 返回 (0, 1] 的随机数 / Returns value in (0, 1]

参考 / References:
    - Vakhnin & Sopov, CC-SHADE-ML, Algorithms 2022, 15, 451
    - Box-Muller transform for normal distribution sampling
"""

import math
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 数学常量 / Mathematical Constants
# ══════════════════════════════════════════════════════════════════════════════
E  = math.e    # 自然常数 / Euler's number
PI = math.pi   # 圆周率 / Pi

# 全局随机数生成器，通过 set_seed() 重置 / Global RNG, reset via set_seed()
_rng = np.random.default_rng()


def set_seed(seed: int) -> None:
    """
    重置全局随机数生成器种子 / Reset the global random number generator seed.

    用于保证实验可复现性，每次独立运行前调用。
    Used to ensure reproducibility; call before each independent run.

    Args:
        seed (int): 随机种子值 / Random seed value

    Raises:
        TypeError: 当 seed 不是整数时 / When seed is not an integer
    """
    if not isinstance(seed, int):
        raise TypeError(f"种子必须为整数 / Seed must be an integer, got {type(seed)}")
    global _rng
    _rng = np.random.default_rng(seed)
    logger.debug(f"随机种子已设置 / Random seed set: {seed}")


def RANDOM() -> float:
    """
    生成 (0, 1] 均匀分布随机数 / Generate uniform random number in (0, 1].

    对应 C++ RANDOM()，排除 0 以避免 log(0) 等数值错误。
    Corresponds to C++ RANDOM(); excludes 0 to avoid numerical errors like log(0).

    Returns:
        float: (0, 1] 范围内的随机数 / Random number in (0, 1]
    """
    x = -1.0
    while x <= 0.0 or x > 1.0:
        x = _rng.random()
    return x


def randn(M: float, s: float) -> float:
    """
    生成正态分布随机数 N(M, s²) / Generate normal distribution random number N(M, s²).

    使用 Box-Muller 变换，与 C++ randn(M, s) 实现保持一致。
    Uses Box-Muller transform, consistent with C++ randn(M, s) implementation.

    Args:
        M (float): 均值 / Mean of the distribution
        s (float): 标准差 / Standard deviation of the distribution

    Returns:
        float: 正态分布采样值 / Sample from normal distribution N(M, s²)
    """
    r1 = RANDOM()
    r2 = RANDOM()
    return M + s * math.sqrt(-2.0 * math.log(r1)) * math.sin(2.0 * PI * r2)


def randc(a: float, b: float) -> float:
    """
    生成柯西分布随机数 Cauchy(a, b) / Generate Cauchy distribution random number.

    对应 C++ randc(a, b)，用于 SHADE 中缩放因子 F 的采样。
    Corresponds to C++ randc(a, b); used for sampling scale factor F in SHADE.

    Args:
        a (float): 位置参数（中位数）/ Location parameter (median)
        b (float): 尺度参数 / Scale parameter

    Returns:
        float: 柯西分布采样值 / Sample from Cauchy distribution
    """
    r = RANDOM()
    return a + b * math.tan(PI * (r - 0.5))
