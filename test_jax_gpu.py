#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import time

def test_jax_gpu():
    print("🔍 JAX 环境信息:")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    print("-" * 50)

    size = 2000
    print(f"\n创建 {size}x{size} 矩阵并执行矩阵乘法...")

    # 使用 jax.device_put 将数据放到默认设备（GPU 如果可用）
    a = jax.device_put(jnp.ones((size, size), dtype=jnp.float32))
    b = jax.device_put(jnp.ones((size, size), dtype=jnp.float32))


    # 编译并执行计算（JIT 可选，这里直接运行）
    start = time.time()
    c = jnp.dot(a, b)  # 矩阵乘法
    jax.block_until_ready(c)  # 确保异步计算完成
    elapsed = time.time() - start

    # 验证结果（全为 size 的矩阵）
    expected_sum = size * size * size  # 每个元素是 size，总和是 size^3
    actual_sum = jnp.sum(c).item()
    print(f"\n✅ 计算完成！")
    print(f"  - 结果总和: {actual_sum:.1f} (期望: {expected_sum})")
    print(f"  - 误差: {abs(actual_sum - expected_sum):.2e}")
    print(f"  - 耗时: {elapsed:.3f} 秒")

    if abs(actual_sum - expected_sum) < 1e-3:
        print("\n🎉 JAX GPU 运算测试通过！")
        return True
    else:
        print("\n❌ 结果不正确！")
        return False

if __name__ == "__main__":
    test_jax_gpu()