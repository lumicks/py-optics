import pytest
import numpy as np
import lumicks.pyoptics.mathutils.czt as czt


@pytest.mark.parametrize(
    "x_shape", ((3, 4), (4, 3), (10, 12), (356, 5), (56, 678), (1, 8), (11, 1))
)
@pytest.mark.parametrize("multiplier", (1, 2))
def test_czt_3d_rect(x_shape, multiplier):
    """Test equivalence of czt and fft for A = 1.0 and w = exp(-2j pi / N)"""
    x = np.random.rand(*x_shape)
    M = x_shape[0] * multiplier
    a = 1.0
    w = np.exp(-2j * np.pi / M)
    x = np.tile(x, (1, 1, 10))
    init = czt.init_czt(x, M, w, a)
    czt_init_exec = czt.exec_czt(x, init)
    czt_direct = czt.czt(x, M, w, a)
    fft_x = np.fft.fft(x, n=M, axis=0)
    np.testing.assert_allclose(czt_init_exec, fft_x, atol=0)
    np.testing.assert_allclose(czt_direct, fft_x, atol=0)


@pytest.mark.parametrize("x_size", (1, 5, 10, 16, 128))
@pytest.mark.parametrize("multiplier", (1, 2))
def test_czt_3d_eye(x_size, multiplier):
    """Test equivalence of czt and fft for A = 1.0 and w = exp(-2j pi / N)"""
    x = np.eye(x_size)
    M = x_size * multiplier
    a = 1.0
    w = np.exp(-2j * np.pi / M)
    x = np.tile(x, (1, 1, 10))
    init = czt.init_czt(x, M, w, a)
    czt_init_exec = czt.exec_czt(x, init)
    czt_direct = czt.czt(x, M, w, a)
    fft_x = np.fft.fft(x, n=M, axis=0)
    np.testing.assert_allclose(czt_init_exec, fft_x, atol=0)
    np.testing.assert_allclose(czt_direct, fft_x, atol=0)


@pytest.mark.parametrize("x_size", (1, 5, 10, 16, 128))
@pytest.mark.parametrize("multiplier", (1, 2))
def test_czt_2d_eye(x_size, multiplier):
    """Test equivalence of czt and fft for A = 1.0 and w = exp(-2j pi / N)"""
    x = np.eye(x_size)
    M = x.shape[0] * multiplier
    a = 1.0
    w = np.exp(-2j * np.pi / M)
    init = czt.init_czt(x, M, w, a)
    czt_init_exec = czt.exec_czt(x, init)
    czt_direct = czt.czt(x, M, w, a)
    fft_x = np.fft.fft(x, n=M, axis=0)
    np.testing.assert_allclose(czt_init_exec, fft_x, atol=0)
    np.testing.assert_allclose(czt_direct, fft_x, atol=0)


@pytest.mark.parametrize(
    "x_shape", ((3, 4), (4, 3), (10, 12), (356, 5), (56, 678), (1, 8), (11, 1))
)
@pytest.mark.parametrize("multiplier", (1, 2))
def test_czt_2d_rect(x_shape, multiplier):
    """Test equivalence of czt and fft for A = 1.0 and w = exp(-2j pi / N)"""
    x = np.random.rand(*x_shape)
    M = x_shape[0] * multiplier
    a = 1.0
    w = np.exp(-2j * np.pi / M)
    init = czt.init_czt(x, M, w, a)
    czt_init_exec = czt.exec_czt(x, init)
    czt_direct = czt.czt(x, M, w, a)
    fft_x = np.fft.fft(x, n=M, axis=0)
    np.testing.assert_allclose(czt_init_exec, fft_x, atol=0)
    np.testing.assert_allclose(czt_direct, fft_x, atol=0)


@pytest.mark.parametrize("x_len", (2, 3, 5, 13, 20, 32, 200, 256))
@pytest.mark.parametrize("multiplier", (1, 2))
def test_czt_array(x_len, multiplier):
    x = np.zeros(x_len)
    M = x_len * multiplier
    idx = np.random.randint(low=0, high=x_len - 1)
    x[idx] = 1.0

    a = 1.0
    w = np.exp(-2j * np.pi / M)
    init = czt.init_czt(x, M, w, a)
    czt_init_exec = czt.exec_czt(x, init)
    czt_direct = czt.czt(x, M, w, a)
    fft_x = np.fft.fft(x, n=M, axis=0)
    np.testing.assert_allclose(czt_init_exec, fft_x, atol=0)
    np.testing.assert_allclose(czt_direct, fft_x, atol=0)
