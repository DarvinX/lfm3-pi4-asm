"""Scalar IEEE-FMA oracle for the pinned Arm operations, validation only."""
import ctypes
import struct
import numpy as np

F = np.float32
libm = ctypes.CDLL('libm.so.6')
libm.fmaf.argtypes = [ctypes.c_float]*3
libm.fmaf.restype = ctypes.c_float


def fma(a, b, c):
    return F(libm.fmaf(float(a), float(b), float(c)))


def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def floating(x):
    return F(struct.unpack('<f', struct.pack('<I', x & 0xffffffff))[0])


def native_round(scale, x):
    return min(127, (bits(fma(scale, x, 12582912)) & 0x7fffff)-0x400000)


def native_swiglu(x, up):
    constants = [F(float.fromhex(s)) for s in ['0x1.715476p+0',
        '0x1.62e4p-1', '0x1.7f7d1cp-20', '0x1.ffffecp-1',
        '0x1.fffdb6p-2', '0x1.555e66p-3', '0x1.573e2ep-5', '0x1.0e4020p-7']]
    log2, lnhi, lnlo, c0, c1, c2, c3, c4 = constants
    result = []
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        for gate, weight in zip(x, up):
            neg = F(F(0)-gate)
            z = fma(neg, log2, 12582912)
            n = F(z-F(12582912))
            b = fma(-n, lnlo, fma(-n, lnhi, neg))
            e = (bits(z) << 23) & 0xffffffff
            k = floating(e+bits(F(1)))
            u = F(b*b)
            j = fma(fma(fma(c4, b, c3), u, fma(c2, b, c1)), u, F(c0*b))
            if abs(n) > 126:
                d = 0x82000000 if n <= 0 else 0
                s1, s2 = floating(d+0x7f000000), floating(e-d)
                exponential = F(s1*s1) if abs(n) > 192 else F(fma(s2, j, s2)*s1)
            else:
                exponential = fma(j, k, k)
            result.append(F(F(gate/F(F(1)+exponential))*weight))
    return np.asarray(result, dtype='f4')
