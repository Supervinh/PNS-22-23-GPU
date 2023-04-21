# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import math

import numpy as np
from numba import cuda


@cuda.jit
def scanKernel(arrayGPU, n, m):
    monteKernel(arrayGPU, n, m)
    if cuda.grid(1) == 0:
        arrayGPU[n - 1] = 0
    descenteKernel(arrayGPU, n, m)


@cuda.jit
def monteKernel(arrayGPU, n, m):
    x = cuda.grid(1)
    for d in range(0, m):
        k = n // 2 ** (d + 1)
        if x < k:
            arrayGPU[x * 2 ** (d + 1) + 2 ** (d + 1) - 1] += arrayGPU[x * 2 ** (d + 1) + 2 ** d - 1]
        cuda.syncthreads()


@cuda.jit
def descenteKernel(arrayGPU, n, m):
    x = cuda.grid(1)
    for d in range(m - 1, -1, -1):
        k = n // 2 ** (d + 1)
        if x < k:
            t = arrayGPU[x * 2 ** (d + 1) + 2 ** d - 1]
            arrayGPU[x * 2 ** (d + 1) + 2 ** d - 1] = arrayGPU[x * 2 ** (d + 1) + 2 ** (d + 1) - 1]
            arrayGPU[x * 2 ** (d + 1) + 2 ** (d + 1) - 1] += t
        cuda.syncthreads()


def scanGPU(array, tb, bg):
    tb = tb
    bg = bg
    n = array.shape[0]
    m = math.ceil(math.log2(n))
    arrayGPU = cuda.to_device(array)
    scanKernel[bg, tb](arrayGPU, n, m)
    array = arrayGPU.copy_to_host()
    return array


def monte(array, n, m):
    for d in range(0, m):
        for k in range(0, n - 1, 2 ** (d + 1)):
            array[k + 2 ** (d + 1) - 1] += array[k + 2 ** d - 1]


def descente(array, n, m):
    array[n - 1] = 0
    for d in range(m - 1, -1, -1):
        for k in range(0, n, 2 ** (d + 1)):
            t = array[k + 2 ** d - 1]
            array[k + 2 ** d - 1] = array[k + 2 ** (d + 1) - 1]
            array[k + 2 ** (d + 1) - 1] += t


def scanCPU(array):
    n = array.shape[0]
    m = math.ceil(math.log2(n))
    monte(array, n, m)
    descente(array, n, m)


def run():
    size = 256
    arrayGPU = np.random.randint(1, 10, size)
    arrayCPU = arrayGPU.copy()
    print(arrayGPU)
    arrayGPU = scanGPU(arrayGPU, size, 1)
    scanCPU(arrayCPU)
    print(arrayGPU)
    print(arrayCPU)
    print(arrayGPU == arrayCPU)


if __name__ == '__main__':
    run()


def scanCPU(array):
    array_temp = array.copy()  # On copie l'array pour ne pas modifier l'original mais pas obligatoire
    n = array.size
    m = int(np.log2(n))  # Parce que n = 2^m
    print('array original', array_temp)
    # phase up-sweep
    for d in range(m):
        for k in range(0, n - 1, 2 ** (d + 1)):
            array_temp[k + 2 ** (d + 1) - 1] = array_temp[k + 2 ** (d + 1) - 1] + array_temp[k + 2 ** d - 1]
        print('dans la boucle', array_temp)
    print('up_sweep', array_temp)

    # phase down-sweep
    array_temp[n - 1] = 0
    for d in range(m - 1, -1, -1):
        for k in range(0, n - 1, 2 ** (d + 1)):
            t = array_temp[k + 2 ** d - 1]
            array_temp[k + 2 ** d - 1] = array_temp[k + 2 ** (d + 1) - 1]
            array_temp[k + 2 ** (d + 1) - 1] = array_temp[k + 2 ** (d + 1) - 1] + t
        print('dans la boucle', array_temp)
    print('down_sweep', array_temp)

    print('array final', array_temp)
    return array


def scanGPU(array, blocks_per_grid, threads_per_block):
    len_array = len(array)
    log2_len_array = int(np.log2(len_array))

    array = cuda.to_device(array)

    scanKernel[threads_per_block, blocks_per_grid](array, len_array, log2_len_array)

    return array.copy_to_host()


@cuda.jit
def scanKernel(array, n, m):
    # Up-sweep phase
    thread_id = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # cuda.grid(1)
    x = thread_id

    for d in range(0, m):
        k = n // 2 ** (d + 1)
        if x < k:
            array[x * 2 ** (d + 1) + 2 ** (d + 1) - 1] += array[x * 2 ** (d + 1) + 2 ** d - 1]
        cuda.syncthreads()

    if thread_id == 0:
        array[n - 1] = 0

    # Down-sweep phase
    for d in range(m - 1, -1, -1):
        k = n // 2 ** (d + 1)
        if x < k:
            t = array[x * 2 ** (d + 1) + 2 ** d - 1]
            array[x * 2 ** (d + 1) + 2 ** d - 1] = array[x * 2 ** (d + 1) + 2 ** (d + 1) - 1]
            array[x * 2 ** (d + 1) + 2 ** (d + 1) - 1] += t
        cuda.syncthreads()