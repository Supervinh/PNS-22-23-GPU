import math

import numpy as np
from numba import cuda
import numba as nb


THREAD_BLOCK=128


""" Version CPU
On prend en paramètre un numpy array de valeurs int32
On suppose une taille n=2^m (plus simple)
On effectue up-sweep et down-sweep
On vérifie chaque étape en affichant l'array
"""


def scanCPU(array):
    array_temp = array.copy()  # On copie l'array pour ne pas modifier l'original mais pas obligatoire
    n = array.size
    m = int(np.log2(n))  # Parce que n = 2^m
    #print('array original', array_temp)
    # phase up-sweep
    for d in range(m):
        for k in range(0, n - 1, 2 ** (d + 1)):
            array_temp[k + 2 ** (d + 1) - 1] = array_temp[k + 2 ** (d + 1) - 1] + array_temp[k + 2 ** d - 1]
        #print('dans la boucle', array_temp)
    #print('up_sweep', array_temp)

    # phase down-sweep
    array_temp[n - 1] = 0
    for d in range(m - 1, -1, -1):
        for k in range(0, n - 1, 2 ** (d + 1)):
            t = array_temp[k + 2 ** d - 1]
            array_temp[k + 2 ** d - 1] = array_temp[k + 2 ** (d + 1) - 1]
            array_temp[k + 2 ** (d + 1) - 1] = array_temp[k + 2 ** (d + 1) - 1] + t
        #print('dans la boucle', array_temp)
    #print('down_sweep', array_temp)

    #print('array final', array_temp)
    return array


""" Version GPU single thread
On prend en paramètre un numpy array qui peut être calculé sur un seul thread block, donc taille au max=1024 éléments
"""


# def scanGPU(array):
#     n = array.size
#     array_device = cuda.to_device(array)
#     threads_per_block = 4
#     blocs_per_grid = 1
#     scanKernel[blocs_per_grid, threads_per_block](array_device, n)
#     cuda.synchronize()
#     array = array_device.copy_to_host()
#     return array
#
#
# @cuda.jit
# def scanKernel(array, array_size):
#     tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x # On récupère le thread id
#     m = int(math.log2(array_size))
#
#     # Up-sweep phase
#     for d in range(m):
#         k = array_size // 2 ** (d + 1)
#         if tid < k :
#             l = tid * 2 ** (d + 1) # On simule la boucle for du CPU
#             array[l + 2 ** (d + 1) - 1] = array[l + 2 ** (d + 1) - 1] + array[l + 2 ** d - 1]
#         cuda.syncthreads() # on synchronise les threads à chaque itération pour éviter les erreurs
#
#     if tid == 0:
#         array[array_size-1] = 0
#
#     # Down-sweep phase
#     for d in range(m - 1, -1, -1):
#         k = array_size // 2 ** (d + 1)
#         if tid < k:
#             l = tid * 2 ** (d + 1) # On simule la boucle for du CPU
#             t = array[l + 2 ** d - 1]
#             array[l + 2 ** d - 1] = array[l + 2 ** (d + 1) - 1]
#             array[l + 2 ** (d + 1) - 1] = array[l + 2 ** (d + 1) - 1] + t
#         cuda.syncthreads() # on synchronise les threads à chaque itération pour éviter les erreurs


""" Version GPU shared memory pour un seul bloc"""
def scanGPUShared(array):
    n = array.size
    m = math.ceil(math.log2(n))
    threads_per_block = THREAD_BLOCK
    blocs_per_grid = 1
    array_device_A = cuda.to_device(array)
    scanKernelShared[blocs_per_grid, threads_per_block](array_device_A, n, m)
    array = array_device_A.copy_to_host()
    return array


@cuda.jit
def scanKernelShared(array_A, n, m):
    if n != 2 ** m:
        n = 2 ** m
    cuda.syncthreads()
    tid = cuda.threadIdx.x
    global_id = cuda.grid(1)
    cuda.syncthreads()

    shared_filter=cuda.shared.array(THREAD_BLOCK, dtype=nb.int32)
    shared_filter[tid] = array_A[tid]
    # On utilise la mémoire partagée
    cuda.syncthreads()

    # Up-sweep phase
    for d in range(m):
        k = n // 2 ** (d + 1)
        if global_id < k:
            l = global_id * 2 ** (d + 1)
            shared_filter[l + 2 ** (d + 1) - 1] = shared_filter[l + 2 ** (d + 1) - 1] + shared_filter[l + 2 ** d - 1]
        cuda.syncthreads()

    if global_id == 0:
        shared_filter[n - 1] = 0
    cuda.syncthreads()

    # Down-sweep phase
    for d in range(m - 1, -1, -1):
        k = n // 2 ** (d + 1)
        if global_id < k:
            l = global_id * 2 ** (d + 1)
            t = shared_filter[l + 2 ** d - 1]
            shared_filter[l + 2 ** d - 1] = shared_filter[l + 2 ** (d + 1) - 1]
            shared_filter[l + 2 ** (d + 1) - 1] = shared_filter[l + 2 ** (d + 1) - 1] + t
        cuda.syncthreads()


    array_A[tid] = shared_filter[tid]
    cuda.syncthreads()



if __name__ == '__main__':
    #array = np.array([2, 3, 4, 6], dtype=np.int32)
    array = np.random.randint(1, 10, size=128, dtype=np.int32)
    #print("CPU", scanCPU(array))
    print("array", array, "\n\n\n")
    result = scanGPUShared(array)
    print("GPU", result, "\n\n\n")
