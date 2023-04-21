import math
import sys

import numba as nb
import numpy as np
from numba import cuda
import warnings
warnings.filterwarnings("ignore")

THREAD_BLOCK = 32


def scanGPU(array, independant_recursif):
    n = array.shape[0]
    blocs_per_grid = int(math.ceil(n / THREAD_BLOCK))

    # On copie le tableau sur la mémoire GPU
    array_device = cuda.to_device(array)

    # On alloue l'espace nécessaire pour le tableau intermédiaire sur la mémoire GPU
    array_intermediaire = cuda.device_array(blocs_per_grid, dtype=np.int32)

    # On lance le kernel
    scanKernel[blocs_per_grid, THREAD_BLOCK](array_device, array_intermediaire)
    cuda.synchronize()

    # On récupère le tableau depuis la mémoire GPU
    array = array_device.copy_to_host()

    # Si on n'a pas donné de paramètre indépendant, on lance la fonction récursive sur le tableau intermédiaire
    if not (independant_recursif or blocs_per_grid == 1):
        array_intermediaire = scanGPU(array_intermediaire, independant_recursif)

        # On alloue l'espace nécessaire pour le tableau intermédiaire sur la mémoire GPU
        array_intermediaire_device = cuda.to_device(array_intermediaire)

        # On ajoute les tableaux dans un dernier kernel
        fullArray[blocs_per_grid, THREAD_BLOCK](array_device, array_intermediaire_device)
        cuda.synchronize()

        # On récupère le tableau depuis la mémoire GPU
        array = array_device.copy_to_host()

    return array



@cuda.jit
def scanKernel(array_a, array_b):
    n = array_a.shape[0]
    m = int(math.ceil(math.log2(n)))
    if n > THREAD_BLOCK:
        n = THREAD_BLOCK  # On ne peut pas utiliser plus de THREAD_BLOCK threads
    elif n != 2 ** m:  # Si n n'est pas une puissance de 2 on le transforme en une puissance de 2
        n = 2 ** m
    cuda.syncthreads()
    tid = cuda.threadIdx.x
    global_id = cuda.grid(1)
    cuda.syncthreads()

    shared_filter = cuda.shared.array(THREAD_BLOCK, dtype=nb.int32)
    shared_filter[tid] = array_a[global_id]
    cuda.syncthreads()

    # Up-sweep phase
    for d in range(m):
        k = n // 2 ** (d + 1)
        if tid < k:
            l = tid * 2 ** (d + 1)
            shared_filter[l + 2 ** (d + 1) - 1] = shared_filter[l + 2 ** (d + 1) - 1] + shared_filter[l + 2 ** d - 1]
        cuda.syncthreads()

    if tid == 0:
        array_b[cuda.blockIdx.x] = shared_filter[n - 1]
        shared_filter[n - 1] = 0
    cuda.syncthreads()

    # Down-sweep phase
    for d in range(m - 1, -1, -1):
        k = n // 2 ** (d + 1)
        if tid < k:
            l = tid * 2 ** (d + 1)
            t = shared_filter[l + 2 ** d - 1]
            shared_filter[l + 2 ** d - 1] = shared_filter[l + 2 ** (d + 1) - 1]
            shared_filter[l + 2 ** (d + 1) - 1] = shared_filter[l + 2 ** (d + 1) - 1] + t
        cuda.syncthreads()

    array_a[global_id] = shared_filter[tid]
    cuda.syncthreads()


@cuda.jit
def fullArray(array_a, array_b):
    global_id = cuda.grid(1)
    bid = cuda.blockIdx.x
    array_a[global_id] = array_a[global_id] + array_b[bid]


def showArray(array):
    output = ""
    for i in range(array.size):
        output += str(array[i])
        if i != array.size - 1:
            output += ","
    print(output)


def runScan():
    # Va nous servir à controler les paramètres
    global THREAD_BLOCK
    independant = False
    inclusive = False

    if len(sys.argv) < 2:  # On vérifie que l'utilisateur rentre bien un fichier en paramètre
        print("Usage: python project-gpu.py <inputFile> [--tb int] [--independent] [--inclusive]")
        sys.exit(1)

    # On lit le fichier et on récupère l'array à scanner
    file = sys.argv[1]
    array_to_scan = open(file, "r").read().split(',')
    if array_to_scan[-1] == '':  # On regarde si le fichier est vide
        print("le fichier est vide")
        sys.exit(1)
    array_to_scan = np.array(array_to_scan, dtype=np.int32)

    # On récupère les paramètres
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == "--tb":
            if i + 1 >= len(sys.argv):
                print("Usage: python project-gpu.py <inputFile> [--tb int] [--independent] [--inclusive]")
                sys.exit(1)
            if not sys.argv[i + 1].isdigit():
                print("Usage: python project-gpu.py <inputFile> [--tb int] [--independent] [--inclusive]")
                sys.exit(1)
            thread_block_temp = int(sys.argv[i + 1])
            if math.log2(thread_block_temp) != int(math.log2(thread_block_temp)):
                THREAD_BLOCK = 2 ** int(math.ceil(math.log2(thread_block_temp)))
            else:
                THREAD_BLOCK = thread_block_temp
            i += 1
        elif sys.argv[i] == "--independent":
            independant = True
        elif sys.argv[i] == "--inclusive":
            inclusive = True

    # On copie l'array
    array_to_gpu = array_to_scan.copy()

    # On lance le scan gpu
    array_to_gpu = scanGPU(array_to_gpu, independant)

    # Si l'utilisateur a demandé un scan inclusif
    if inclusive:
        array_to_gpu = array_to_gpu + array_to_scan

    # On affiche le scan
    showArray(array_to_gpu)


if __name__ == '__main__':
    runScan()  # On doit passer par une fonction sinon on ne peut pas exploiter la variable globale THREAD_BLOCK
