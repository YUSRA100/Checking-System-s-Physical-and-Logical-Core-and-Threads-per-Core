
import multiprocessing
import psutil 
import tensorflow as tf

#-------------- cores-----------------
print(multiprocessing.cpu_count())
print(psutil.cpu_count(logical = False))
#actual number of pysical cores
print(psutil.cpu_count(logical = True))

#----------checks on windows prompt, threads per core ----------
#prompt ->wmic cpu list /format:list
#cmd - > msinfo32

#----------- gpu testing--------------

print(tf.test.is_gpu_available())
print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('CPU'))
print(tf.config.list_physical_devices('GPU'))
