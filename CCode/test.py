import ctypes

# find the shared library, which is a file with extension .so
# 1. find the shared library




# 1. open the shared library
mylib = ctypes.CDLL("/Users/matteobalice/Desktop/retopology/CCode/build/lib.macosx-10.9-universal2-3.9/setCurvature.cpython-39-darwin.so")

# 3. call function mysum
mylib.main(ctypes.c_int(1), None)
