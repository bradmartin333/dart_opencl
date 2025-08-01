import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart' as ffilib;
import 'package:opencl/opencl.dart';

class Kernel {
  Kernel(this.kernel, this.dcl);

  cl_kernel kernel;
  OpenCL dcl;

  void retain() {
    final ret = dcl.clRetainKernel(kernel);
    assert(ret == CL_SUCCESS);
  }

  void release() {
    final ret = dcl.clReleaseKernel(kernel);
    assert(ret == CL_SUCCESS);
  }

  int setKernelArgMem(int argIndex, Mem argValue) {
    final sizeofClmem =
        ffi.sizeOf<ffi.Pointer<ffi.Void>>(); //sizeof (cl_mem) is
    // typically the size of a pointer
    final memPtr = ffilib.calloc<ffi.Pointer<ffi.Void>>()
      ..value = argValue.mem.cast();
    final ret =
        dcl.clSetKernelArg(kernel, argIndex, sizeofClmem, memPtr.cast());

    assert(ret == CL_SUCCESS);
    ffilib.calloc.free(memPtr);
    return ret;
  }

  int setKernelArgLocal(int argIndex, int sizeInBytes) {
    return dcl.clSetKernelArg(kernel, argIndex, sizeInBytes, ffi.nullptr);
  }

  int setKernelArgInt(int argIndex, int argValue) {
    final dataPtr = ffilib.calloc<ffi.Int32>()..value = argValue;
    final ret = dcl.clSetKernelArg(
        kernel, argIndex, ffi.sizeOf<ffi.Int32>(), dataPtr.cast());

    assert(ret == CL_SUCCESS);
    ffilib.calloc.free(dataPtr);
    return ret;
  }

  int setKernelArgLong(int argIndex, int argValue) {
    final dataPtr = ffilib.calloc<ffi.Int64>()..value = argValue;
    final ret = dcl.clSetKernelArg(
        kernel, argIndex, ffi.sizeOf<ffi.Int64>(), dataPtr.cast());

    assert(ret == CL_SUCCESS);
    ffilib.calloc.free(dataPtr);
    return ret;
  }

  int setKernelArgFloat(int argIndex, double argValue) {
    final dataPtr = ffilib.calloc<ffi.Float>()..value = argValue;
    final ret = dcl.clSetKernelArg(
        kernel, argIndex, ffi.sizeOf<ffi.Float>(), dataPtr.cast());

    assert(ret == CL_SUCCESS);
    ffilib.calloc.free(dataPtr);
    return ret;
  }

  int setKernelArgDouble(int argIndex, double argValue) {
    final dataPtr = ffilib.calloc<ffi.Double>()..value = argValue;
    final ret = dcl.clSetKernelArg(
        kernel, argIndex, ffi.sizeOf<ffi.Double>(), dataPtr.cast());

    assert(ret == CL_SUCCESS);
    ffilib.calloc.free(dataPtr);
    return ret;
  }
}
