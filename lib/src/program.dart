import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart' as ffilib;
import 'package:opencl/opencl.dart';

class Program {
  Program(this.program, this.dcl);

  cl_program program;
  OpenCL dcl;

  void retain() {
    final ret = dcl.clRetainProgram(program);
    assert(ret == CL_SUCCESS);
  }

  void release() {
    final ret = dcl.clReleaseProgram(program);
    assert(ret == CL_SUCCESS);
  }

  /// Builds the program. (wraps clBuildProgram)
  ///
  /// Please read the manual of clBuildProgram.
  /// The build logs will be appended to the supplied buildLogs list.
  int buildProgram(
      List<Device> devices, String options, List<String> buildLogs) {
    final devicesHandles = ffilib.calloc<cl_device_id>(devices.length);
    for (var i = 0; i < devices.length; ++i) {
      devicesHandles[i] = devices[i].device;
    }
    final nativeOptions = options.toNativeUtf8();
    final errcode = dcl.clBuildProgram(program, devices.length, devicesHandles,
        nativeOptions.cast(), ffi.nullptr, ffi.nullptr);
    if (errcode == CL_BUILD_PROGRAM_FAILURE) {
      const logSize = 4096;
      final buildLog = ffilib.calloc<ffi.Char>(logSize).cast();
      final retSize = ffilib.calloc<ffi.Size>();

      for (final device in devices) {
        dcl.clGetProgramBuildInfo(program, device.device, CL_PROGRAM_BUILD_LOG,
            logSize, buildLog.cast(), retSize);

        buildLogs.add(buildLog.toString());
      }
      ffilib.calloc.free(retSize);
      ffilib.calloc.free(buildLog);
    }

    ffilib.calloc.free(devicesHandles);
    ffilib.calloc.free(nativeOptions);
    return errcode;
  }

  Kernel createKernel(String kernelName) {
    final errcodeRet = ffilib.calloc<ffi.Int32>();
    final nativeName = kernelName.toNativeUtf8();

    final kernel = dcl.clCreateKernel(program, nativeName.cast(), errcodeRet);
    assert(errcodeRet.value == CL_SUCCESS);
    ffilib.calloc.free(errcodeRet);
    ffilib.calloc.free(nativeName);
    return Kernel(kernel, dcl);
  }
}
