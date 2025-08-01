import 'dart:ffi' as ffi;
import 'dart:io' as io show Platform;

import 'package:ffi/ffi.dart' as ffilib;
import 'package:opencl/src/context.dart';
import 'package:opencl/src/device.dart';
import 'package:opencl/src/native_cl.dart';
import 'package:opencl/src/platform.dart';

class OpenCL extends NativeCL {
  OpenCL(this.openCLDynLib) : super(openCLDynLib);

  final ffi.DynamicLibrary openCLDynLib;

  static ffi.DynamicLibrary openDynLib() {
    var libraryPath = 'libOpenCL.so';
    if (io.Platform.isMacOS) {
      libraryPath = 'libOpenCL.dylib';
    } else if (io.Platform.isWindows) {
      libraryPath = 'OpenCL.dll';
    }
    return ffi.DynamicLibrary.open(libraryPath);
  }

  List<Platform> getPlatforms() {
    final numPlatforms = ffilib.calloc<ffi.Uint32>();
    var ret = clGetPlatformIDs(0, ffi.nullptr, numPlatforms);
    assert(ret == CL_SUCCESS);
    final platformsList = ffilib.calloc<cl_platform_id>(numPlatforms.value);
    ret = clGetPlatformIDs(numPlatforms.value, platformsList, numPlatforms);
    assert(ret == CL_SUCCESS);
    final platforms = List.generate(numPlatforms.value,
        (index) => createPlatform(platformsList[index], this));
    ffilib.calloc.free(numPlatforms);
    ffilib.calloc.free(platformsList);
    return platforms;
  }

  Context createContext(List<Device> devices) {
    final errcodeRet = ffilib.calloc<ffi.Int32>();
    final devicesHandles = ffilib.calloc<cl_device_id>(devices.length);
    for (var i = 0; i < devices.length; ++i) {
      devicesHandles[i] = devices[i].device;
    }

    final context = clCreateContext(ffi.nullptr, devices.length, devicesHandles,
        ffi.nullptr, ffi.nullptr, errcodeRet);
    assert(errcodeRet.value == CL_SUCCESS);
    ffilib.calloc.free(errcodeRet);
    ffilib.calloc.free(devicesHandles);
    return Context(context, this);
  }
}
