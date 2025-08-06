import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart' as ffilib;
import 'package:opencl/opencl.dart';
import 'package:opencl/src/device.dart';

class Platform {
  Platform(this.platform, this.dcl, this.devices, this.profile, this.version,
      this.name, this.vendor, this.extensions, this.hostTimerResolution);

  cl_platform_id platform;
  OpenCL dcl;
  List<Device> devices;
  String profile;
  String version;
  String name;
  String vendor;
  int hostTimerResolution;
  List<String> extensions;
}

String getPlatformStringInfo(
    cl_platform_id platform, OpenCL dcl, int paramName) {
  // query the size of the string
  final outSize = ffilib.calloc<ffi.Size>();
  var ret = dcl.clGetPlatformInfo(platform, paramName, 0, ffi.nullptr, outSize);
  if (ret != CL_SUCCESS) {
    throw Exception(
        'Failed to get platform info size for $paramName with error $ret');
  }

  // allocate a buffer and get the string data
  final strBuf =
      ffilib.calloc<ffi.Char>(outSize.value + 1); // +1 for null terminator
  ret = dcl.clGetPlatformInfo(
      platform, paramName, outSize.value, strBuf.cast(), ffi.nullptr);
  if (ret != CL_SUCCESS) {
    ffilib.calloc.free(strBuf);
    throw Exception(
        'Failed to get platform info for $paramName with error $ret');
  }

  // convert the C string to a Dart string
  final result = strBuf.cast<ffilib.Utf8>().toDartString();

  // free the allocated memory
  ffilib.calloc.free(strBuf);
  ffilib.calloc.free(outSize);

  return result;
}

Platform createPlatform(cl_platform_id platform, OpenCL dcl) {
  final numDevices = ffilib.calloc<ffi.Uint32>();
  dcl.clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, ffi.nullptr, numDevices);

  final devicesList = ffilib.calloc<cl_device_id>(numDevices.value);
  dcl.clGetDeviceIDs(
      platform, CL_DEVICE_TYPE_ALL, numDevices.value, devicesList, numDevices);

  final devices = List.generate(
      numDevices.value, (index) => createDevice(devicesList[index], dcl));

  ffilib.calloc.free(numDevices);
  ffilib.calloc.free(devicesList);

  final profile = getPlatformStringInfo(platform, dcl, CL_PLATFORM_PROFILE);
  final version = getPlatformStringInfo(platform, dcl, CL_PLATFORM_VERSION);
  final name = getPlatformStringInfo(platform, dcl, CL_PLATFORM_NAME);
  final vendor = getPlatformStringInfo(platform, dcl, CL_PLATFORM_VENDOR);
  final extensionsString =
      getPlatformStringInfo(platform, dcl, CL_PLATFORM_EXTENSIONS);
  final extensions = extensionsString.split(' ');

  final uLongBuf = ffilib.calloc<ffi.UnsignedLong>();
  final outSize = ffilib.calloc<ffi.Size>();

  dcl.clGetPlatformInfo(platform, CL_PLATFORM_HOST_TIMER_RESOLUTION,
      ffi.sizeOf<ffi.UnsignedLong>(), uLongBuf.cast(), outSize);
  final hostTimerResolution = uLongBuf.value;

  ffilib.calloc.free(outSize);
  ffilib.calloc.free(uLongBuf);

  return Platform(platform, dcl, devices, profile, version, name, vendor,
      extensions, hostTimerResolution);
}
