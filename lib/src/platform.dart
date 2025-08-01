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

  /// in nano seconds
  int hostTimerResolution;
  List<String> extensions;
}

Platform createPlatform(cl_platform_id platform, OpenCL dcl) {
  final numDevices = ffilib.calloc<ffi.Uint32>();
  dcl.clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, ffi.nullptr, numDevices);

  final devicesList = ffilib.calloc<cl_device_id>(numDevices.value);
  dcl.clGetDeviceIDs(
      platform, CL_DEVICE_TYPE_ALL, numDevices.value, devicesList, numDevices);
  final devices = List.generate(
      numDevices.value, (index) => createDevice(devicesList[index], dcl));
  //free used memory
  ffilib.calloc.free(numDevices);
  ffilib.calloc.free(devicesList);
  //I hope that's enough...
  const bufsize = 4096;
  final strbuf = ffilib.calloc<ffi.Int8>(bufsize).cast();
  final outSize = ffilib.calloc<ffi.Size>();
  final uLongBuf = ffilib.calloc<ffi.UnsignedLong>();

  dcl.clGetPlatformInfo(
      platform, CL_PLATFORM_PROFILE, bufsize, strbuf.cast(), outSize);
  final profile = strbuf.toString();

  dcl.clGetPlatformInfo(
      platform, CL_PLATFORM_VERSION, bufsize, strbuf.cast(), outSize);
  final version = strbuf.toString();

  //get platform name
  dcl.clGetPlatformInfo(
      platform, CL_PLATFORM_NAME, bufsize, strbuf.cast(), outSize);
  final name = strbuf.toString();

  //get platform vendor
  dcl.clGetPlatformInfo(
      platform, CL_PLATFORM_VENDOR, bufsize, strbuf.cast(), outSize);
  final vendor = strbuf.toString();

  //get platform extensions
  dcl.clGetPlatformInfo(
      platform, CL_PLATFORM_EXTENSIONS, bufsize, strbuf.cast(), outSize);
  final extensions = strbuf.toString().split(RegExp(' [a-zA-Z]'));

  dcl.clGetPlatformInfo(platform, CL_PLATFORM_HOST_TIMER_RESOLUTION,
      ffi.sizeOf<ffi.UnsignedLong>(), uLongBuf.cast(), outSize);
  final hostTimerResolution = uLongBuf.value;

  //free used memory
  ffilib.calloc.free(strbuf);
  ffilib.calloc.free(outSize);
  ffilib.calloc.free(uLongBuf);

  return Platform(platform, dcl, devices, profile, version, name, vendor,
      extensions, hostTimerResolution);
}
