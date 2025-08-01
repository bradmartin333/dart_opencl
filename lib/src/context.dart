import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart' as ffilib;
import 'package:opencl/opencl.dart';

class Context {
  Context(this.context, this.dcl);

  cl_context context;
  OpenCL dcl;

  /// increments the ref-count of the context
  void retain() {
    final ret = dcl.clRetainContext(context);
    assert(ret == CL_SUCCESS);
  }

  void release() {
    final ret = dcl.clReleaseContext(context);
    assert(ret == CL_SUCCESS);
  }

  CommandQueue createCommandQueue(Device device, {bool outOfOrder = false}) {
    final errcodeRet = ffilib.calloc<ffi.Int32>();
    final properties =
        ffilib.calloc<ffi.UnsignedLong>(3) as ffi.Pointer<ffi.Uint64>;
    var lastProperty = 0;
    if (outOfOrder) {
      properties[0] = CL_QUEUE_PROPERTIES;
      properties[1] = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
      lastProperty = 2;
    }
    properties[lastProperty] = 0;
    final commandQueue = dcl.clCreateCommandQueueWithProperties(
        context, device.device, properties, errcodeRet);
    assert(errcodeRet.value == CL_SUCCESS);

    ffilib.calloc.free(errcodeRet);
    ffilib.calloc.free(properties);
    return CommandQueue(commandQueue, dcl);
  }

  /// size in bytes.
  Mem createBuffer(int size,
      {NativeBuffer? hostData,
      bool onlyCopy = false,
      bool kernelRead = false,
      bool kernelWrite = false,
      bool hostRead = false,
      bool hostWrite = false}) {
    if (size <= 0) {
      throw ArgumentError('size must be greater than zero');
    }
    var flags = 0;
    if (onlyCopy) {
      if (hostData == null) {
        throw ArgumentError(
            'you must provide non-null hostData while setting onlyCopy');
      }
      flags |= CL_MEM_COPY_HOST_PTR;
    } else if (hostData != null) {
      flags |= CL_MEM_USE_HOST_PTR;
    }

    // TODOvery complex method.
    if (hostRead && hostWrite) {
      throw ArgumentError('hostRead and hostWrite are mutually exclusive');
    }

    if (!hostRead && !hostWrite) {
      flags |= CL_MEM_HOST_NO_ACCESS;
    } else if (hostWrite) {
      flags |= CL_MEM_HOST_WRITE_ONLY;
    } else if (hostRead) {
      flags |= CL_MEM_HOST_READ_ONLY;
    }
    if (kernelRead && kernelWrite) {
      flags |= CL_MEM_READ_WRITE;
    } else {
      if (kernelRead) flags |= CL_MEM_READ_ONLY;
      if (kernelWrite) flags |= CL_MEM_WRITE_ONLY;
    }
    final errcodeRet = ffilib.calloc<ffi.Int32>();

    final memPtr = dcl.clCreateBuffer(
        context, flags, size, hostData?.ptr.cast() ?? ffi.nullptr, errcodeRet);
    assert(errcodeRet.value == CL_SUCCESS);

    ffilib.calloc.free(errcodeRet);

    return Mem(memPtr, dcl);
  }

  Program createProgramWithSource(List<String> strings) {
    final nativeStrings = strings.map((e) => e.toNativeUtf8()).toList();

    final count = nativeStrings.length;
    final stringsPtr = ffilib.calloc<ffi.Pointer<ffilib.Utf8>>(count);
    final lengthsPtr = ffilib.calloc<ffi.Size>(count);
    for (var i = 0; i < count; ++i) {
      stringsPtr[i] = nativeStrings[i];
      lengthsPtr[i] = strings[i].length;
    }
    final errcodeRet = ffilib.calloc<ffi.Int32>();

    final program = dcl.clCreateProgramWithSource(
        context, count, stringsPtr.cast(), lengthsPtr, errcodeRet);

    assert(errcodeRet.value == CL_SUCCESS);

    ffilib.calloc.free(errcodeRet);

    for (final element in nativeStrings) {
      ffilib.calloc.free(element);
    }
    ffilib.calloc.free(stringsPtr);
    ffilib.calloc.free(lengthsPtr);
    return Program(program, dcl);
  }
}
