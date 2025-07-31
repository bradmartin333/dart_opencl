import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart' as ffilib;
import 'package:opencl/opencl.dart';
import 'package:opencl/src/event.dart';

typedef ClEnqueueReadBufferType = int Function(
    cl_command_queue commandQueue,
    cl_mem buffer,
    int blockingWrite,
    int offset,
    int size,
    ffi.Pointer<ffi.Void> ptr,
    int numEventsInWaitList,
    ffi.Pointer<cl_event> eventWaitList,
    ffi.Pointer<cl_event> event);

class CommandQueue {
  CommandQueue(this.commandQueue, this.dcl);
  cl_command_queue commandQueue;
  OpenCL dcl;
  void retain() {
    final ret = dcl.clRetainCommandQueue(commandQueue);
    assert(ret == CL_SUCCESS);
  }

  void release() {
    final ret = dcl.clReleaseCommandQueue(commandQueue);
    assert(ret == CL_SUCCESS);
  }

  void flush() {
    final ret = dcl.clFlush(commandQueue);
    assert(ret == CL_SUCCESS);
  }

  void finish() {
    final ret = dcl.clFinish(commandQueue);
    assert(ret == CL_SUCCESS);
  }

  Event? enqueueNDRangeKernel(Kernel kernel, int workDimensions,
      {required List<int> globalWorkSize,
      bool createEvent = false,
      List<int> globalWorkOffset = const [],
      List<int> localWorkSize = const [],
      List<Event> eventWaitList = const []}) {
    if (globalWorkSize.length != workDimensions) {
      throw ArgumentError(
          'The length of globalWorkSize does not match workDimensions');
    }

    ffi.Pointer<ffi.Size> globalWorkSizePtr;
    globalWorkSizePtr = ffilib.calloc<ffi.Size>(workDimensions);
    for (var i = 0; i < workDimensions; ++i) {
      globalWorkSizePtr[i] = globalWorkSize[i];
    }
    ffi.Pointer<ffi.Size> globalWorkOffsetPtr = ffi.nullptr;
    if (globalWorkOffset.isNotEmpty) {
      if (globalWorkOffset.length != workDimensions) {
        throw ArgumentError(
            'The length of globalWorkOffset does not match workDimensions');
      }
      globalWorkOffsetPtr = ffilib.calloc<ffi.Size>(workDimensions);
      for (var i = 0; i < workDimensions; ++i) {
        globalWorkOffsetPtr[i] = globalWorkOffset[i];
      }
    }
    ffi.Pointer<ffi.Size> localWorkSizePtr = ffi.nullptr;
    if (localWorkSize.isNotEmpty) {
      if (localWorkSize.length != workDimensions) {
        throw ArgumentError(
            'The length of globalWorkOffset does not match workDimensions');
      }
      localWorkSizePtr = ffilib.calloc<ffi.Size>(workDimensions);
      for (var i = 0; i < workDimensions; ++i) {
        localWorkSizePtr[i] = localWorkSize[i];
      }
    }

    ffi.Pointer<cl_event> eventWaitListPtr = ffi.nullptr;
    if (eventWaitList.isNotEmpty) {
      eventWaitListPtr = ffilib.calloc<cl_event>(eventWaitList.length);
      for (var i = 0; i < eventWaitList.length; ++i) {
        eventWaitListPtr[i] = eventWaitList[i].event;
      }
    }
    ffi.Pointer<cl_event> event = ffi.nullptr;
    if (createEvent) {
      event = ffilib.calloc<cl_event>();
    }
    final errcodeRet = dcl.clEnqueueNDRangeKernel(
        commandQueue,
        kernel.kernel,
        workDimensions,
        globalWorkOffsetPtr,
        globalWorkSizePtr,
        localWorkSizePtr,
        eventWaitList.length,
        eventWaitListPtr,
        event);

    assert(errcodeRet == CL_SUCCESS);

    Event? eventObj;
    if (createEvent) {
      eventObj = Event(event.value, dcl);
    }

    ffilib.calloc.free(globalWorkSizePtr);
    if (globalWorkOffsetPtr != ffi.nullptr) {
      ffilib.calloc.free(globalWorkOffsetPtr);
    }
    if (localWorkSizePtr != ffi.nullptr) {
      ffilib.calloc.free(localWorkSizePtr);
    }
    if (eventWaitListPtr != ffi.nullptr) {
      ffilib.calloc.free(eventWaitListPtr);
    }
    if (event != ffi.nullptr) {
      ffilib.calloc.free(event);
    }
    return eventObj;
  }

  int enqueueWaitForEvents(
    List<Event> eventWaitList,
  ) {
    if (eventWaitList.isEmpty) {
      throw ArgumentError('empty event wait list');
    }
    ffi.Pointer<cl_event> eventWaitListPtr;

    eventWaitListPtr = ffilib.calloc<cl_event>(eventWaitList.length);

    for (var i = 0; i < eventWaitList.length; ++i) {
      eventWaitListPtr[i] = eventWaitList[i].event;
    }

    final errcodeRet = dcl.clEnqueueWaitForEvents(
        commandQueue, eventWaitList.length, eventWaitListPtr);
    assert(errcodeRet == CL_SUCCESS);

    ffilib.calloc.free(eventWaitListPtr);

    return errcodeRet;
  }

  Event enqueueMarker() {
    ffi.Pointer<cl_event> event;
    event = ffilib.calloc<cl_event>();
    final errcodeRet = dcl.clEnqueueMarker(commandQueue, event);
    assert(errcodeRet == CL_SUCCESS);
    Event eventObj;
    eventObj = Event(event.value, dcl);
    ffilib.calloc.free(event);
    return eventObj;
  }

  int enqueueBarrier() {
    final errcodeRet = dcl.clEnqueueBarrier(commandQueue);
    assert(errcodeRet == CL_SUCCESS);
    return errcodeRet;
  }

  Event? _enqueueReadWriteBufferCommon(Mem buffer, int offset, int size,
      NativeBuffer ptr, ClEnqueueReadBufferType function,
      {bool createEvent = false,
      List<Event> eventWaitList = const [],
      bool blocking = false}) {
    ffi.Pointer<cl_event> eventWaitListPtr = ffi.nullptr;
    if (eventWaitList.isNotEmpty) {
      eventWaitListPtr = ffilib.calloc<cl_event>(eventWaitList.length);
      for (var i = 0; i < eventWaitList.length; ++i) {
        eventWaitListPtr[i] = eventWaitList[i].event;
      }
    }
    ffi.Pointer<cl_event> event = ffi.nullptr;
    if (createEvent) {
      event = ffilib.calloc<cl_event>();
    }
    final errcodeRet = function(
        commandQueue,
        buffer.mem,
        blocking ? 1 : 0,
        offset,
        size,
        ptr.ptr.cast(),
        eventWaitList.length,
        eventWaitListPtr,
        event);
    assert(errcodeRet == CL_SUCCESS);

    Event? eventObj;
    if (createEvent) {
      eventObj = Event(event.value, dcl);
    }

    if (eventWaitListPtr != ffi.nullptr) {
      ffilib.calloc.free(eventWaitListPtr);
    }
    if (event != ffi.nullptr) {
      ffilib.calloc.free(event);
    }
    return eventObj;
  }

  Event? enqueueReadBuffer(Mem buffer, int offset, int size, NativeBuffer ptr,
      {bool createEvent = false,
      bool blocking = false,
      List<Event> eventWaitList = const []}) {
    return _enqueueReadWriteBufferCommon(
        buffer, offset, size, ptr, dcl.clEnqueueReadBuffer,
        createEvent: createEvent,
        eventWaitList: eventWaitList,
        blocking: blocking);
  }

  Event? enqueueWriteBuffer(Mem buffer, int offset, int size, NativeBuffer ptr,
      {bool createEvent = false,
      bool blocking = false,
      List<Event> eventWaitList = const []}) {
    return _enqueueReadWriteBufferCommon(
        buffer, offset, size, ptr, dcl.clEnqueueWriteBuffer,
        createEvent: createEvent,
        eventWaitList: eventWaitList,
        blocking: blocking);
  }
}
