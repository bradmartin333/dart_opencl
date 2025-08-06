import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart' as ffilib;
import 'package:opencl/opencl.dart';

enum DeviceType { cpu, gpu, accel, custom }

class DeviceAndHostTimer {
  DeviceAndHostTimer(this.deviceTimeStamp, this.hostTimeStamp);

  int deviceTimeStamp;
  int hostTimeStamp;
}

const Map<int, DeviceType> deviceTypeMap = {
  CL_DEVICE_TYPE_CPU: DeviceType.cpu,
  CL_DEVICE_TYPE_GPU: DeviceType.gpu,
  CL_DEVICE_TYPE_ACCELERATOR: DeviceType.accel,
  CL_DEVICE_TYPE_CUSTOM: DeviceType.custom
};

enum DeviceMemCacheType { none, readOnlyCache, readWriteCache }

class Device {
  //TODOAdd missing
  Device(
    this.device,
    this.dcl,
    this.type,
    this.profile,
    this.name,
    this.vendor,
    this.maxComputeUnits,
    this.extensions,
    this.vendorId,
    this.maxWorkItemDimensions,
    this.maxWorkItemSizes,
    this.maxWorkGroupSize,
    this.preferredVectorWidthChar,
    this.preferredVectorWidthShort,
    this.preferredVectorWidthInt,
    this.preferredVectorWidthLong,
    this.preferredVectorWidthFloat,
    this.preferredVectorWidthDouble,
    this.preferredVectorWidthHalf,
    this.nativeVectorWidthChar,
    this.nativeVectorWidthShort,
    this.nativeVectorWidthInt,
    this.nativeVectorWidthLong,
    this.nativeVectorWidthFloat,
    this.nativeVectorWidthDouble,
    this.nativeVectorWidthHalf,
    this.maxClockFrequency,
    this.addressBits,
    this.maxMemAllocSize,
    // ignore: avoid_positional_boolean_parameters - maybe add later
    this.imageSupport,
    this.maxReadImageArgs,
    this.maxWriteImageArgs,
    this.maxReadWriteImageArgs,
    this.ilVersion,
    this.maxParameterSize,
    this.image2DMaxWidth,
    this.image2DMaxHeight,
    this.image3DMaxWidth,
    this.image3DMaxHeight,
    this.image3DMaxDepth,
    this.imageMaxBufferSize,
    this.imageMaxArraySize,
    this.maxSamplers,
    this.imagePitchAlignment,
    this.imageBaseAddressAlignment,
    this.maxPipeArgs,
    this.pipeMaxActiveReservations,
    this.pipeMaxPacketSize,
    this.memBaseAddrAlign,
    this.fpDenorm,
    this.fpInfNan,
    this.fpRoundToNearest,
    this.fpRoundToZero,
    this.fpRoundToInf,
    this.fpFMA,
    this.fpCorrectlyRoundedDivideSqrt,
    this.fpSoftFloat,
    this.dblDenorm,
    this.dblInfNan,
    this.dblRoundToNearest,
    this.dblRoundToZero,
    this.dblRoundToInf,
    this.dblFMA,
    this.dblCorrectlyRoundedDivideSqrt,
    this.dblSoftFloat,
    this.globalMemCacheType,
    this.globalMemCachelineSize,
    this.globalMemCacheSize,
    this.globalMemSize,
    this.maxConstantBufferSize,
    this.maxConstantArgs,
    this.maxGlobalVariableSize,
    this.localMemType,
    this.localMemSize,
  );

  cl_device_id device;
  OpenCL dcl;
  DeviceType type;
  String profile;
  String name;
  String vendor;
  int maxComputeUnits;
  List<String> extensions;
  int vendorId;
  int maxWorkItemDimensions;
  List<int> maxWorkItemSizes;
  int maxWorkGroupSize;
  int preferredVectorWidthChar;
  int preferredVectorWidthShort;
  int preferredVectorWidthInt;
  int preferredVectorWidthLong;
  int preferredVectorWidthFloat;
  int preferredVectorWidthDouble;
  int preferredVectorWidthHalf;

  int nativeVectorWidthChar;
  int nativeVectorWidthShort;
  int nativeVectorWidthInt;
  int nativeVectorWidthLong;
  int nativeVectorWidthFloat;
  int nativeVectorWidthDouble;
  int nativeVectorWidthHalf;

  /// in MHz
  int maxClockFrequency;
  int addressBits;
  int maxMemAllocSize;
  bool imageSupport;
  int maxReadImageArgs;
  int maxWriteImageArgs;
  int maxReadWriteImageArgs;
  String ilVersion;
  int maxParameterSize;
  int image2DMaxWidth;
  int image2DMaxHeight;
  int image3DMaxWidth;
  int image3DMaxHeight;
  int image3DMaxDepth;
  int imageMaxBufferSize;
  int imageMaxArraySize;
  int maxSamplers;
  int imagePitchAlignment;
  int imageBaseAddressAlignment;
  int maxPipeArgs;
  int pipeMaxActiveReservations;
  int pipeMaxPacketSize;
  int memBaseAddrAlign;

  ///Masked CL_​DEVICE_​SINGLE_​FP_​CONFIG
  bool fpDenorm;

  ///Masked CL_​DEVICE_​SINGLE_​FP_​CONFIG
  bool fpInfNan;

  ///Masked CL_​DEVICE_​SINGLE_​FP_​CONFIG
  bool fpRoundToNearest;

  ///Masked CL_​DEVICE_​SINGLE_​FP_​CONFIG
  bool fpRoundToZero;

  ///Masked CL_​DEVICE_​SINGLE_​FP_​CONFIG
  bool fpRoundToInf;

  ///Masked CL_​DEVICE_​SINGLE_​FP_​CONFIG
  bool fpFMA;

  ///Masked CL_​DEVICE_​SINGLE_​FP_​CONFIG
  bool fpCorrectlyRoundedDivideSqrt;

  ///Masked CL_​DEVICE_​SINGLE_​FP_​CONFIG
  bool fpSoftFloat;

  ///Masked CL_​DEVICE_​DOUBLE_​FP_​CONFIG
  bool dblDenorm;

  ///Masked CL_​DEVICE_​DOUBLE_​FP_​CONFIG
  bool dblInfNan;

  ///Masked CL_​DEVICE_​DOUBLE_​FP_​CONFIG
  bool dblRoundToNearest;

  ///Masked CL_​DEVICE_​DOUBLE_​FP_​CONFIG
  bool dblRoundToZero;

  ///Masked CL_​DEVICE_​DOUBLE_​FP_​CONFIG
  bool dblRoundToInf;

  ///Masked CL_​DEVICE_​DOUBLE_​FP_​CONFIG
  bool dblFMA;

  ///Masked CL_​DEVICE_​DOUBLE_​FP_​CONFIG
  bool dblCorrectlyRoundedDivideSqrt;

  ///Masked CL_​DEVICE_​DOUBLE_​FP_​CONFIG
  bool dblSoftFloat;

  DeviceMemCacheType globalMemCacheType;

  int globalMemCachelineSize;

  int globalMemCacheSize;

  int globalMemSize;

  int maxConstantBufferSize;

  int maxConstantArgs;
  int maxGlobalVariableSize;

  DeviceMemCacheType localMemType;

  int localMemSize;

  Map<String, dynamic> toJson() {
    return {
      'type': type.toString(),
      'profile': profile,
      'name': name,
      'vendor': vendor,
      'vendorId': vendorId,
      'maxComputeUnits': maxComputeUnits,
      'extensions': extensions,
      'maxWorkItemDimensions': maxWorkItemDimensions,
      'maxWorkItemSizes': maxWorkItemSizes,
      'maxWorkGroupSize': maxWorkGroupSize,
      'preferredVectorWidthChar': preferredVectorWidthChar,
      'preferredVectorWidthShort': preferredVectorWidthShort,
      'preferredVectorWidthInt': preferredVectorWidthInt,
      'preferredVectorWidthLong': preferredVectorWidthLong,
      'preferredVectorWidthFloat': preferredVectorWidthFloat,
      'preferredVectorWidthDouble': preferredVectorWidthDouble,
      'preferredVectorWidthHalf': preferredVectorWidthHalf,
      'nativeVectorWidthChar': nativeVectorWidthChar,
      'nativeVectorWidthShort': nativeVectorWidthShort,
      'nativeVectorWidthInt': nativeVectorWidthInt,
      'nativeVectorWidthLong': nativeVectorWidthLong,
      'nativeVectorWidthFloat': nativeVectorWidthFloat,
      'nativeVectorWidthDouble': nativeVectorWidthDouble,
      'nativeVectorWidthHalf': nativeVectorWidthHalf,
      'maxClockFrequency': maxClockFrequency,
      'addressBits': addressBits,
      'maxMemAllocSize': maxMemAllocSize,
      'imageSupport': imageSupport,
      'maxReadImageArgs': maxReadImageArgs,
      'maxWriteImageArgs': maxWriteImageArgs,
      'maxReadWriteImageArgs': maxReadWriteImageArgs,
      'ilVersion': ilVersion,
      'maxParameterSize': maxParameterSize,
      'image2DMaxWidth': image2DMaxWidth,
      'image2DMaxHeight': image2DMaxHeight,
      'image3DMaxWidth': image3DMaxWidth,
      'image3DMaxHeight': image3DMaxHeight,
      'image3DMaxDepth': image3DMaxDepth,
      'imageMaxBufferSize': imageMaxBufferSize,
      'imageMaxArraySize': imageMaxArraySize,
      'maxSamplers': maxSamplers,
      'imagePitchAlignment': imagePitchAlignment,
      'imageBaseAddressAlignment': imageBaseAddressAlignment,
      'maxPipeArgs': maxPipeArgs,
      'pipeMaxActiveReservations': pipeMaxActiveReservations,
      'pipeMaxPacketSize': pipeMaxPacketSize,
      'memBaseAddrAlign': memBaseAddrAlign,
      'fpDenorm': fpDenorm,
      'fpInfNan': fpInfNan,
      'fpRoundToNearest': fpRoundToNearest,
      'fpRoundToZero': fpRoundToZero,
      'fpRoundToInf': fpRoundToInf,
      'fpFMA': fpFMA,
      'fpCorrectlyRoundedDivideSqrt': fpCorrectlyRoundedDivideSqrt,
      'fpSoftFloat': fpSoftFloat,
      'dblDenorm': dblDenorm,
      'dblInfNan': dblInfNan,
      'dblRoundToNearest': dblRoundToNearest,
      'dblRoundToZero': dblRoundToZero,
      'dblRoundToInf': dblRoundToInf,
      'dblFMA': dblFMA,
      'dblCorrectlyRoundedDivideSqrt': dblCorrectlyRoundedDivideSqrt,
      'dblSoftFloat': dblSoftFloat,
      'globalMemCacheType': globalMemCacheType,
      'globalMemCachelineSize': globalMemCachelineSize,
      'globalMemCacheSize': globalMemCacheSize,
      'globalMemSize': globalMemSize,
      'maxConstantBufferSize': maxConstantBufferSize,
      'maxConstantArgs': maxConstantArgs,
      'maxGlobalVariableSize': maxGlobalVariableSize,
      'localMemType': localMemType,
      'localMemSize': localMemSize
    };
  }
}

Device createDevice(cl_device_id device, OpenCL dcl) {
  const strBufferSize = 4096;
  final strbuf = ffilib.calloc<ffi.Int8>(strBufferSize);
  final ulongbuf = ffilib.calloc<ffi.UnsignedLong>();
  final uintbuf = ffilib.calloc<ffi.Uint32>();
  final outSize = ffilib.calloc<ffi.Size>();
  final sizeTbuffer = ffilib.calloc<ffi.Size>();

  dcl.clGetDeviceInfo(device, CL_DEVICE_TYPE, ffi.sizeOf<ffi.UnsignedLong>(),
      ulongbuf.cast(), outSize);
  final type = deviceTypeMap[ulongbuf.value] ?? DeviceType.custom;

  dcl.clGetDeviceInfo(
      device, CL_DEVICE_PROFILE, strBufferSize, strbuf.cast(), outSize);
  final profile = strbuf.toString();

  dcl.clGetDeviceInfo(
      device, CL_DEVICE_NAME, strBufferSize, strbuf.cast(), outSize);
  final name = strbuf.toString();

  dcl.clGetDeviceInfo(
      device, CL_DEVICE_VENDOR, strBufferSize, strbuf.cast(), outSize);
  final vendor = strbuf.toString();

  dcl.clGetDeviceInfo(
      device, CL_DEVICE_EXTENSIONS, strBufferSize, strbuf.cast(), outSize);
  final extensions = strbuf.toString().split(RegExp(' [a-zA-Z]'));

  dcl.clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final maxComputeUnits = uintbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, ffi.sizeOf<ffi.Uint32>(),
      uintbuf.cast(), outSize);
  final vendorId = uintbuf.value;
  dcl.clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final maxWorkItemDimensions = uintbuf.value;

  final dimensionsBuf = ffilib.calloc<ffi.Size>(maxWorkItemDimensions);

  dcl.clGetDeviceInfo(
      device,
      CL_DEVICE_MAX_WORK_ITEM_SIZES,
      ffi.sizeOf<ffi.Size>() * maxWorkItemDimensions,
      dimensionsBuf.cast(),
      outSize);
  final maxWorkItemSizes = List<int>.generate(
      maxWorkItemDimensions, (index) => dimensionsBuf[index]);

  dcl.clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
      ffi.sizeOf<ffi.Size>(), sizeTbuffer.cast(), outSize);
  final maxWorkGroupSize = sizeTbuffer.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final preferredVectorWidthChar = uintbuf.value;
  dcl.clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final preferredVectorWidthShort = uintbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final preferredVectorWidthInt = uintbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final preferredVectorWidthLong = uintbuf.value;
  dcl.clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final preferredVectorWidthFloat = uintbuf.value;
  dcl.clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final preferredVectorWidthDouble = uintbuf.value;
  dcl.clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final preferredVectorWidthHalf = uintbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final nativeVectorWidthChar = uintbuf.value;
  dcl.clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final nativeVectorWidthShort = uintbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final nativeVectorWidthInt = uintbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final nativeVectorWidthLong = uintbuf.value;
  dcl.clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final nativeVectorWidthFloat = uintbuf.value;
  dcl.clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final nativeVectorWidthDouble = uintbuf.value;
  dcl.clGetDeviceInfo(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final nativeVectorWidthHalf = uintbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final maxClockFrequency = uintbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, ffi.sizeOf<ffi.Uint32>(),
      uintbuf.cast(), outSize);
  final addressBits = uintbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
      ffi.sizeOf<ffi.UnsignedLong>(), ulongbuf.cast(), outSize);
  final maxMemAllocSize = ulongbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, ffi.sizeOf<ffi.Uint32>(),
      uintbuf.cast(), outSize);
  final imageSupport = uintbuf.value != 0;

  dcl.clGetDeviceInfo(device, CL_DEVICE_MAX_READ_IMAGE_ARGS,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final maxReadImageArgs = uintbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final maxWriteImageArgs = uintbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final maxReadWriteImageArgs = uintbuf.value;

  dcl.clGetDeviceInfo(
      device, CL_DEVICE_IL_VERSION, strBufferSize, strbuf.cast(), outSize);
  final ilVersion = strbuf.toString();

  /*
  CL_DEVICE_IMAGE2D_MAX_WIDTH
  CL_DEVICE_IMAGE2D_MAX_HEIGHT
  CL_DEVICE_IMAGE3D_MAX_WIDTH
  CL_DEVICE_IMAGE3D_MAX_HEIGHT
  CL_DEVICE_IMAGE3D_MAX_DEPTH
  CL_DEVICE_IMAGE_MAX_BUFFER_SIZE
  CL_DEVICE_IMAGE_MAX_ARRAY_SIZE*/

  dcl.clGetDeviceInfo(device, CL_DEVICE_MAX_PARAMETER_SIZE,
      ffi.sizeOf<ffi.Size>(), sizeTbuffer.cast(), outSize);
  final maxParameterSize = sizeTbuffer.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH,
      ffi.sizeOf<ffi.Size>(), sizeTbuffer.cast(), outSize);
  final image2DMaxWidth = sizeTbuffer.value;
  dcl.clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT,
      ffi.sizeOf<ffi.Size>(), sizeTbuffer.cast(), outSize);
  final image2DMaxHeight = sizeTbuffer.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_WIDTH,
      ffi.sizeOf<ffi.Size>(), sizeTbuffer.cast(), outSize);
  final image3DMaxWidth = sizeTbuffer.value;
  dcl.clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_HEIGHT,
      ffi.sizeOf<ffi.Size>(), sizeTbuffer.cast(), outSize);
  final image3DMaxHeight = sizeTbuffer.value;
  dcl.clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_DEPTH,
      ffi.sizeOf<ffi.Size>(), sizeTbuffer.cast(), outSize);
  final image3DMaxDepth = sizeTbuffer.value;
  dcl.clGetDeviceInfo(device, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE,
      ffi.sizeOf<ffi.Size>(), sizeTbuffer.cast(), outSize);
  final imageMaxBufferSize = sizeTbuffer.value;
  dcl.clGetDeviceInfo(device, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE,
      ffi.sizeOf<ffi.Size>(), sizeTbuffer.cast(), outSize);
  final imageMaxArraySize = sizeTbuffer.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_MAX_SAMPLERS, ffi.sizeOf<ffi.Uint32>(),
      uintbuf.cast(), outSize);
  final maxSamplers = uintbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_IMAGE_PITCH_ALIGNMENT,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final imagePitchAlignment = uintbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final imageBaseAddressAlignment = uintbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_MAX_PIPE_ARGS, ffi.sizeOf<ffi.Uint32>(),
      uintbuf.cast(), outSize);
  final maxPipeArgs = uintbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final pipeMaxActiveReservations = uintbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_PIPE_MAX_PACKET_SIZE,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final pipeMaxPacketSize = uintbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final memBaseAddrAlign = uintbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_SINGLE_FP_CONFIG,
      ffi.sizeOf<ffi.UnsignedLong>(), ulongbuf.cast(), outSize);
  final singleFPConfig = ulongbuf.value;
  final fpDenorm = (singleFPConfig & CL_FP_DENORM) != 0;
  final fpInfNan = (singleFPConfig & CL_FP_INF_NAN) != 0;
  final fpRoundToNearest = (singleFPConfig & CL_FP_ROUND_TO_NEAREST) != 0;
  final fpRoundToZero = (singleFPConfig & CL_FP_ROUND_TO_ZERO) != 0;
  final fpRoundToInf = (singleFPConfig & CL_FP_ROUND_TO_INF) != 0;
  final fpFMA = (singleFPConfig & CL_FP_FMA) != 0;
  final fpCorrectlyRoundedDivideSqrt =
      (singleFPConfig & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) != 0;
  final fpSoftFloat = (singleFPConfig & CL_FP_SOFT_FLOAT) != 0;

  dcl.clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG,
      ffi.sizeOf<ffi.UnsignedLong>(), ulongbuf.cast(), outSize);
  final doubleFPConfig = ulongbuf.value;
  final dblDenorm = (doubleFPConfig & CL_FP_DENORM) != 0;
  final dblInfNan = (doubleFPConfig & CL_FP_INF_NAN) != 0;
  final dblRoundToNearest = (doubleFPConfig & CL_FP_ROUND_TO_NEAREST) != 0;
  final dblRoundToZero = (doubleFPConfig & CL_FP_ROUND_TO_ZERO) != 0;
  final dblRoundToInf = (doubleFPConfig & CL_FP_ROUND_TO_INF) != 0;
  final dblFMA = (doubleFPConfig & CL_FP_FMA) != 0;
  final dblCorrectlyRoundedDivideSqrt =
      (doubleFPConfig & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) != 0;
  final dblSoftFloat = (doubleFPConfig & CL_FP_SOFT_FLOAT) != 0;

  dcl.clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final globalMemCacheType = DeviceMemCacheType.values[uintbuf.value];

  dcl.clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
      ffi.sizeOf<ffi.Int32>(), uintbuf.cast(), outSize);
  final globalMemCachelineSize = uintbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
      ffi.sizeOf<ffi.UnsignedLong>(), ulongbuf.cast(), outSize);
  final globalMemCacheSize = ulongbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE,
      ffi.sizeOf<ffi.UnsignedLong>(), ulongbuf.cast(), outSize);
  final globalMemSize = ulongbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
      ffi.sizeOf<ffi.UnsignedLong>(), ulongbuf.cast(), outSize);
  final maxConstantBufferSize = ulongbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_ARGS,
      ffi.sizeOf<ffi.Int32>(), uintbuf.cast(), outSize);
  final maxConstantArgs = uintbuf.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE,
      ffi.sizeOf<ffi.Size>(), sizeTbuffer.cast(), outSize);
  final maxGlobalVariableSize = sizeTbuffer.value;

  dcl.clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE,
      ffi.sizeOf<ffi.Uint32>(), uintbuf.cast(), outSize);
  final localMemType = DeviceMemCacheType.values[uintbuf.value];

  dcl.clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE,
      ffi.sizeOf<ffi.UnsignedLong>(), ulongbuf.cast(), outSize);
  final localMemSize = ulongbuf.value;

  ffilib.calloc.free(strbuf);
  ffilib.calloc.free(outSize);
  ffilib.calloc.free(ulongbuf);
  ffilib.calloc.free(uintbuf);
  ffilib.calloc.free(sizeTbuffer);
  ffilib.calloc.free(dimensionsBuf);
  return Device(
      device,
      dcl,
      type,
      profile,
      name,
      vendor,
      maxComputeUnits,
      extensions,
      vendorId,
      maxWorkItemDimensions,
      maxWorkItemSizes,
      maxWorkGroupSize,
      preferredVectorWidthChar,
      preferredVectorWidthShort,
      preferredVectorWidthInt,
      preferredVectorWidthLong,
      preferredVectorWidthFloat,
      preferredVectorWidthDouble,
      preferredVectorWidthHalf,
      nativeVectorWidthChar,
      nativeVectorWidthShort,
      nativeVectorWidthInt,
      nativeVectorWidthLong,
      nativeVectorWidthFloat,
      nativeVectorWidthDouble,
      nativeVectorWidthHalf,
      maxClockFrequency,
      addressBits,
      maxMemAllocSize,
      imageSupport,
      maxReadImageArgs,
      maxWriteImageArgs,
      maxReadWriteImageArgs,
      ilVersion,
      maxParameterSize,
      image2DMaxWidth,
      image2DMaxHeight,
      image3DMaxWidth,
      image3DMaxHeight,
      image3DMaxDepth,
      imageMaxBufferSize,
      imageMaxArraySize,
      maxSamplers,
      imagePitchAlignment,
      imageBaseAddressAlignment,
      maxPipeArgs,
      pipeMaxActiveReservations,
      pipeMaxPacketSize,
      memBaseAddrAlign,
      fpDenorm,
      fpInfNan,
      fpRoundToNearest,
      fpRoundToZero,
      fpRoundToInf,
      fpFMA,
      fpCorrectlyRoundedDivideSqrt,
      fpSoftFloat,
      dblDenorm,
      dblInfNan,
      dblRoundToNearest,
      dblRoundToZero,
      dblRoundToInf,
      dblFMA,
      dblCorrectlyRoundedDivideSqrt,
      dblSoftFloat,
      globalMemCacheType,
      globalMemCachelineSize,
      globalMemCacheSize,
      globalMemSize,
      maxConstantBufferSize,
      maxConstantArgs,
      maxGlobalVariableSize,
      localMemType,
      localMemSize);
}
