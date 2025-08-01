import 'dart:ffi';
import 'dart:math';
import 'dart:typed_data';

import 'package:opencl/opencl.dart';
import 'package:test/test.dart';

const sizeOfHalf = 2;
const sizeOfFloat = 4;

/// Converts a standard Dart double (64-bit) to a 16-bit integer
/// representing an IEEE 754 half-precision float.
int float32ToFloat16Bits(double value) {
  final buffer = Uint8List(4).buffer;
  final byteData = ByteData.view(buffer)..setFloat32(0, value, Endian.host);
  final float32Bits = byteData.getUint32(0, Endian.host);
  final sign = (float32Bits >> 31) & 0x01;
  var exponent = (float32Bits >> 23) & 0xff;
  final significand = float32Bits & 0x7fffff;

  if (exponent == 0) {
    // Handle special cases
    return sign << 15; // Zero or denormalized
  } else if (exponent == 0xff) {
    // Infinity or NaN
    if (significand == 0) {
      // Infinity
      return (sign << 15) | 0x7c00;
    } else {
      // NaN
      return (sign << 15) | 0x7c00 | 0x0200; // Quiet NaN
    }
  }

  // Handle normalized numbers
  exponent = exponent - 127 + 15; // Re-bias from 127 to 15

  // Check for overflow (exponent > 30)
  if (exponent >= 31) {
    return (sign << 15) | 0x7c00; // Infinity with correct sign
  }

  // Check for underflow (exponent <= 0)
  if (exponent <= 0) {
    // This is where denormalized numbers are generated.
    // The exponent becomes 0, and the significand is shifted right.
    final expOffset = 1 - exponent;
    var denormalSignificand = (1 << 23) | significand;
    denormalSignificand >>= expOffset;
    return (sign << 15) | denormalSignificand;
  }

  // Normalized conversion
  final float16Bits = (sign << 15) | (exponent << 10) | (significand >> 13);
  return float16Bits;
}

/// Converts a 16-bit integer representing an IEEE 754 half-precision float
/// to a standard Dart double (64-bit).
double float16BitsToDouble(int halfBits) {
  // Deconstruct the 16-bit integer into its components:
  // 1 sign bit, 5 exponent bits, and 10 significand (mantissa) bits.
  final sign = (halfBits >> 15) & 0x01;
  final exponent = (halfBits >> 10) & 0x1f;
  final significand = halfBits & 0x03ff;

  // Handle special cases based on the exponent value.
  if (exponent == 0) {
    // Exponent is zero. This can be zero or a denormalized number.
    if (significand == 0) {
      // Both exponent and significand are zero, so the value is zero.
      return sign == 0 ? 0.0 : -0.0;
    } else {
      // Denormalized number. Exponent is effectively -14.
      return (sign == 0 ? 1 : -1) * pow(2, -14) * (significand / pow(2, 10));
    }
  } else if (exponent == 31) {
    // Exponent is all 1s. This indicates infinity or NaN.
    if (significand == 0) {
      // Significand is zero, so the value is infinity.
      return sign == 0 ? double.infinity : double.negativeInfinity;
    } else {
      // Significand is non-zero, so the value is NaN.
      return double.nan;
    }
  } else {
    // This is a normalized number.
    // The exponent bias is 15.
    // The implicit leading bit of the significand is 1.
    return (sign == 0 ? 1 : -1) *
        pow(2, exponent - 15) *
        (1 + significand / pow(2, 10));
  }
}

void main() {
  late List<Platform> platforms;
  late Context context;
  late CommandQueue queue;

  setUp(() async {
    DynamicLibrary libraryCL;
    try {
      libraryCL = OpenCL.openDynLib();
    } catch (e) {
      throw Exception('could not load OpenCL dynamic library');
    }

    final cl = OpenCL(libraryCL);
    platforms = cl.getPlatforms();
    context = cl.createContext(platforms[0].devices);
    queue = context.createCommandQueue(platforms[0].devices[0]);
  });

  test('add half', () {
    const vAddHalf = '''
__kernel void scalar_add(__global const half2 *A, __global const half2 *B, __global half2 *C) {
  int i = get_global_id(0);
  C[i] = A[i] + B[i];
}
''';

    final aBuf = NativeBuffer(sizeOfHalf * 2);
    final bBuf = NativeBuffer(sizeOfHalf * 2);
    final cBuf = NativeBuffer(sizeOfHalf * 2);
    final aList = aBuf.byteBuffer.asByteData();
    final bList = bBuf.byteBuffer.asByteData();

    final halfA1 = float32ToFloat16Bits(1.234);
    final halfA2 = float32ToFloat16Bits(2);
    final halfB1 = float32ToFloat16Bits(5.678);
    final halfB2 = float32ToFloat16Bits(3);

    aList
      ..setUint16(0, halfA1, Endian.little)
      ..setUint16(2, halfA2, Endian.little);
    bList
      ..setUint16(0, halfB1, Endian.little)
      ..setUint16(2, halfB2, Endian.little);

    final aMem = context.createBuffer(aBuf.size,
        hostData: aBuf, onlyCopy: true, kernelRead: true);
    final bMem = context.createBuffer(bBuf.size,
        hostData: bBuf, onlyCopy: true, kernelRead: true);
    final cMem =
        context.createBuffer(cBuf.size, kernelWrite: true, hostRead: true);

    final vAddHalfProg = context.createProgramWithSource([vAddHalf]);
    final buildLog = <String>[];
    vAddHalfProg.buildProgram(platforms[0].devices, '', buildLog);
    final vAddHalfKernel = vAddHalfProg.createKernel('scalar_add')
      ..setKernelArgMem(0, aMem)
      ..setKernelArgMem(1, bMem)
      ..setKernelArgMem(2, cMem);

    queue
      ..enqueueNDRangeKernel(vAddHalfKernel, 1,
          globalWorkSize: [1], localWorkSize: [1])
      ..enqueueReadBuffer(cMem, 0, cBuf.size, cBuf, blocking: true)
      ..flush()
      ..finish()
      ..release();
    vAddHalfKernel.release();
    vAddHalfProg.release();
    aMem.release();
    bMem.release();

    final cListRead = cBuf.byteBuffer.asByteData();
    final cBits1 = cListRead.getUint16(0, Endian.little);
    final cBits2 = cListRead.getUint16(2, Endian.little);
    final result1 = float16BitsToDouble(cBits1);
    final result2 = float16BitsToDouble(cBits2);

    aBuf.free();
    bBuf.free();
    cBuf.free();

    expect((result1 - (1.234 + 5.678)).abs() < 0.002, true);
    expect((result2 - (2.0 + 3.0)).abs() < 0.002, true);
  });

  test('multiply half', () {
    const vMultiplyHalf = '''
__kernel void scalar_multiply(__global const half2 *A, __global const half2 *B, __global half2 *C) {
  int i = get_global_id(0);
  C[i] = A[i] * B[i];
}
''';

    final aBuf = NativeBuffer(sizeOfHalf);
    final bBuf = NativeBuffer(sizeOfHalf);
    final cBuf = NativeBuffer(sizeOfHalf);
    final aList = aBuf.byteBuffer.asByteData();
    final bList = bBuf.byteBuffer.asByteData();
    final cList = cBuf.byteBuffer.asByteData();

    final halfA = float32ToFloat16Bits(1.5);
    final halfB = float32ToFloat16Bits(3);

    aList.setUint16(0, halfA, Endian.little);
    bList.setUint16(0, halfB, Endian.little);

    final aMem = context.createBuffer(aBuf.size,
        hostData: aBuf, onlyCopy: true, kernelRead: true);
    final bMem = context.createBuffer(bBuf.size,
        hostData: bBuf, onlyCopy: true, kernelRead: true);
    final cMem =
        context.createBuffer(cBuf.size, kernelWrite: true, hostRead: true);

    final vMultiplyHalfProg = context.createProgramWithSource([vMultiplyHalf]);
    final buildLog = <String>[];
    vMultiplyHalfProg.buildProgram(platforms[0].devices, '', buildLog);
    final vMultiplyHalfKernel =
        vMultiplyHalfProg.createKernel('scalar_multiply')
          ..setKernelArgMem(0, aMem)
          ..setKernelArgMem(1, bMem)
          ..setKernelArgMem(2, cMem);

    queue
      ..enqueueNDRangeKernel(vMultiplyHalfKernel, 1,
          globalWorkSize: [1], localWorkSize: [1])
      ..enqueueReadBuffer(cMem, 0, cBuf.size, cBuf, blocking: true)
      ..flush()
      ..finish()
      ..release();
    vMultiplyHalfKernel.release();
    vMultiplyHalfProg.release();
    aMem.release();
    bMem.release();

    final cBits = cList.getUint16(0, Endian.little);
    final result = float16BitsToDouble(cBits);

    aBuf.free();
    bBuf.free();
    cBuf.free();

    expect((result - (1.5 * 3.0)).abs() < 0.002, true);
  });

  test('convert float to half', () {
    const vConvertHalf = '''
__kernel void convert_float_to_half(__global const float *A, __global half *B) {
  int i = get_global_id(0);
  vstore_half(A[i], i, B);
}
''';

    final aBuf = NativeBuffer(sizeOfFloat);
    final bBuf = NativeBuffer(sizeOfHalf);
    aBuf.byteBuffer.asByteData().setFloat32(0, 1.234, Endian.little);

    final aMem = context.createBuffer(aBuf.size,
        hostData: aBuf, onlyCopy: true, kernelRead: true);
    final bMem =
        context.createBuffer(bBuf.size, kernelWrite: true, hostRead: true);

    final vConvertHalfProg = context.createProgramWithSource([vConvertHalf]);
    final buildLog = <String>[];
    vConvertHalfProg.buildProgram(platforms[0].devices, '', buildLog);
    final vConvertHalfKernel =
        vConvertHalfProg.createKernel('convert_float_to_half')
          ..setKernelArgMem(0, aMem)
          ..setKernelArgMem(1, bMem);

    queue
      ..enqueueNDRangeKernel(vConvertHalfKernel, 1,
          globalWorkSize: [1], localWorkSize: [1])
      ..enqueueReadBuffer(bMem, 0, bBuf.size, bBuf, blocking: true)
      ..flush()
      ..finish()
      ..release();
    vConvertHalfKernel.release();
    vConvertHalfProg.release();
    aMem.release();

    final bList = bBuf.byteBuffer.asByteData();
    final bBits = bList.getUint16(0, Endian.little);
    final result = float16BitsToDouble(bBits);

    aBuf.free();
    bBuf.free();

    final manualHalf = float32ToFloat16Bits(1.234);
    final manualResult = float16BitsToDouble(manualHalf);

    expect(result, closeTo(manualResult, 0.001));
  });

  test('convert half to float', () {
    const vConvertHalfToFloat = '''
__kernel void convert_half_to_float(__global const half *A, __global float *C) {
  int i = get_global_id(0);
  C[i] = vload_half(i, A);
}
''';

    final aBuf = NativeBuffer(sizeOfHalf);
    final cBuf = NativeBuffer(sizeOfFloat);

    final aList = aBuf.byteBuffer.asByteData();
    final halfA = float32ToFloat16Bits(123.456);
    aList.setUint16(0, halfA, Endian.little);

    final aMem = context.createBuffer(aBuf.size,
        hostData: aBuf, onlyCopy: true, kernelRead: true);
    final cMem =
        context.createBuffer(cBuf.size, kernelWrite: true, hostRead: true);

    final vConvertHalfToFloatProg =
        context.createProgramWithSource([vConvertHalfToFloat]);
    final buildLog = <String>[];
    vConvertHalfToFloatProg.buildProgram(platforms[0].devices, '', buildLog);
    final vConvertHalfToFloatKernel =
        vConvertHalfToFloatProg.createKernel('convert_half_to_float')
          ..setKernelArgMem(0, aMem)
          ..setKernelArgMem(1, cMem);

    queue
      ..enqueueNDRangeKernel(vConvertHalfToFloatKernel, 1,
          globalWorkSize: [1], localWorkSize: [1])
      ..enqueueReadBuffer(cMem, 0, cBuf.size, cBuf, blocking: true)
      ..flush()
      ..finish()
      ..release();
    vConvertHalfToFloatKernel.release();
    vConvertHalfToFloatProg.release();
    aMem.release();

    final cList = cBuf.byteBuffer.asByteData();
    final result = cList.getFloat32(0, Endian.little);

    aBuf.free();
    cBuf.free();

    // The half-to-float conversion should be lossless,
    // so we expect a near-perfect match
    final manualConverted = float16BitsToDouble(halfA);
    expect((result - manualConverted).abs() < 0.000001, true);
  });
}
