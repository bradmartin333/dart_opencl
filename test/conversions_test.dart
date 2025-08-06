import 'dart:ffi';

import 'package:opencl/opencl.dart';
import 'package:test/test.dart';

const sizeOfFloat = 4;
const sizeOfInt32 = 4;

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

  test('implicit conversion', () {
    const vImplicitConversion = '''
__kernel void foo(__global float *ret1, __global int *ret2) {
  float f = 3; // implicit conversion to float value 3.0
  int i = 5.23f; // implicit conversion to integer value 5
  ret1[0] = f;
  ret2[0] = i;
}
''';

    final ret1Buf = NativeBuffer(sizeOfFloat);
    final ret2Buf = NativeBuffer(sizeOfInt32);

    final ret1Mem = context.createBuffer(sizeOfFloat,
        hostData: ret1Buf, kernelWrite: true, hostRead: true);
    final ret2Mem = context.createBuffer(sizeOfInt32,
        hostData: ret2Buf, kernelWrite: true, hostRead: true);

    final vImplicitProg =
        context.createProgramWithSource([vImplicitConversion]);
    final buildLog = <String>[];
    vImplicitProg.buildProgram(platforms[0].devices, '', buildLog);

    final vImplicitKernel = vImplicitProg.createKernel('foo')
      ..setKernelArgMem(0, ret1Mem)
      ..setKernelArgMem(1, ret2Mem);

    queue
      ..enqueueNDRangeKernel(vImplicitKernel, 1,
          globalWorkSize: [1], localWorkSize: [1])
      ..enqueueReadBuffer(ret1Mem, 0, ret1Buf.size, ret1Buf, blocking: true)
      ..enqueueReadBuffer(ret2Mem, 0, ret2Buf.size, ret2Buf, blocking: true)
      ..flush()
      ..finish()
      ..release();

    vImplicitKernel.release();
    vImplicitProg.release();
    ret1Mem.release();
    ret2Mem.release();

    final ret1 = ret1Buf.byteBuffer.asFloat32List();
    final ret2 = ret2Buf.byteBuffer.asInt32List();

    expect(ret1[0], 3.0);
    expect(ret2[0], 5);

    ret1Buf.free();
    ret2Buf.free();
  });

  test('invalid conversion', () {
    const vImplicitConversion = '''
__kernel void foo() {
  float4 f;
  int4 i;
  f = i; // illegal implicit conversion between vector data types
}
''';

    final vImplicitProg =
        context.createProgramWithSource([vImplicitConversion]);
    final buildLog = <String>[];
    vImplicitProg.buildProgram(platforms[0].devices, '', buildLog);
    expect(
      () => vImplicitProg.createKernel('foo').release(),
      throwsA(isA<AssertionError>()),
    );

    vImplicitProg.release();
  });

  test('legal vector arithmetic conversions', () {
    const vLegalConversion = '''
__kernel void foo(__global float4 *ret1, __global float4 *ret2) {
  int a = 5;
  float4 b = (float4)(1.0f, 2.0f, 3.0f, 4.0f);
  float4 c = b + a; // int a converted to float4
  ret1[0] = c;

  short s = 10;
  float4 d = (float4)(5.0f, 6.0f, 7.0f, 8.0f);
  float4 e = d + s; // short s converted to float4
  ret2[0] = e;
}
''';

    final ret1Buf = NativeBuffer(sizeOfFloat * 4);
    final ret2Buf = NativeBuffer(sizeOfFloat * 4);

    final ret1Mem = context.createBuffer(sizeOfFloat * 4,
        hostData: ret1Buf, kernelWrite: true, hostRead: true);
    final ret2Mem = context.createBuffer(sizeOfFloat * 4,
        hostData: ret2Buf, kernelWrite: true, hostRead: true);

    final vLegalProg = context.createProgramWithSource([vLegalConversion]);
    final buildLog = <String>[];
    vLegalProg.buildProgram(platforms[0].devices, '', buildLog);

    final vLegalKernel = vLegalProg.createKernel('foo')
      ..setKernelArgMem(0, ret1Mem)
      ..setKernelArgMem(1, ret2Mem);

    queue
      ..enqueueNDRangeKernel(vLegalKernel, 1,
          globalWorkSize: [1], localWorkSize: [1])
      ..enqueueReadBuffer(ret1Mem, 0, ret1Buf.size, ret1Buf, blocking: true)
      ..enqueueReadBuffer(ret2Mem, 0, ret2Buf.size, ret2Buf, blocking: true)
      ..flush()
      ..finish()
      ..release();

    vLegalKernel.release();
    vLegalProg.release();
    ret1Mem.release();
    ret2Mem.release();

    final ret1 = ret1Buf.byteBuffer.asFloat32List();
    final ret2 = ret2Buf.byteBuffer.asFloat32List();

    expect(ret1[0], 6.0);
    expect(ret1[1], 7.0);
    expect(ret1[2], 8.0);
    expect(ret1[3], 9.0);

    expect(ret2[0], 15.0);
    expect(ret2[1], 16.0);
    expect(ret2[2], 17.0);
    expect(ret2[3], 18.0);

    ret1Buf.free();
    ret2Buf.free();
  });

  test('illegal vector arithmetic conversions', () {
    const vIllegalConversion = '''
__kernel void foo() {
  int a;
  short4 b;
  short4 c = b + a; // illegal: cannot convert & widen int to short4
}
''';

    final vIllegalProg = context.createProgramWithSource([vIllegalConversion]);
    final buildLog = <String>[];
    vIllegalProg.buildProgram(platforms[0].devices, '', buildLog);
    expect(
      () => vIllegalProg.createKernel('foo').release(),
      throwsA(isA<AssertionError>()),
    );

    vIllegalProg.release();
  });

  test('legal explicit casts', () {
    const vExplicitConversion = '''
__kernel void foo(__global int *ret1, __global float4 *ret2, __global float4 *ret3, __global int2 *ret4, __global int4 *ret5, __global int4 *ret6) {
  float f_scalar = 1.0f;
  int i_scalar = (int)f_scalar; // scalar float to int
  ret1[0] = i_scalar;

  float f_vec_scalar = 1.0f;
  float4 va = (float4)f_vec_scalar; // scalar to float4
  ret2[0] = va;

  uchar u_vec_scalar = 0xFF; // 255
  float4 vb = (float4)u_vec_scalar; // uchar to float4
  ret3[0] = vb;

  float f_int2_scalar = 2.0f;
  int2 vc = (int2)f_int2_scalar; // float to int2 (round toward zero)
  ret4[0] = vc;

  bool b_true = true;
  int4 i_true = (int4)b_true; // bool true to int4
  ret5[0] = i_true;

  bool b_false = false;
  int4 i_false = (int4)b_false; // bool false to int4
  ret6[0] = i_false;
}
''';

    final ret1Buf = NativeBuffer(sizeOfInt32);
    final ret2Buf = NativeBuffer(sizeOfFloat * 4);
    final ret3Buf = NativeBuffer(sizeOfFloat * 4);
    final ret4Buf = NativeBuffer(sizeOfInt32 * 2);
    final ret5Buf = NativeBuffer(sizeOfInt32 * 4);
    final ret6Buf = NativeBuffer(sizeOfInt32 * 4);

    final ret1Mem = context.createBuffer(sizeOfInt32,
        hostData: ret1Buf, kernelWrite: true, hostRead: true);
    final ret2Mem = context.createBuffer(sizeOfFloat * 4,
        hostData: ret2Buf, kernelWrite: true, hostRead: true);
    final ret3Mem = context.createBuffer(sizeOfFloat * 4,
        hostData: ret3Buf, kernelWrite: true, hostRead: true);
    final ret4Mem = context.createBuffer(sizeOfInt32 * 2,
        hostData: ret4Buf, kernelWrite: true, hostRead: true);
    final ret5Mem = context.createBuffer(sizeOfInt32 * 4,
        hostData: ret5Buf, kernelWrite: true, hostRead: true);
    final ret6Mem = context.createBuffer(sizeOfInt32 * 4,
        hostData: ret6Buf, kernelWrite: true, hostRead: true);

    final vExplicitProg =
        context.createProgramWithSource([vExplicitConversion]);
    final buildLog = <String>[];
    vExplicitProg.buildProgram(platforms[0].devices, '', buildLog);

    final vExplicitKernel = vExplicitProg.createKernel('foo')
      ..setKernelArgMem(0, ret1Mem)
      ..setKernelArgMem(1, ret2Mem)
      ..setKernelArgMem(2, ret3Mem)
      ..setKernelArgMem(3, ret4Mem)
      ..setKernelArgMem(4, ret5Mem)
      ..setKernelArgMem(5, ret6Mem);

    queue
      ..enqueueNDRangeKernel(vExplicitKernel, 1,
          globalWorkSize: [1], localWorkSize: [1])
      ..enqueueReadBuffer(ret1Mem, 0, ret1Buf.size, ret1Buf, blocking: true)
      ..enqueueReadBuffer(ret2Mem, 0, ret2Buf.size, ret2Buf, blocking: true)
      ..enqueueReadBuffer(ret3Mem, 0, ret3Buf.size, ret3Buf, blocking: true)
      ..enqueueReadBuffer(ret4Mem, 0, ret4Buf.size, ret4Buf, blocking: true)
      ..enqueueReadBuffer(ret5Mem, 0, ret5Buf.size, ret5Buf, blocking: true)
      ..enqueueReadBuffer(ret6Mem, 0, ret6Buf.size, ret6Buf, blocking: true)
      ..flush()
      ..finish()
      ..release();

    vExplicitKernel.release();
    vExplicitProg.release();
    ret1Mem.release();
    ret2Mem.release();
    ret3Mem.release();
    ret4Mem.release();
    ret5Mem.release();
    ret6Mem.release();

    expect(ret1Buf.byteBuffer.asInt32List()[0], 1); // scalar float to int

    final va = ret2Buf.byteBuffer.asFloat32List();
    expect(va[0], 1.0);
    expect(va[1], 1.0);
    expect(va[2], 1.0);
    expect(va[3], 1.0);

    final vb = ret3Buf.byteBuffer.asFloat32List();
    expect(vb[0], 255.0);
    expect(vb[1], 255.0);
    expect(vb[2], 255.0);
    expect(vb[3], 255.0);

    final vc = ret4Buf.byteBuffer.asInt32List();
    expect(vc[0], 2);
    expect(vc[1], 2);

    final iTrue = ret5Buf.byteBuffer.asInt32List();
    expect(iTrue[0], -1);
    expect(iTrue[1], -1);
    expect(iTrue[2], -1);
    expect(iTrue[3], -1);

    final iFalse = ret6Buf.byteBuffer.asInt32List();
    expect(iFalse[0], 0);
    expect(iFalse[1], 0);
    expect(iFalse[2], 0);
    expect(iFalse[3], 0);

    ret1Buf.free();
    ret2Buf.free();
    ret3Buf.free();
    ret4Buf.free();
    ret5Buf.free();
    ret6Buf.free();
  });

  test('illegal explicit vector casts', () {
    const vIllegalVectorCast1 = '''
__kernel void foo() {
  int4 i;
  uint4 u = (uint4)i; // compile error: explicit cast between vector types
}
''';

    final vIllegalProg1 =
        context.createProgramWithSource([vIllegalVectorCast1]);
    final buildLog1 = <String>[];
    vIllegalProg1.buildProgram(platforms[0].devices, '', buildLog1);

    var ok1 = true;
    try {
      vIllegalProg1.createKernel('foo').release();
    } catch (e) {
      ok1 = false;
    }
    expect(ok1, false);
    vIllegalProg1.release();

    const vIllegalVectorCast2 = '''
__kernel void foo() {
  float4 f;
  int4 i = (int4)f; // compile error: explicit cast between vector types
}
''';

    final vIllegalProg2 =
        context.createProgramWithSource([vIllegalVectorCast2]);
    final buildLog2 = <String>[];
    vIllegalProg2.buildProgram(platforms[0].devices, '', buildLog2);

    var ok2 = true;
    try {
      vIllegalProg2.createKernel('foo').release();
    } catch (e) {
      ok2 = false;
    }
    expect(ok2, false);
    vIllegalProg2.release();

    const vIllegalVectorCast3 = '''
__kernel void foo() {
  float4 f;
  int8 i = (int8)f; // compile error: explicit cast between vector types (different sizes)
}
''';

    final vIllegalProg3 =
        context.createProgramWithSource([vIllegalVectorCast3]);
    final buildLog3 = <String>[];
    vIllegalProg3.buildProgram(platforms[0].devices, '', buildLog3);

    var ok3 = true;
    try {
      vIllegalProg3.createKernel('foo').release();
    } catch (e) {
      ok3 = false;
    }
    expect(ok3, false);
    vIllegalProg3.release();
  });
}
