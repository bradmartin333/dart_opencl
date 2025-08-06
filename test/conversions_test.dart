import 'dart:ffi';
import 'dart:typed_data';

import 'package:opencl/opencl.dart';
import 'package:test/test.dart';

const sizeOfFloat = 4;
const sizeOfInt32 = 4;
const sizeOfUint32 = 4;

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
    expect(
      () => vIllegalProg1.createKernel('foo').release(),
      throwsA(isA<AssertionError>()),
    );
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
    expect(
      () => vIllegalProg2.createKernel('foo').release(),
      throwsA(isA<AssertionError>()),
    );
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
    expect(
      () => vIllegalProg3.createKernel('foo').release(),
      throwsA(isA<AssertionError>()),
    );
    vIllegalProg3.release();
  });

  group('rounding modes', () {
    final floatValues = Float32List.fromList([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]);
    final sizeOfValues = sizeOfFloat * floatValues.length;

    late NativeBuffer valuesBuf;
    late NativeBuffer retBuf;
    late Mem valuesMem;
    late Mem retMem;

    setUp(() {
      valuesBuf = NativeBuffer(sizeOfValues);
      valuesBuf.byteBuffer.asFloat32List().setAll(0, floatValues);
      retBuf = NativeBuffer(sizeOfInt32 * floatValues.length);

      valuesMem = context.createBuffer(sizeOfValues,
          hostData: valuesBuf, kernelRead: true);
      retMem = context.createBuffer(sizeOfInt32 * floatValues.length,
          hostData: retBuf, kernelWrite: true, hostRead: true);
    });

    tearDown(() {
      valuesMem.release();
      retMem.release();
      valuesBuf.free();
      retBuf.free();
    });

    void runTest(String programSource, List<int> expectedResults) {
      final program = context.createProgramWithSource([programSource])
        ..buildProgram(platforms[0].devices, '', <String>[]);

      final kernel = program.createKernel('round_test')
        ..setKernelArgMem(0, valuesMem)
        ..setKernelArgMem(1, retMem);

      queue
        ..enqueueNDRangeKernel(kernel, 1,
            globalWorkSize: [floatValues.length], localWorkSize: [1])
        ..enqueueReadBuffer(retMem, 0, retBuf.size, retBuf, blocking: true)
        ..flush()
        ..finish()
        ..release();

      kernel.release();
      program.release();

      final results = retBuf.byteBuffer.asInt32List();
      expect(results, orderedEquals(expectedResults));
    }

    test('_rte (round to nearest even)', () {
      const programSource = '''
__kernel void round_test(__global float* values, __global int* ret) {
  int gid = get_global_id(0);
  ret[gid] = convert_int_rte(values[gid]);
}
''';
      runTest(programSource, [-2, -2, 0, 0, 2, 2]);
    });

    test('_rtz (round toward zero)', () {
      const programSource = '''
__kernel void round_test(__global float* values, __global int* ret) {
  int gid = get_global_id(0);
  ret[gid] = convert_int_rtz(values[gid]);
  }
''';
      runTest(programSource, [-2, -1, 0, 0, 1, 2]);
    });

    test('_rtp (round toward positive infinity)', () {
      const programSource = '''
__kernel void round_test(__global float* values, __global int* ret) {
  int gid = get_global_id(0);
  ret[gid] = convert_int_rtp(values[gid]);
}
''';
      runTest(programSource, [-2, -1, 0, 1, 2, 3]);
    });

    test('_rtn (round toward negative infinity)', () {
      const programSource = '''
__kernel void round_test(__global float* values, __global int* ret) {
  int gid = get_global_id(0);
  ret[gid] = convert_int_rtn(values[gid]);
}
''';
      runTest(programSource, [-3, -2, -1, 0, 1, 2]);
    });

    test('default rounding mode (round toward zero)', () {
      const programSource = '''
__kernel void round_test(__global float* values, __global int* ret) {
  int gid = get_global_id(0);
  ret[gid] = (int)values[gid];
}
''';
      runTest(programSource, [-2, -1, 0, 0, 1, 2]);
    });
  });

  group('data reinterpretation', () {
    test('masking off sign bit of a float', () {
      const kernelSource = '''
__kernel void mask_sign_bit(__global float *input_f, __global float *output_f) {
  float f = input_f[0];
  uint u = as_uint(f);
  f = as_float(u & ~(1 << 31));
  output_f[0] = f;
}
''';

      final inputBuf = NativeBuffer(sizeOfFloat);
      inputBuf.byteBuffer.asFloat32List()[0] =
          -1.0; // IEEE 754 for -1.0 is 0xBF800000
      final outputBuf = NativeBuffer(sizeOfFloat);

      final inputMem = context.createBuffer(sizeOfFloat,
          hostData: inputBuf, kernelRead: true);
      final outputMem = context.createBuffer(sizeOfFloat,
          hostData: outputBuf, kernelWrite: true, hostRead: true);

      final program = context.createProgramWithSource([kernelSource])
        ..buildProgram(platforms[0].devices, '', <String>[]);

      final kernel = program.createKernel('mask_sign_bit')
        ..setKernelArgMem(0, inputMem)
        ..setKernelArgMem(1, outputMem);

      queue
        ..enqueueNDRangeKernel(kernel, 1,
            globalWorkSize: [1], localWorkSize: [1])
        ..enqueueReadBuffer(outputMem, 0, outputBuf.size, outputBuf,
            blocking: true)
        ..flush()
        ..finish()
        ..release();

      kernel.release();
      program.release();
      inputMem.release();
      outputMem.release();

      // Expected output: 1.0 (sign bit masked off from -1.0)
      expect(outputBuf.byteBuffer.asFloat32List()[0], 1.0);

      inputBuf.free();
      outputBuf.free();
    });

    test('reinterpret uint as float (0x3f800000 to 1.0f)', () {
      const kernelSource = '''
__kernel void uint_to_float(__global uint *input_u, __global float *output_f) {
  uint u = input_u[0];
  float f = as_float(u);
  output_f[0] = f;
}
''';

      final inputBuf = NativeBuffer(sizeOfUint32);
      inputBuf.byteBuffer.asUint32List()[0] = 0x3f800000; // Represents 1.0f
      final outputBuf = NativeBuffer(sizeOfFloat);

      final inputMem = context.createBuffer(sizeOfUint32,
          hostData: inputBuf, kernelRead: true);
      final outputMem = context.createBuffer(sizeOfFloat,
          hostData: outputBuf, kernelWrite: true, hostRead: true);

      final program = context.createProgramWithSource([kernelSource])
        ..buildProgram(platforms[0].devices, '', <String>[]);

      final kernel = program.createKernel('uint_to_float')
        ..setKernelArgMem(0, inputMem)
        ..setKernelArgMem(1, outputMem);

      queue
        ..enqueueNDRangeKernel(kernel, 1,
            globalWorkSize: [1], localWorkSize: [1])
        ..enqueueReadBuffer(outputMem, 0, outputBuf.size, outputBuf,
            blocking: true)
        ..flush()
        ..finish()
        ..release();

      kernel.release();
      program.release();
      inputMem.release();
      outputMem.release();

      expect(outputBuf.byteBuffer.asFloat32List()[0], 1.0);

      inputBuf.free();
      outputBuf.free();
    });

    test('reinterpret float4 as int4', () {
      const kernelSource = '''
__kernel void float4_to_int4(__global float4 *input_f4, __global int4 *output_i4) {
  float4 f = input_f4[0];
  int4 i = as_int4(f);
  output_i4[0] = i;
}
''';

      final inputBuf = NativeBuffer(sizeOfFloat * 4);
      inputBuf.byteBuffer.asFloat32List().setAll(0, [1.0, 2.0, 3.0, 4.0]);
      final outputBuf = NativeBuffer(sizeOfInt32 * 4);

      final inputMem = context.createBuffer(sizeOfFloat * 4,
          hostData: inputBuf, kernelRead: true);
      final outputMem = context.createBuffer(sizeOfInt32 * 4,
          hostData: outputBuf, kernelWrite: true, hostRead: true);

      final program = context.createProgramWithSource([kernelSource])
        ..buildProgram(platforms[0].devices, '', <String>[]);

      final kernel = program.createKernel('float4_to_int4')
        ..setKernelArgMem(0, inputMem)
        ..setKernelArgMem(1, outputMem);

      queue
        ..enqueueNDRangeKernel(kernel, 1,
            globalWorkSize: [1], localWorkSize: [1])
        ..enqueueReadBuffer(outputMem, 0, outputBuf.size, outputBuf,
            blocking: true)
        ..flush()
        ..finish()
        ..release();

      kernel.release();
      program.release();
      inputMem.release();
      outputMem.release();

      final expectedIntValues = Int32List.fromList([
        0x3f800000, // 1.0f
        0x40000000, // 2.0f
        0x40400000, // 3.0f
        0x40800000 // 4.0f
      ]);
      expect(
          outputBuf.byteBuffer.asInt32List(), orderedEquals(expectedIntValues));

      inputBuf.free();
      outputBuf.free();
    });

    test('ternary selection operator with as_typen', () {
      const kernelSource = '''
__kernel void ternary_select(__global float4 *f_in, __global float4 *g_in, __global float4 *f_out) {
  float4 f = f_in[0];
  float4 g = g_in[0];
  int4 is_less = f < g; // Each component will be 0 if false, -1 (all bits set) if true.
  f = as_float4(as_int4(f) & is_less);
  f_out[0] = f;
}
''';

      final fInBuf = NativeBuffer(sizeOfFloat * 4);
      fInBuf.byteBuffer.asFloat32List().setAll(0, [10.0, 2.0, 30.0, 4.0]);
      final gInBuf = NativeBuffer(sizeOfFloat * 4);
      gInBuf.byteBuffer.asFloat32List().setAll(0, [5.0, 15.0, 25.0, 40.0]);
      final fOutBuf = NativeBuffer(sizeOfFloat * 4);

      final fInMem = context.createBuffer(sizeOfFloat * 4,
          hostData: fInBuf, kernelRead: true);
      final gInMem = context.createBuffer(sizeOfFloat * 4,
          hostData: gInBuf, kernelRead: true);
      final fOutMem = context.createBuffer(sizeOfFloat * 4,
          hostData: fOutBuf, kernelWrite: true, hostRead: true);

      final program = context.createProgramWithSource([kernelSource])
        ..buildProgram(platforms[0].devices, '', <String>[]);

      final kernel = program.createKernel('ternary_select')
        ..setKernelArgMem(0, fInMem)
        ..setKernelArgMem(1, gInMem)
        ..setKernelArgMem(2, fOutMem);

      queue
        ..enqueueNDRangeKernel(kernel, 1,
            globalWorkSize: [1], localWorkSize: [1])
        ..enqueueReadBuffer(fOutMem, 0, fOutBuf.size, fOutBuf, blocking: true)
        ..flush()
        ..finish()
        ..release();

      kernel.release();
      program.release();
      fInMem.release();
      gInMem.release();
      fOutMem.release();

      // f < g: [false, true, false, true]
      // Expected f: [0.0, 2.0, 0.0, 4.0]
      expect(fOutBuf.byteBuffer.asFloat32List(),
          orderedEquals([0.0, 2.0, 0.0, 4.0]));

      fInBuf.free();
      gInBuf.free();
      fOutBuf.free();
    });

    test('legal 4-component to 3-component vector reinterpretation', () {
      const kernelSource = '''
__kernel void float4_to_float3(__global float4 *input_f4, __global float3 *output_f3) {
  float4 f = input_f4[0];
  float3 g = as_float3(f); // g.xyz will have same values as f.xyz. g.w is undefined.
  output_f3[0] = g;
}
''';

      final inputBuf = NativeBuffer(sizeOfFloat * 4);
      inputBuf.byteBuffer.asFloat32List().setAll(0, [1.1, 2.2, 3.3, 4.4]);
      final outputBuf = NativeBuffer(sizeOfFloat * 3);

      final inputMem = context.createBuffer(sizeOfFloat * 4,
          hostData: inputBuf, kernelRead: true);
      final outputMem = context.createBuffer(sizeOfFloat * 3,
          hostData: outputBuf, kernelWrite: true, hostRead: true);

      final program = context.createProgramWithSource([kernelSource])
        ..buildProgram(platforms[0].devices, '', <String>[]);

      final kernel = program.createKernel('float4_to_float3')
        ..setKernelArgMem(0, inputMem)
        ..setKernelArgMem(1, outputMem);

      queue
        ..enqueueNDRangeKernel(kernel, 1,
            globalWorkSize: [1], localWorkSize: [1])
        ..enqueueReadBuffer(outputMem, 0, outputBuf.size, outputBuf,
            blocking: true)
        ..flush()
        ..finish()
        ..release();

      kernel.release();
      program.release();
      inputMem.release();
      outputMem.release();

      final outputList = outputBuf.byteBuffer.asFloat32List();
      expect(outputList.length, 3);
      expect(outputList[0].toStringAsFixed(1), '1.1');
      expect(outputList[1].toStringAsFixed(1), '2.2');
      expect(outputList[2].toStringAsFixed(1), '3.3');

      inputBuf.free();
      outputBuf.free();
    });

    test('illegal explicit vector casts (different sizes - float4 to double4)',
        () {
      const kernelSource = '''
__kernel void illegal_cast_float4_to_double4() {
  float4 f;
  double4 g = as_double4(f); // Error. Result and operand have different sizes.
}
''';

      final program = context.createProgramWithSource([kernelSource]);
      final buildLog = <String>[];
      program.buildProgram(platforms[0].devices, '', buildLog);
      expect(
        () => program.createKernel('illegal_cast_float4_to_double4').release(),
        throwsA(isA<AssertionError>()),
      );
      program.release();
    });
  });
}
