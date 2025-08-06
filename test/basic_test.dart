// ignore_for_file: avoid_print for test

import 'dart:ffi';

import 'package:opencl/opencl.dart';
import 'package:test/test.dart';

String vAddKernelDef = '''
__kernel void vector_add(__global const int4 *A, __global const int4 *B, __global int4 *C) {
  int i = get_global_id(0); // Get the index of the current element to be processed
  C[i] = A[i] + B[i];       // Do the operation
}
''';

const int listSize = 24 * 256 * 1024;

void main() {
  test('basic test', () {
    DynamicLibrary libraryCL;
    try {
      libraryCL = OpenCL.openDynLib();
    } catch (e) {
      throw Exception('could not load OpenCL dynamic library');
    }

    final cl = OpenCL(libraryCL);
    final platforms = cl.getPlatforms()
      ..forEach((platform) {
        print('Platform ${platform.name}:');
        print(platform.vendor);
        print(platform.version);
        print(platform.profile);
        print(platform.extensions);
        print(platform.hostTimerResolution);
        for (final device in platform.devices) {
          print('Device ${device.name}');
          print('Maximum compute units: ${device.maxComputeUnits}');
          print(device.profile);
          print(device.type);
          print(device.vendor);
          print(device.extensions);
          print(device.toJson());
        }
      });
    final context = cl.createContext(platforms[0].devices);
    final queue = context.createCommandQueue(platforms[0].devices[0]);
    final aBuf = NativeBuffer(4 * listSize);
    final bBuf = NativeBuffer(4 * listSize);
    final cBuf = NativeBuffer(4 * listSize);
    final aList = aBuf.byteBuffer.asInt32List();
    final bList = bBuf.byteBuffer.asInt32List();
    final cList = cBuf.byteBuffer.asInt32List();
    for (var i = 0; i < listSize; ++i) {
      aList[i] = i + 1;
      bList[i] = i * 2;
      cList[i] = -1;
    }

    final aMem = context.createBuffer(4 * listSize,
        hostData: aBuf, onlyCopy: true, kernelRead: true);
    final bMem = context.createBuffer(4 * listSize,
        hostData: bBuf, onlyCopy: true, kernelRead: true);
    final cMem =
        context.createBuffer(4 * listSize, kernelWrite: true, hostRead: true);

    final vAddProg = context.createProgramWithSource([vAddKernelDef]);
    final buildLog = <String>[];
    vAddProg.buildProgram(platforms[0].devices, '', buildLog);
    final vAddKernel = vAddProg.createKernel('vector_add')
      ..setKernelArgMem(0, aMem)
      ..setKernelArgMem(1, bMem)
      ..setKernelArgMem(2, cMem);
    final stopwatch = Stopwatch()..start();

    queue
      ..enqueueNDRangeKernel(vAddKernel, 1,
          globalWorkSize: [listSize ~/ 4], localWorkSize: [64])
      ..enqueueReadBuffer(cMem, 0, listSize * 4, cBuf, blocking: true);
    stopwatch.stop();
    print('cl executed in ${stopwatch.elapsed}');

    queue
      ..flush()
      ..finish()
      ..release();
    vAddKernel.release();
    vAddProg.release();
    aMem.release();
    bMem.release();
    cMem.release();

    context.release();

    stopwatch
      ..reset()
      ..start();

    final c4 = cList.buffer.asInt32x4List();
    final a4 = aList.buffer.asInt32x4List();
    final b4 = bList.buffer.asInt32x4List();
    for (var i = 0; i < listSize ~/ 4; ++i) {
      c4[i] = b4[i] + a4[i];
    }
    print('native executed in ${stopwatch.elapsed}');
    aBuf.free();
    bBuf.free();
    cBuf.free();

    expect(true, true);
  });
}
