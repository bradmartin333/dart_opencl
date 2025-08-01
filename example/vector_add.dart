import 'dart:ffi';

import 'package:opencl/opencl.dart';

const String vAddKernelDef = '''
__kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    // Do the operation
    C[i] = A[i] + B[i];

}
''';

const listSize = 1024;
const sizeOfInt32 = 4;

void main() {
  DynamicLibrary libraryCL;
  try {
    libraryCL = OpenCL.openDynLib();
  } catch (e) {
    throw Exception('could not load OpenCL dynamic library');
  }
  final cl = OpenCL(libraryCL);

  final platforms = cl.getPlatforms();
  // get first platform with at least one gpu device.
  final gpuPlatform = platforms.firstWhere((platform) =>
      platform.devices.any((device) => device.type == DeviceType.gpu));
  //get the first gpu device
  final gpuDevice =
      gpuPlatform.devices.firstWhere((device) => device.type == DeviceType.gpu);

  final context = cl.createContext([gpuDevice]);
  final queue = context.createCommandQueue(gpuDevice);
  final aBuf = NativeBuffer(sizeOfInt32 * listSize);
  final bBuf = NativeBuffer(sizeOfInt32 * listSize);
  final cBuf = NativeBuffer(sizeOfInt32 * listSize);
  final aList = aBuf.byteBuffer.asInt32List();
  final bList = bBuf.byteBuffer.asInt32List();
  final cList = cBuf.byteBuffer.asInt32List();
  for (var i = 0; i < listSize; ++i) {
    aList[i] = i + 1;
    bList[i] = i * 2;
    cList[i] = -1;
  }
  final aMem = context.createBuffer(sizeOfInt32 * listSize,
      hostData: aBuf, onlyCopy: true, kernelRead: true);
  final bMem = context.createBuffer(sizeOfInt32 * listSize,
      hostData: bBuf, onlyCopy: true, kernelRead: true);
  final cMem = context.createBuffer(sizeOfInt32 * listSize,
      kernelWrite: true, hostRead: true);

  final vAddProg = context.createProgramWithSource([vAddKernelDef])
    ..buildProgram([gpuDevice], '', []);
  final vAddKernel = vAddProg.createKernel('vector_add')
    ..setKernelArgMem(0, aMem)
    ..setKernelArgMem(1, bMem)
    ..setKernelArgMem(2, cMem);
  queue
    ..enqueueNDRangeKernel(vAddKernel, 1,
        globalWorkSize: [listSize], localWorkSize: [32])
    ..enqueueReadBuffer(cMem, 0, listSize * 4, cBuf, blocking: true)
    ..flush()
    ..finish()
    ..release();

  vAddKernel.release();
  vAddProg.release();
  aMem.release();
  bMem.release();
  cMem.release();
  aBuf.free();
  bBuf.free();
  cBuf.free();
  context.release();
}
