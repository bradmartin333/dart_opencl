// ignore_for_file: avoid_print for example

import 'dart:ffi';
import 'dart:math';
import 'dart:typed_data';

import 'package:opencl/opencl.dart';

String vMatMulVecKern = '''
__kernel void mat_mul_vec(const int N,__global const float4 *mat, __global const float4 *vec, __local float4* localVec, __global float *result) {
 
    // Get the index of the current element to be processed
    int gid = get_global_id(0);
    int iloc = get_local_id(0);
    int nloc = get_local_size(0);
    int k;
    int NCeil = ((N+3)/4)*4;
    int qN = NCeil/4;
    for (k=iloc; k<qN; k+=nloc)
    {
      localVec[k] = vec[k];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    float4 temp = (float4)(0.0f,0.0f,0.0f,0.0f);
    int rowStart= gid*qN;
    __global float4 *row = &mat[rowStart];
    for (k=0; k<qN; ++k)
    {
      temp+=localVec[k]*row[k];
      //temp+=vec[k]*row[k];
    }
    result[gid]=temp.x+temp.y+temp.z+temp.w;
}
''';

const matColumns = 200;
const matRows = 200;
const sizeOfFloat = 4;

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
  print('executing on ${gpuDevice.name}');
  final context = cl.createContext([gpuDevice]);
  final queue = context.createCommandQueue(gpuDevice);
  final matBuf = NativeBuffer(matColumns * matRows * sizeOfFloat);
  final vecBuf = NativeBuffer(matColumns * sizeOfFloat);
  final outBuf = NativeBuffer(matRows * sizeOfFloat);
  final matList = matBuf.byteBuffer.asFloat32List();
  final vecList = vecBuf.byteBuffer.asFloat32List();
  final outList = outBuf.byteBuffer.asFloat32List();
  final r = Random();
  for (var i = 0; i < matColumns * matRows; ++i) {
    matList[i] = r.nextDouble() * 2 - 1;
  }
  for (var i = 0; i < matColumns; ++i) {
    vecList[i] = r.nextDouble() * 2 - 1;
  }

  print('initialized random array');
  final matMem = context.createBuffer(matColumns * matRows * sizeOfFloat,
      hostData: matBuf, onlyCopy: true, kernelRead: true);
  final vecMem = context.createBuffer(matColumns * sizeOfFloat,
      hostData: vecBuf, onlyCopy: true, kernelRead: true);
  final outMem = context.createBuffer(matRows * sizeOfFloat,
      kernelWrite: true, hostRead: true);

  final vAddProg = context.createProgramWithSource([vMatMulVecKern]);
  final buildLogs = <String>[];
  final buildRet = vAddProg.buildProgram([gpuDevice], '', buildLogs);
  if (buildRet == CL_BUILD_PROGRAM_FAILURE) {
    print(buildLogs);
  }
  final vAddKernel = vAddProg.createKernel('mat_mul_vec')
    ..setKernelArgInt(0, matColumns)
    ..setKernelArgMem(1, matMem)
    ..setKernelArgMem(2, vecMem)
    ..setKernelArgLocal(3, matColumns * sizeOfFloat)
    ..setKernelArgMem(4, outMem);
  print('finished compiling');
  final stopwatch = Stopwatch()..start();

  queue
    ..enqueueNDRangeKernel(vAddKernel, 1,
        globalWorkSize: [matRows], localWorkSize: [200])
    ..enqueueReadBuffer(outMem, 0, matRows * sizeOfFloat, outBuf,
        blocking: true);
  stopwatch.stop();
  print('cl executed in ${stopwatch.elapsed}');

  queue
    ..flush()
    ..finish()
    ..release();
  vAddKernel.release();
  vAddProg.release();
  matMem.release();
  vecMem.release();
  outMem.release();

  context.release();

  stopwatch
    ..reset()
    ..start();

  final matList4 = matList.buffer.asFloat32x4List();
  final vecList4 = vecList.buffer.asFloat32x4List();
  final nativeList = Float32List(matRows);

  for (var i = 0; i < matRows; ++i) {
    var temp = Float32x4.zero();
    final rowStart = i * (matColumns ~/ 4);
    for (var j = 0; j < matColumns / 4; ++j) {
      temp += matList4[rowStart + j] * vecList4[j];
    }
    nativeList[i] = temp.x + temp.y + temp.z + temp.w;
  }

  print('native executed in ${stopwatch.elapsed}');
  for (var i = 0; i < matRows; ++i) {
    if (outList[i] != nativeList[i]) {
      print('$i: ${outList[i]} != ${nativeList[i]} ');
    }
  }

  matBuf.free();
  vecBuf.free();
  outBuf.free();
}
