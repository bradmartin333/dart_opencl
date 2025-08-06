import 'dart:ffi';
import 'package:opencl/opencl.dart';
import 'package:test/test.dart';

const sizeOfFloat = 4;

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
    queue = context.createCommandQueue(platforms[0].devices[0],
        enableProfiling: true);
  });

  test('get execution time from event', () {
    const vAccessComponents = '''
__kernel void complex_kernel(__global float2 *pos, __global float *output) {
  const int global_id = get_global_id(0);
  float sum = 0.0f;
  for (int i = 0; i < 1000000; i++) {
    float x = pos[global_id].x;
    float y = pos[global_id].y;
    x = sin(x) * cos(y);
    y = tan(x) + pow(y, 2.0f);
    sum += sqrt(x * x + y * y);
  }
  output[global_id] = sum;
}
''';

    const workSize = 1024;
    final posBuf = NativeBuffer(sizeOfFloat * 2 * workSize);
    final posList = posBuf.byteBuffer.asFloat32List();

    for (var i = 0; i < workSize; i++) {
      posList[i * 2] = i.toDouble();
      posList[i * 2 + 1] = (i + 1).toDouble();
    }

    final posMem =
        context.createBuffer(sizeOfFloat * 2 * workSize, hostData: posBuf);

    final outputBuf = NativeBuffer(sizeOfFloat * workSize);
    final outputMem = context.createBuffer(sizeOfFloat * workSize,
        kernelWrite: true, hostRead: true);

    final vAccessProg = context.createProgramWithSource([vAccessComponents]);
    final buildLog = <String>[];
    vAccessProg.buildProgram(platforms[0].devices, '', buildLog);
    final vAccessKernel = vAccessProg.createKernel('complex_kernel')
      ..setKernelArgMem(0, posMem)
      ..setKernelArgMem(1, outputMem);

    queue.enqueueNDRangeKernel(vAccessKernel, 1,
        globalWorkSize: [workSize], localWorkSize: [32]);

    final event = queue.enqueueReadBuffer(
        outputMem, 0, outputBuf.size, outputBuf,
        blocking: true, createEvent: true);

    expect(event!.executionTimeNs, greaterThan(0));

    queue
      ..flush()
      ..finish()
      ..release();
    vAccessKernel.release();
    vAccessProg.release();
    posMem.release();
    outputMem.release();
    posBuf.free();
    outputBuf.free();
  });
}
