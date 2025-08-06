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
    queue = context.createCommandQueue(platforms[0].devices[0]);
  });

  test('update position data', () {
    const vAccessComponents = '''
__kernel void foo(__global const float2 *pos) {
  pos->y = pos->x * 2.0f;
}
''';

    final posBuf = NativeBuffer(sizeOfFloat * 2);
    final posList = posBuf.byteBuffer.asFloat32List();
    posList[0] = 2.0;

    final posMem = context.createBuffer(sizeOfFloat * 2,
        hostData: posBuf, kernelWrite: true, hostRead: true);

    final vAccessProg = context.createProgramWithSource([vAccessComponents]);
    final buildLog = <String>[];
    vAccessProg.buildProgram(platforms[0].devices, '', buildLog);
    final vAccessKernel = vAccessProg.createKernel('foo')
      ..setKernelArgMem(0, posMem);

    queue
      ..enqueueNDRangeKernel(vAccessKernel, 1,
          globalWorkSize: [1], localWorkSize: [1])
      ..enqueueReadBuffer(posMem, 0, posBuf.size, posBuf, blocking: true)
      ..flush()
      ..finish()
      ..release();
    vAccessKernel.release();
    vAccessProg.release();

    final pos = posBuf.byteBuffer.asFloat32List();
    expect(pos[0] * 2.0 == pos[1], true);

    posMem.release();
    posBuf.free();
  });

  test('illegal access position data', () {
    const vAccessComponents = '''
__kernel void foo(__global const float2 *pos) {
  float invalid = pos->z;
}
''';

    final posBuf = NativeBuffer(sizeOfFloat * 2);

    final posMem = context.createBuffer(sizeOfFloat * 2,
        hostData: posBuf, kernelWrite: true, hostRead: true);

    final vAccessProg = context.createProgramWithSource([vAccessComponents]);
    final buildLog = <String>[];
    vAccessProg.buildProgram(platforms[0].devices, '', buildLog);
    expect(
      () => vAccessProg.createKernel('foo').release(),
      throwsA(isA<AssertionError>()),
    );
    
    vAccessProg.release();
    posMem.release();
    posBuf.free();
  });

  test('vector swizzling', () {
    const vAccessComponents = '''
__kernel void foo(__global const float4 *pos, __global float2 *ret) {
  int i = get_global_id(0);
  float4 vf = pos[i];

  float2 odd = vf.yw;   // vf.odd
  float2 even = vf.xz;  // vf.even

  float2 result;
  result.x = vf.x + vf.y + odd.x + odd.y + even.x + even.y;
  result.y = vf.z + vf.w + odd.x + odd.y + even.x + even.y;

  ret[i] = result;
}
''';

    final posBuf = NativeBuffer(sizeOfFloat * 4);
    final posList = posBuf.byteBuffer.asFloat32List();
    for (var i = 0; i < 4; i++) {
      posList[i] = i.toDouble() + 1.0; // [1.0, 2.0, 3.0, 4.0]
    }
    final retBuf = NativeBuffer(sizeOfFloat * 2);

    final posMem = context.createBuffer(sizeOfFloat * 4,
        hostData: posBuf, onlyCopy: true, kernelRead: true);
    final retMem = context.createBuffer(sizeOfFloat * 2,
        hostData: retBuf, kernelWrite: true, hostRead: true);

    final vAccessProg = context.createProgramWithSource([vAccessComponents]);
    final buildLog = <String>[];
    vAccessProg.buildProgram(platforms[0].devices, '', buildLog);

    final vAccessKernel = vAccessProg.createKernel('foo')
      ..setKernelArgMem(0, posMem)
      ..setKernelArgMem(1, retMem);

    queue
      ..enqueueNDRangeKernel(vAccessKernel, 1,
          globalWorkSize: [1], localWorkSize: [1])
      ..enqueueReadBuffer(retMem, 0, retBuf.size, retBuf, blocking: true)
      ..flush()
      ..finish()
      ..release();

    vAccessKernel.release();
    vAccessProg.release();
    posMem.release();
    retMem.release();
    posBuf.free();

    final ret = retBuf.byteBuffer.asFloat32List();
    expect(ret[0] == 13.0, true);
    expect(ret[1] == 17.0, true);

    retBuf.free();
  });
}
