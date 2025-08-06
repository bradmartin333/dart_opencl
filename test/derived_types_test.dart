import 'dart:ffi';

import 'package:opencl/opencl.dart';
import 'package:test/test.dart';

const sizeOfFloat = 4;
const sizeOfInt32 = 4;

final class SimpleStruct extends Struct {
  @Float()
  external double a;
}

final class ComplexStruct extends Struct {
  @Float()
  external double a;

  @Array(2, 2)
  external Array<Array<Float>> matrix;
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

  test('derived struct access', () {
    const vStructAccess = '''
      typedef struct {
        float a;
      } SimpleStruct;

      __kernel void foo(__global SimpleStruct *ret) {
        ret[0].a = 12.34f;
      }
    ''';

    final structBuffer = NativeBuffer(sizeOf<SimpleStruct>());
    final structMem = context.createBuffer(
      structBuffer.size,
      hostData: structBuffer,
      kernelWrite: true,
      hostRead: true,
    );

    final structProg = context.createProgramWithSource([vStructAccess]);
    final buildLog = <String>[];
    structProg.buildProgram(platforms[0].devices, '', buildLog);

    final structKernel = structProg.createKernel('foo')
      ..setKernelArgMem(0, structMem);

    queue
      ..enqueueNDRangeKernel(
        structKernel,
        1,
        globalWorkSize: [1],
        localWorkSize: [1],
      )
      ..enqueueReadBuffer(structMem, 0, structBuffer.size, structBuffer,
          blocking: true)
      ..flush()
      ..finish()
      ..release();

    structKernel.release();
    structProg.release();
    structMem.release();

    final struct = structBuffer.byteBuffer.asFloat32List();
    expect(struct[0].toStringAsFixed(2), '12.34');

    structBuffer.free();
  });

  test('complex derived struct with matrix', () {
    const vComplexStructAccess = '''
      typedef struct {
        float a;
        float matrix[2][2];
      } ComplexStruct;

      __kernel void foo(__global ComplexStruct *ret) {
        ret[0].a = 42.0f;
        ret[0].matrix[0][0] = 1.0f;
        ret[0].matrix[0][1] = 2.0f;
        ret[0].matrix[1][0] = 3.0f;
        ret[0].matrix[1][1] = 4.0f;
      }
    ''';

    final structBuffer = NativeBuffer(sizeOf<ComplexStruct>());
    final structMem = context.createBuffer(
      structBuffer.size,
      hostData: structBuffer,
      kernelWrite: true,
      hostRead: true,
    );

    final structProg = context.createProgramWithSource([vComplexStructAccess]);
    final buildLog = <String>[];
    structProg.buildProgram(platforms[0].devices, '', buildLog);

    final structKernel = structProg.createKernel('foo')
      ..setKernelArgMem(0, structMem);

    queue
      ..enqueueNDRangeKernel(
        structKernel,
        1,
        globalWorkSize: [1],
        localWorkSize: [1],
      )
      ..enqueueReadBuffer(structMem, 0, structBuffer.size, structBuffer,
          blocking: true)
      ..flush()
      ..finish()
      ..release();

    structKernel.release();
    structProg.release();
    structMem.release();

    final complexStruct = structBuffer.byteBuffer.asFloat32List();
    expect(complexStruct[0], 42.0);
    expect(complexStruct[1], 1.0);
    expect(complexStruct[2], 2.0);
    expect(complexStruct[3], 3.0);
    expect(complexStruct[4], 4.0);

    structBuffer.free();
  });
}
