import 'package:opencl/opencl.dart';

class Mem {
  Mem(this.mem, this.dcl);

  cl_mem mem;
  OpenCL dcl;

  void retain() {
    final ret = dcl.clRetainMemObject(mem);
    assert(ret == CL_SUCCESS);
  }

  void release() {
    final ret = dcl.clReleaseMemObject(mem);
    assert(ret == CL_SUCCESS);
  }
}
