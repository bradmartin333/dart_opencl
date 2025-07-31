import 'package:opencl/opencl.dart';

class Mem {
  cl_mem mem;
  OpenCL dcl;
  Mem(this.mem, this.dcl);
  void retain() {
    int ret = dcl.clRetainMemObject(mem);
    assert(ret == CL_SUCCESS);
  }

  void release() {
    int ret = dcl.clReleaseMemObject(mem);
    assert(ret == CL_SUCCESS);
  }
}
