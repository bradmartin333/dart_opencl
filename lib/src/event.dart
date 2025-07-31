import 'package:opencl/opencl.dart';

class Event
{
  cl_event event;
  OpenCL dcl;
  Event(this.event, this.dcl);
  void retain() {
    int ret = dcl.clRetainEvent(event);
    assert(ret == CL_SUCCESS);
  }

  void release() {
    int ret = dcl.clReleaseEvent(event);
    assert(ret == CL_SUCCESS);
  }
}

