import 'package:opencl/opencl.dart';

class Event {
  Event(this.event, this.dcl, {this.executionTimeNs = 0});

  cl_event event;
  OpenCL dcl;
  int executionTimeNs;

  void retain() {
    final ret = dcl.clRetainEvent(event);
    assert(ret == CL_SUCCESS);
  }

  void release() {
    final ret = dcl.clReleaseEvent(event);
    assert(ret == CL_SUCCESS);
  }
}
