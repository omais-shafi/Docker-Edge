## Steps

1.  `` mv TensorRTYolo trtexec `` 
2.  `` cp -r trtexec/ /usr/src/tensorrt/samples ``
3.  `` cd trtexec ``
4.  `` make -j 4 ``
5.  `` cd /usr/src/tensorrt/bin ``
6.  `` ./trtexec --loadEngine= yolov3-tiny-416.trt --fp16 ``


You can refer to README.md to read more about trtexec.
