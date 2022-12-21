## Steps

1.  <i> mv TensorRTYolo trtexec </i> 
2.  <i> cp -r trtexec/ /usr/src/tensorrt/samples </i>
3.  <i> cd trtexec </i>
4.  <i> make -j 4 </i>
5.  <i> cd /usr/src/tensorrt/bin </i>
6.  <i> ./trtexec --loadEngine= yolov3-tiny-416.trt --fp16 </i>


You can refer to README.md to read more about trtexec.
