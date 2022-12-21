## Running the above applications on the Host machine

1. Install the JetPack on the hostmachine using this link: https://developer.nvidia.com/embedded/jetpack
2. It will download all the necessary packages like TensorRT, CUDA etc.
3. Try running each of the applications on your host machine.



## Running the above applications inside the docker container

<h2> Installing a docker image with jetpack </h2>

 ``sudo docker pull nvcr.io/nvidia/l4t-jetpack:r35.1.0 `` -->This will download the docker containing all the necessary files needed to run TensorRT, cuda programs

<h4> After the installation, you can do testing using these commands</h4>

  1.  ``sudo docker images`` ---> This will list the docker images present in your system including the above one.
  2. `` sudo docker run -it <image_id> `` ---> This will start the docker
  3. `` sudo nvidia-docker run -it <image_id> `` ---> This will enable the GPU inside the docker <br>
<b> NOTE:</b> To ensure GPU is detected inside the docker, the host and the docker cuda version should be exactly similar.
  4. ``sudo docker ps ``   ---> This will list the docker processes running along with the container id's.
  4. `` sudo docker commit <container_id> <docker_new name> ``  ---> To commit to a docker so that the changes are not lost



<h4> For all the applications, the README are provided in their respective directories.<h4>



