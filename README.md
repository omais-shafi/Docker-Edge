



# Docker-Edge

<h2> Installing a docker image with jetpack.<h3>

  <b> <i> sudo docker pull nvcr.io/nvidia/l4t-jetpack:r35.1.0 </i> <b> ---> This will download the docker containing all the necessary files needed to run TensorRT, cuda programs.

<h4> After the installation, you can do testing using these commands<h4>

  1.  <i>sudo docker images</i> ---> This will list the docker images present in your system including the above one.
  2. <i> sudo docker run -it <image_id> </i> ---> This will start the docker
  3. <i> sudo nvidia-docker run -it <image_id> </i> ---> This will enable the GPU inside the docker <br>
<b> NOTE:</b> To ensure GPU is detected inside the docker, the host and the docker cuda version should be exactly similar.
  4. <i> sudo docker ps </i>   ---> This will list the docker processes running along with the container id's.
  4. <i> sudo docker commit <container_id> <docker_new name> </i>  ---> To commit to a docker so that the changes are not lost



<h4> For all the applications, the README are provided in their respective directories.<h4>



