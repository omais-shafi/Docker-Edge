



# Docker-Edge

## Installing a docker image with jetpack.

<b> sudo docker pull nvcr.io/nvidia/l4t-jetpack:r35.1.0 <b>     ## This will download the docker containing all the necessary files needed to run TensorRT, cuda programs.

## After the installation, you can do testing using these commands.

* sudo docker images      ## This will list the docker images present in your system including the above one.
* sudo docker run -it <image_id>   ## This will start the docker
* sudo nvidia-docker run -it <image_id>  ## This will enable the GPU inside the docker <b> NOTE</b> To ensure GPU is detected inside the docker, the host and the docker cuda version should be exactly similar.
* sudo docker commit <container_id> <docker_new name>  ## To commit to a docker so that the changes are not lost



## For all the applications, the README are provided in their respective directories.



