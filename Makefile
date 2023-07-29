project_dir := $(shell basename $(shell pwd))
project_name := $(shell echo $(project_dir) | tr A-Z a-z)
container_ssh_port := 1428
container_jupyter_port := 1429
container_tensorboard_port := 1430
container_mlflow_port := 1431

username := $(shell whoami)

src_dir := "/home/${username}/Projects/${project_dir}"

docker_container_name := "${username}-${project_name}"

all: build run

build:
	docker build \
		--tag ${project_name} \
		--file Dockerfile  \
		.

run:
	if [ ! -d ${src_dir} ]; then mkdir ${src_dir}; fi

	docker run \
		--detach \
		--name=${docker_container_name} \
		--interactive \
		--tty \
		--shm-size=64g \
		--ipc=host \
		--gpus all \
		--publish ${container_ssh_port}:4444 \
		--publish ${container_jupyter_port}:8888 \
		--publish ${container_tensorboard_port}:6006 \
		--publish ${container_mlflow_port}:5000 \
		--volume ${src_dir}:/src \
		${project_name}:latest

rm:
	docker stop ${docker_container_name}
	docker rm ${docker_container_name}
