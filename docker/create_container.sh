#!/bin/bash

read -p "Enter image name: [default: "${USER}"] " image_name
if [ -z ${image_name} ]; then
image_name=${USER}
fi

read  -p "Enter image tag: [default: "v1"] " image_tag
if [ -z ${image_tag} ]; then
image_tag="v1"
fi

# Create a tag or name for the image
docker_tag="aicregistry:5000/"${image_name}:${image_tag}
export GROUP_ID=$(id -g)
export USER_ID=$USER
echo "Docker tag: "${docker_tag}

# Build the image using your Dockerfile and arguments
echo "Building the image..."
echo "docker build -f Dockerfile . --tag ${docker_tag} --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg USER=${USER} --progress=plain --no-cache"
docker build -f Dockerfile . --tag ${docker_tag} --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg USER=${USER} --progress=plain --no-cache

# Push the built image to aicregistry
docker push ${docker_tag}
