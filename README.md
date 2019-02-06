# Getting started

Create a virtual env with the requirements:

`mkvirtualenv –p /usr/bin/python3 –r requirements.txt snacks-hacks-tfgan`

OR run the TensorFlow image (remove “-py3”if you’re old-fashioned):

`docker run -v .:/opt/anchormen -it tensorflow/tensorflow:latest-py3 bash`

OR (if you need more control) modify, build and start the provided Dockerfile:

`docker build –t snacks-hacks-tfgan Dockerfile.tfgan`

`docker run -v .:/opt/anchormen -it snacks-hacks-tfgan bash`
