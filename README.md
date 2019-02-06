# Getting started

Create a virtual env with the requirements:

`mkvirtualenv –p /usr/bin/python3.5 –r requirements.txt snacks-hacks-tfgan`

OR run the TensorFlow image (remove “-py3”if you’re old-fashioned):

`docker run -v .:/opt/anchormen -it tensorflow/tensorflow:latest-py3 bash`

OR (if you need more control) modify, build and start the provided Dockerfile:

`docker build –t snacks-hacks-tfgan Dockerfile.tfgan`

`docker run -v .:/opt/anchormen -it snacks-hacks-tfgan bash`

# Assignment

Port the TensorFlow MNIST GAN (tf_mnist_examply.py) to TFGAN using the template (tfgan_mnist_template.py)

Challenge: Use the Pokemon dataset to generate a data set:

800 images of 256x256: https://www.kaggle.com/pedrolimasilva/starter-pokemon-images-dataset-75f0a782-b

Those same ones at 32x32: https://dataexchange.anchormen.nl:5001/sharing/FWp54dNf9

150 images of 32x32: https://github.com/CShorten/PokemonGANs

