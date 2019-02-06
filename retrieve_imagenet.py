from tqdm import tqdm
import requests
import tarfile


# Retrieve and untar imagenet urls

url = 'http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz'
tarname= "imagenet_fall11_urls.tar"

try :

    tar = tarfile.open(tarname)

except FileNotFoundError :

    print("Downloading ImageNet urls...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    with open(tarname, "wb") as handle:
        for data in tqdm(response.iter_content(block_size), total = total_size // block_size, unit ='KB'):
            handle.write(data)

    tar = tarfile.open(tarname)

finally :

    print("Extracting ImageNet urls...")
    tar.extractall()
    tar.close()

# Retrieve imagenet - wordnet mapping

url = 'http://image-net.org/archive/words.txt'
mappingname = 'mapping.txt'
print("Downloading ImageNet - WordNet mapping...")

response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))
block_size = 1024
with open(mappingname, "wb") as handle:
    for data in tqdm(response.iter_content(block_size), total = total_size // block_size, unit ='KB'):
        handle.write(data)
