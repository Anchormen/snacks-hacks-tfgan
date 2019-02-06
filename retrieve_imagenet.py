from tqdm import tqdm
import requests
import tarfile
import os

# Retrieve imagenet - wordnet mapping

url = 'http://image-net.org/archive/words.txt'
mappingname = 'mapping.txt'

try :
    mapping = open(mappingname)

except FileNotFoundError :

    print("Downloading ImageNet - WordNet mapping...")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    with open(mappingname, "wb") as handle:
        for data in tqdm(response.iter_content(block_size), total = total_size // block_size, unit ='KB'):
            handle.write(data)

    mapping = mapping.open(mappingname)

finally :

    mappings = [line.split("\t") for line in mapping]
    mappings = {description.strip().lower():wnid for wnid, description in mappings}

# Ask user for category of interest

wnid = None
while not wnid :

    cat_of_interest = input("Enter WordNet category of interest to restrict image download (see {} for all available categories): ".format(mappingname))

    try :
        wnid = mappings[cat_of_interest.lower()]
    except KeyError :
        print("Category not found in mappings.")

print("WordNetID found: {}".format(wnid))

# Retrieve and untar imagenet urls
if not os.path.isfile('fall11_ urls.txt') :

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

# Retrieve all imagenet urls for category of interest
with open('fall11_urls.txt') as urls :
    allurls = [line.split('\t') for line in urls]
    wnidurls = {id:url for id, url in allurls if wnid in id}
    print ("{} images found for selected WordNetID".format(len(wnidurls)))

# Retrieve all images
imgfolder = 'imagenet'
if not os.path.exists(imgfolder):
    os.makedirs(imgfolder)

for id, url in wnidurls :
    print("Downloading {}.jpg from {}".format(id, url))

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    with open(os.path.join(imgfolder, id, ".jpg"), "wb") as handle:
        for data in tqdm(response.iter_content(block_size), total = total_size // block_size, unit ='KB'):
            handle.write(data)

# Normalize all images
