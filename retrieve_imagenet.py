# Author: r.dotsch@anchormen.nl

from tqdm import tqdm
import requests
import tarfile
import os
import codecs
import pandas as pd

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

    cat_of_interest = input("Enter WordNet category of interest to restrict image download (see http://image-net.org/search): ")

    try :
        wnid = mappings[cat_of_interest.lower()]
    except KeyError :
        print("Category not found in mappings.")

print("WordNetID found: {}".format(wnid))

# Retrieve and untar imagenet urls
if not os.path.isfile('fall11_urls.txt') :

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
print("Looking for all imagenet urls that belong to WordNetID...")
urls = pd.read_csv('fall11_urls.txt', sep='\t',  warn_bad_lines=False, error_bad_lines=False, encoding="ISO-8859-1")
urls.columns = ['id', 'url'] # Dirty hack, losing first line because of this
wnidurls = urls[urls['id'].str.contains(wnid)]

print ("{} images found for selected WordNetID".format(len(wnidurls)))

# Retrieve all images

imgfolder = 'imagenet'
if not os.path.exists(imgfolder):
    os.makedirs(imgfolder)

print("Downloading images...")

for id, url in tqdm(wnidurls.itertuples(index=False), total=len(wnidurls), unit='image') :
    target = os.path.join(imgfolder, "{}.jpg".format(id))
    if not os.path.isfile(target) :
        #print("Downloading {}.jpg from {}".format(id, url))

        try :
            response = requests.get(url, stream=True, timeout=2)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            with open(target, "wb") as handle:
                for data in tqdm(response.iter_content(block_size), total = total_size // block_size, unit ='KB'):
                    handle.write(data)
        except requests.exceptions.ConnectionError:
            pass

        except requests.exceptions.ReadTimeout:
            pass

        except requests.exceptions.TooManyRedirects:
            pass

# Clean up bad images

# Normalize all images


print("Done.")
