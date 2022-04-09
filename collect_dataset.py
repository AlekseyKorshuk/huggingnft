import requests
import os
import asyncio
from aiohttp import ClientSession
import aiofiles
import argparse
from os import listdir
from os.path import isfile, join
from datasets import load_dataset

headers = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.4 Safari/605.1.15"
}


def parse_collection(collection_name):
    collection = []
    cursor = ""
    while cursor is not None:
        url = f"https://api.opensea.io/api/v1/assets?collection={collection_name}&format=json&limit=200&cursor={cursor}"
        response = requests.request("GET", url, headers=headers)
        result = response.json()
        collection = collection + result["assets"]
        cursor = result["next"]
    return collection


async def make_request(session, asset):
    try:
        resp = await session.request(method="GET", url=asset["image_url"])
    except Exception as ex:
        print(ex)
        return

    if resp.status == 200:
        path = f'images/{asset["image_url"].replace("https://lh3.googleusercontent.com/", "")}.png'

        async with aiofiles.open(path, 'wb') as f:
            await f.write(await resp.read())


async def bulk_request(collection):
    """Make requests concurrently"""
    async with ClientSession() as session:
        tasks = []
        for asset in collection:
            tasks.append(
                make_request(session, asset)
            )
        await asyncio.gather(*tasks)


def download_images(collection):
    os.makedirs("images", exist_ok=True)
    asyncio.run(bulk_request(collection))


def main(args):
    collection = parse_collection(args.collection_name)

    collection_data = {
        "banner_image_url": collection[0]["collection"]["banner_image_url"],
        "description": collection[0]["collection"]["description"],
        "featured_image_url": collection[0]["collection"]["featured_image_url"],
        "image_url": collection[0]["collection"]["image_url"],
        "name": collection[0]["collection"]["name"],
    }

    download_images(collection)

    images = [f"images/{f}" for f in listdir("images") if isfile(join("images", f))]
    dataset = load_dataset("imagefolder", data_files={"train": images})
    dataset["train"] = dataset["train"].remove_columns("label")
    nft_match = {}
    for asset in collection:
        name = f'{asset["image_url"].replace("https://lh3.googleusercontent.com/", "")}'
        nft_match[name] = {
            "id": asset["id"],
            "token_metadata": asset["token_metadata"],
            "image_original_url": asset["image_original_url"],
        }
    columns = {
        "id": [],
        "token_metadata": [],
        "image_original_url": [],
    }
    for image in dataset["train"]["image"]:
        name = os.path.basename(image.filename).replace(".png", "")
        for column_key in columns.keys():
            columns[column_key].append(nft_match[name][column_key])
    for column_key in columns.keys():
        dataset["train"] = dataset["train"].add_column(column_key, columns[column_key])

    dataset.push_to_hub(f"huggingnft/{args.collection_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection_name", type=str, help="Name of OpenSea collection")
    args = parser.parse_args()
    main(args)
