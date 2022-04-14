import requests
import os
import asyncio
from aiohttp import ClientSession
import aiofiles
import argparse
from os import listdir
from os.path import isfile, join
from datasets import load_dataset
import json
from selenium import webdriver
import shutil

headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
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


def parse_collection_selenium(collection_name, driver_path):
    driver = webdriver.Chrome(executable_path=driver_path)
    collection = []
    cursor = ""
    while cursor is not None:
        url = f"https://api.opensea.io/api/v1/assets?collection={collection_name}&format=json&limit=200&cursor={cursor}"
        driver.get(url)
        pre = driver.find_element_by_tag_name("pre").text
        result = json.loads(pre)
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
        try:
            async with aiofiles.open(path, 'wb') as f:
                await f.write(await resp.read())
        except Exception as ex:
            print(ex)
            return


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
    try:
        shutil.rmtree("images")
    except:
        pass
    if args.use_selenium:
        collection = parse_collection_selenium(args.collection_name, args.driver_path)
    else:
        collection = parse_collection(args.collection_name)

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
    for image in images:
        name = os.path.basename(image).replace(".png", "")
        for column_key in columns.keys():
            columns[column_key].append(nft_match[name][column_key])
    for column_key in columns.keys():
        dataset["train"] = dataset["train"].add_column(column_key, columns[column_key])

    dataset.push_to_hub(f"huggingnft/{args.collection_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection_name", type=str, default="cryptopunks", help="Name of OpenSea collection")
    parser.add_argument(
        "--use_selenium",
        action="store_true",
        help="Whether to use selenium Chrome driver for parsing.",
    )
    parser.add_argument("--driver_path", type=str, default="./chromedriver", help="Path to Chrome driver")

    args = parser.parse_args()
    main(args)
