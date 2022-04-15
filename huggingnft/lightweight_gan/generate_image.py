import argparse
from huggingnft.lightweight_gan.train import timestamped_filename
from huggingnft.lightweight_gan.lightweight_gan import load_lightweight_model


def main(args):
    model = load_lightweight_model(f"huggingnft/{args.collection_name}")
    image_saved_path, generated_image = model.generate_app(
        num=timestamped_filename(),
        nrow=args.nrows,
        checkpoint=-1,
        types=args.generation_type
    )
    print(f"Resulting image saved here: {image_saved_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection_name", type=str, default="cryptopunks", help="Name of OpenSea collection")
    parser.add_argument(
        "--nrows",
        type=int,
        default=8,
        help="Number of rows in the grid",
    )
    parser.add_argument(
        "--generation_type",
        type=str,
        default="default",
        choices=["default", "ema"],
        help="Generation type: default or ema",
    )
    args = parser.parse_args()
    main(args)
