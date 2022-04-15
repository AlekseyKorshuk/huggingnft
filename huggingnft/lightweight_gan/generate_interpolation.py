import argparse
from huggingnft.lightweight_gan.train import timestamped_filename
from huggingnft.lightweight_gan.lightweight_gan import load_lightweight_model


def main(args):
    model = load_lightweight_model(f"huggingnft/{args.collection_name}")
    gif_saved_path = model.generate_interpolation(
        num=timestamped_filename(),
        num_image_tiles=args.nrows,
        num_steps=args.num_steps,
        save_frames=False
    )
    print(f"Resulting gif saved here: {gif_saved_path}")


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
        "--num_steps",
        type=int,
        default=100,
        help="Number of steps to generate",
    )
    args = parser.parse_args()
    main(args)
