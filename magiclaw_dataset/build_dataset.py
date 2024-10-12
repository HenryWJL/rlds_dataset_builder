import click
import subprocess
from format_converter import FormatConverter


@click.command(help="Training policy on the datasets.")
@click.option("-ed", "--episode_dir", type=str, required=True, help="The directory used for loading episodes.")
@click.option("-s", "--split", type=click.Choice(['train', 'val']), default='train', help="The dataset split.")
@click.option("-it", "--interp_type", type=click.Choice(['linear', 'nearest', 'nearest-up', 'zero', 'slinear', \
            'quadratic', 'cubic', 'previous', 'next']), default='nearest', help="The interpolation type.")
def main(episode_dir, split, interp_type) -> None:
    fc = FormatConverter(
        episode_dir=episode_dir,
        split=split,
        interp_type=interp_type
    )
    fc.run()
    ret = subprocess.run(['tfds', 'build'])
    if ret.returncode == 0:
        print(f"Find dataset at ~/tensorflow_datasets/magiclaw_dataset.")
    else:
        raise RuntimeError("Building TensorFlow dataset failed!")


if __name__ == "__main__":
    main()