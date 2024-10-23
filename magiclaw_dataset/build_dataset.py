import os
from pathlib import Path
import click
import subprocess
from format_converter import FormatConverter


@click.command(help="Building TensorFlow datasets.")
@click.option("-f", "--file", type=str, required=True, help="The ZIP file exported from MagiClaw app.")
@click.option("-s", "--split", type=click.Choice(['train', 'val']), default='train', help="The dataset split.")
@click.option("-t", "--type", type=click.Choice(['linear', 'nearest', 'nearest-up', 'zero', 'slinear', \
            'quadratic', 'cubic', 'previous', 'next']), default='nearest', help="The interpolation type.")
def main(file, split, type) -> None:
    # Unzip @file
    file = Path(os.path.expanduser(file)).absolute()
    zip_dir = file.parent.joinpath("data")
    subprocess.run([
        'unzip',
        '-d',
        str(zip_dir),
        str(file)
    ])
    # Rename and unzip episode files 
    episode_paths = list(zip_dir.glob("*.magiclaw"))
    episode_dir = str(Path.cwd().joinpath("raw_data"))
    for episode_path in episode_paths:
        zip_path = str(episode_path).replace("magiclaw", "zip")
        subprocess.run([
            'mv',
            str(episode_path),
            zip_path
        ])
        subprocess.run([
            'unzip',
            '-d',
            episode_dir,
            zip_path
        ])
    subprocess.run([
        'rm',
        '-r',
        str(zip_dir)
    ])
    # Convert format
    save_dir = str(Path.cwd().joinpath("data"))
    fc = FormatConverter(
        episode_dir=episode_dir,
        save_dir=save_dir,
        split=split,
        type=type
    )
    fc.run()
    subprocess.run([
        'rm',
        '-r',
        episode_dir
    ])
    # Build TensorFlow dataset
    ret = subprocess.run(['tfds', 'build'])
    if ret.returncode == 0:
        print(f"Find dataset at ~/tensorflow_datasets/magiclaw_dataset.")
        subprocess.run([
            'rm',
            '-r',
            save_dir
        ])
    else:
        raise RuntimeError("Building TensorFlow dataset failed!")


if __name__ == "__main__":
    main()
    