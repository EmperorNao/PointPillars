import os
import argparse


def parse_files(path):
    files = os.listdir(path)
    return list(map(lambda x: x.strip().rsplit('.', 1)[0], files))


def main(path_to_data, path_to_write):
    train_ids = parse_files(os.path.join(path_to_data, "train", "image_2"))
    with open(os.path.join(path_to_write, "train.txt"), "w+") as f:
        print("\n".join(train_ids), file=f)

    val_ids = parse_files(os.path.join(path_to_data, "val", "image_2"))
    with open(os.path.join(path_to_write, "val.txt"), "w+") as f:
        print("\n".join(val_ids), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NuScenes parser')
    parser.add_argument('--path_to_data', default='nuscenes_kitti')
    parser.add_argument('--path_to_write', default='dataset/ImageSetsNuscenes')
    args = parser.parse_args()

    main(args.path_to_data, args.path_to_write)