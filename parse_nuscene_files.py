import os


def parse_files(path):
    files = os.listdir(path)
    return list(map(lambda x: x.strip().rsplit('.', 1)[0], files))


def main(path_to_data, path_to_write):
    train_ids = parse_files(os.path.join(path_to_data, "training", "image_2"))
    with open(os.path.join(path_to_write, "train.txt"), "w+") as f:
        print("\n".join(train_ids), file=f)

    with open(os.path.join(path_to_write, "val.txt"), "w+") as f:
        print("\n".join(train_ids), file=f)        

    test_ids = parse_files(os.path.join(path_to_data, "testing", "image_2"))
    with open(os.path.join(path_to_write, "test.txt"), "w+") as f:
        print("\n".join(test_ids), file=f)    


if __name__ == "__main__":
    path_to_data = "nuscenes_kitti"
    path_to_write = "dataset/ImageSetsNuscenes"
    main(path_to_data, path_to_write)