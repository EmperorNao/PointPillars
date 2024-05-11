wget https://www.nuscenes.org/data/v1.0-mini.tgz
mkdir nuscenes
tar -xvf v1.0-mini.tgz -C nuscenes
sudo mkdir /data
sudo mkdir /data/sets
ln -s $(pwd)/nuscenes /data/sets/
mkdir nuscenes_kitti
python3 -m nuscenes.scripts.export_kitti nuscenes_gt_to_kitti --nusc_kitti_dir nuscenes_kitti
cp -r nuscenes_kitti/mini_train nuscenes_kitti/training
cp -r nuscenes_kitti/mini_train nuscenes_kitti/testing
mkdir dataset/ImageSetsNuscenes
python3 parse_nuscene_files.py
