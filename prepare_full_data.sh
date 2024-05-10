wget https://www.nuscenes.org/data/v1.0-mini.tgz
mkdir nuscenes
tar -xvf v1.0-mini.tgz -C nuscenes
sudo mkdir /data
sudpo mkdir /data/set
ln -s $(pwd)/nuscenes /data/set/
mkdir nuscenes_kitti
python3 -m nuscenes.scripts.export_kitti nuscenes_gt_to_kitti --nusc_kitti_dir nuscenes_kitti
mkdir dataset/ImageSetsNuscenes
python3 parse_nuscene_files.py
