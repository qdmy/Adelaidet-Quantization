cd /root/
mkdir Datasets
rsync -av --progress /userhome/liujing/coco2014.tar Datasets/
cd Datasets
tar -xvf coco2014.tar
cd /userhome/liujing/Adelaidet-Quantization