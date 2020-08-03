

```shell
cd /home/mydir/pyspace/research-charnet
mkdir weights
ln -s /home/mydir/pretrained_model/icdar2015_hourglass88.pth weights/icdar2015_hourglass88.pth

python tools/test_net.py configs/icdar2015_hourglass88.yaml /home/mydir/dataset/xf_ocr/test ./result
```

