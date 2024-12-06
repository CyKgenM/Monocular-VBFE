# Monocular-VBFE

Source control for my final-year project! Once it's in usable quality I'll make it public.

**DOES NOT WORK YET**

Monocular Depth Estimation based force estimation, primarly used in surgical applications, during RAMIS operations.
It is based on a TCN method.
Inference utilizes ROS to publish the estimated force to the dVRK console.
The extract.py was used to gather frames from the ECM for training and testing.
The main.py was used for the training itself. The relevant Jupyter notebooks are located in the notebooks folder.



# Citations

## Pytorch implementation of PointNet and PointNet++
```
@misc{benny_yanx27pointnet_pointnet2_pytorch_2024,
	title = {yanx27/{Pointnet}\_Pointnet2\_pytorch},
	copyright = {MIT},
	url = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
	abstract = {PointNet and PointNet++ implemented by pytorch (pure python) and on ModelNet, ShapeNet and S3DIS.},
	urldate = {2024-10-16},
	author = {Benny},
	month = oct,
	year = {2024},
	note = {original-date: 2019-03-04T14:24:30Z},
	keywords = {classification, modelnet, point-cloud, pointcloud, pointnet, pointnet2, pytorch, s3dis, segmentation, shapenet, visualization},
}
```

## Metric3D for Monocular Depth Estimation
```
@misc{yin_yvanyinmetric3d_2024,
	title = {{YvanYin}/{Metric3D}},
	copyright = {BSD-2-Clause},
	url = {https://github.com/YvanYin/Metric3D},
	abstract = {The repo for "Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image" and "Metric3Dv2: A Versatile Monocular Geometric Foundation Model..."},
	urldate = {2024-10-29},
	author = {Yin, Wei},
	month = oct,
	year = {2024},
	note = {original-date: 2023-07-15T05:34:09Z},
	keywords = {3d-reconstruction, 3d-scenes, depth, depth-map, metric-depth-estimation, monocular-depth, monocular-depth-estimation, single-image-depth-prediction, zero-shot, zero-shot-transfer},
}
```