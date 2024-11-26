# Monocular-VBFE

Source control for my final-year project! Once it's in usable quality I'll make it public.

Monocular Depth Estimation based force estimation, primarly used in surgical applications, during RAMIS operations.
It is based on a TCN method.
Inference utilizes ROS to publish the estimated force to the dVRK console.
The extract.py was used to gather frames from the ECM for training and testing.
The main.py was used for the training itself. The relevant Jupyter notebooks are located in the notebooks folder.