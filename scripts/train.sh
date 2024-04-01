# Copyright (c) Meta Platforms, Inc. and affiliates.

# stunnel (obstacle + congestion)
python train.py experiment=stunnel prob.sigma=0.5,1,2 -m # bm
python train.py experiment=stunnel prob.sigma=0.5,1,2 csoc.IW=true -m # bm-IW
python train.py experiment=stunnel_eam prob.sigma=0.5,1,2 -m # eam
python train.py experiment=stunnel_eam prob.sigma=0.5,1,2 csoc.IW=true -m # eam-IW

# vneck (obstacle + entropy)
python train.py experiment=vneck prob.sigma=1,2 -m

# gmm (obstacle + entropy)
python train.py experiment=gmm prob.sigma=1,2 -m

# lidar
python train.py experiment=lidar -m

# opinion_2d
python train.py experiment=opinion_2d -m

# opinion
python train.py experiment=opinion -m
