# ARIC: Automatic Real-to-Sim-to-Real via Iterative Complement

This repository is the official implementation of "Automatic Real-to-Sim-to-Real System through Iterative Interactions for Robust Robot Manipulation Policy Learning with Unseen Objects".

## Implementation Reference

1. FastSAM (./References/FastSAM)
> In this repository, we use the official implementation of "FastSAM" (link: https://github.com/CASIA-IVA-Lab/FastSAM).

2. SAM2 (./References/SAM2)
> In this repository, we use the official implementation of "SAM2" (link: https://github.com/facebookresearch/sam2).

3. Gaussian Surfels (./References/Gaussian_Surfels)
> In this repository, we use the official implementation of "Gaussian Surfels" (link: https://github.com/turandai/gaussian_surfels).

4. Lepard (./References/Lepard)
> In this repository, we use the official implementation of "Lepard" (link: https://github.com/rabbityl/lepard).

5. IDR (./References/IDR)
> In this repository, we use the official implementation of the "IDR" library (link: https://github.com/lioryariv/idr/tree/main).

6. rl_games (in-your-virtualenv-packages)
> We modify and use the implementation of the "rl_games" library (link: https://github.com/Denys88/rl_games).

> You can download the modified rl_games package from "https://drive.google.com/file/d/16yJqIANft8j-fK4QsjkFTv8Uxp-aP8Ib/view?usp=drive_link".

## Requirements

We use python3.10 & Ubuntu 20.04 for the ARIC implementation.

Please refer to the original github page for installation instructions for each reference.

## ARIC: Real-to-Sim Phase

You can find the Real-to-Sim process of ARIC in "Object_Reconstruction.ipynb".

> An example of observations of the real workspace can be downloaded from "https://drive.google.com/file/d/1kFubcwKMoUMQo5h8NZbI6gZ1pPv78EyH/view?usp=drive_link".


## ARIC: Sim-to-Real Phase

You can find the Sim-to-Real process of ARIC in "Task1.Push-to-Goal.ipynb".

> Observation examples for the Push-to-Goal task can be downloaded from "https://drive.google.com/file/d/1ZH69a-inL0y_ZrNdx-B82AHH5gLWhP8q/view?usp=drive_link".

> Pre-trained checkpoint for the Push-to-Goal task can be downloaded from "https://drive.google.com/file/d/1Qfuq3HkCQcXpqYYQJQMZbqyXyeKCn1hf/view?usp=drive_link".

