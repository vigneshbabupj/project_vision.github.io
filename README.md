
![pytorch][pytorch-shield]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<br />
<p align="center">
  <a href="https://github.com/vigneshbabupj/Project_Vision/blob/main/">
    <img src="documents/Computer_vision.png" alt="Logo" width="80" height="80">
  </a>

  <h1 align="center">Project Vision</h1>

  <p align="center">
    Phase 1 Capstone Project
    <br />
    <a href="https://theschoolof.ai"><strong>Extension Vision AI Program 5</strong></a>
    <br />
    <br />
    <a href="https://github.com/vigneshbabupj/Project_Vision">View Repo</a>
    ·
    <a href="mailto:vigneshbabupj@gmail.com">Vignesh Babu P J</a>
  </p>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#markdown">Markdown</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


### Problem Statement
The assignment is to create a network that can perform 3 tasks simultaneously:
  1. Predict the boots, PPE, hardhat, and mask if there is an image
  2. Predict the depth map of the image
  3. Predict the Planar Surfaces in the region

The strategy is to use pre-trained networks and use their outputs as the ground truth data:
  - [Midas](https://github.com/intel-isl/MiDaS) Depth model for generating depth map
  - [Planercnn](https://github.com/NVlabs/planercnn) network for identifying Plane surface
  - [Yolov3 ](https://github.com/theschoolofai/YoloV3) network for object detection



### Model

The Network is of Encoder-Decoder Architecture.

![vision](https://github.com/vigneshbabupj/Project_Vision/blob/main/documents/vision.png)

- Layers: 1056
- Parameters: 23.7m

### Dataset 

The data used for the training of the model is as below.

[Download Data](https://drive.google.com/file/d/1-FKOqy2sofWTBETl4e7177Cf4PxLfPAM/view?usp=sharing)

The high level steps taken to create the dataset is as below:
  1. Collect images from different website of people wearing hardhat, masks, PPE and boots.
  2. For object detection, use YoloV3 annotation tool to draw bounding box for the labels.
  3. Use MidasNet by Intel to generate the depth for the images
  4. Use Planercnn to generate plane segmentations of the images

A detailed explanation and code can be found in this [Repo](https://github.com/vigneshbabupj/workers_safety_equipment_data.git)

***Issues faced***
  - By default Planercnn gives the following outputs for each input image
    - Image.png
    - Plane_parameters.npy
    - Plane_masks.npy
    - Depth_ori.png
    - Depth_np.png
    - Segmentation.png
    - Segmentation_final.png
  - For our purpose we do not need the depth prediction of the Planercnn, therefore we can omit depth*.png,image.png is same the input image hence we can omit that also.so for our use case, plane_parameters.npy, plane_masks.npy and segmentation_final.png is only required
  - We frequently run outo Disk space as _*.npy_ files are heavy, so to handle it i replace both the *np.save()* of the .npy files with a single *np.savez_compressed()* line, This helps to save disk space as well as store numpy files
  - The output files are saved with index number rather than their actual names, this can be handled by replacing the _visualizebatchPair()_ parameter from the index number to the image filename in Visualise_utils.py


***Additional Data***

[Download link](https://drive.google.com/file/d/1-I4Gbj1Z1gCELTZ5amMq-8irQDhc6uTE/view?usp=sharing)
- Additional data for the training of Planercnn has been created
- A [Youtube Video](https://www.youtube.com/watch?v=mUtSU5u9AMM) of indoor surfaces is used to create images by generating frame every 0.5 second,the frames are then used to generate the Planercnn output.


### Model Development

In this section I will explain the steps taken to reach the final trainable model.


Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/vigneshbabupj/project_vision.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.

[pytorch-shield]: http://img.shields.io/badge/pytorch-1.7-red?style=for-the-badge&logo=PyTorch
[license-shield]: https://img.shields.io/apm/l/vim-mode?style=for-the-badge
[license-url]: https://github.com/vigneshbabupj/project_vision.github.io/blob/main/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/pjvb/
[product-screenshot]: images/screenshot.png
