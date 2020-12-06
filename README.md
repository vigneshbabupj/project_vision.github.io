
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
      <a href="#problem-statement">Problem Statement</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#model">Model</a></li>
    <li>
      <a href="#dataset">Dataset</a>
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


## Problem Statement
The assignment is to create a network that can perform 3 tasks simultaneously:
  1. Predict the boots, PPE, hardhat, and mask if there is an image
  2. Predict the depth map of the image
  3. Predict the Planar Surfaces in the region

The strategy is to use pre-trained networks and use their outputs as the ground truth data:
  - [Midas](https://github.com/intel-isl/MiDaS) Depth model for generating depth map
  - [Planercnn](https://github.com/NVlabs/planercnn) network for identifying Plane surface
  - [Yolov3 ](https://github.com/theschoolofai/YoloV3) network for object detection



## Model

The Network is of Encoder-Decoder Architecture.

![vision](https://github.com/vigneshbabupj/Project_Vision/blob/main/documents/vision.png)

- Layers: 1056
- Parameters: 23.7m

## Dataset 

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


## Model Development

In this section I will explain the steps taken to reach the final trainable model.
Significant amount of time was invested in the initial to read all the research papers of each model and get a understanding of their architecture, this would enable us to split their encoder from their decoder.

  1. **Step 1:** To define the high outline of the final model and then start to give definition for each of its components
      - The structure of the model defined is as below

    ```python

          class VisionNet(nn.Module):

            '''
              Network for detecting objects, generate depth map and identify plane surfaces
            '''

            def __init__(self,yolo_input,midas_input,planercnn_input):
              super(VisionNet, self).__init__()
              """
                Get required configuration for all the 3 models
              
              """
              self.yolo_params = yolo_input
              self.midas_params = midas_input
              self.planercnn_params = planercnn_input
              
              self.encoder = Define Encoder()

              self.plane_decoder = Define Plane decoder(self.planercnn_params)

              self.depth_decoder = Define Depth decoder(self.midas_params)
              
              self.bbox_decoder =  Define Yolo decoder(self.yolo_params)
              

            def forward(self,x):

              x = self.encoder(x)

              plane_out = self.plane_decoder(x)

              depth_out = self.depth_decoder(x)
              
              bbox_out = self.bbox_decoder(x)

              return  plane_out, bbox_out, depth_out

    ```

  2. **Step 2:** Define Encoder Block
      - The 3 different encoder block in each of the networks:
        - MidasNet - ResNext101_32x8d_wsl
        - Planercnn - ResNet101
        - Yolov3 - Darknet-53
      - My initial thoughts was to use Darknet as the base encoder, as the similar accuracy as ResNet and it is almost 2x faster based on performance on ImageNet dataset, but the downside of it is compartively complex to separate only the config of Darknet from Yolov3 config and then run the same code blocks from Yolov3 from model definition and forward method, This could mean i have to recreate those code blocks with changes so that only Darknet encoder is proccesed.
      Hence, as the enocder-decoder of Yolov3 is tighly coupled in code i decided against using it.
      - On other two options, I had tried both of them separately as the encoder blocks, based on the benchmarks ResNext-101 has perfomed better than Resnet-101 and ResNext WSL is maintained by facebook and are pre-trained in weakly-supervised fashion on 940 million public images with 1.5K hashtags matching with 1000 ImageNet1K synsets, followed by fine-tuning on ImageNet1K dataset, So the below ResNext block is used as enoder with the pretrained weights

```python
     resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
```

      - The encoder is defined with 4 pretrained layers


    ```python
      def _make_resnet_backbone(resnet):
          pretrained = nn.Module()
          pretrained.layer1 = nn.Sequential(
              resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
          )

          pretrained.layer2 = resnet.layer2
          pretrained.layer3 = resnet.layer3
          pretrained.layer4 = resnet.layer4

          return pretrained

    ```


  3. **Step 3:** Define Depth decoder block
    - This was pretty direct reference form the midasnet block excluding only the pretrained encoder


    ```python
    class MidasNet_decoder(nn.Module):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256,non_negative=True):
        super(MidasNet_decoder, self).__init__()
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.

        """
        self.scratch = _make_encoder_scratch(features)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )


    def forward(self, *xs):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """

        layer_1, layer_2, layer_3, layer_4 =  [xs[0][i] for i in range(4)]


        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        #print('layer_4_rn',layer_4_rn[0][0])

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)
        #print('out',out.size(),out)
        #print('out squeeze',torch.squeeze(out, dim=1).size(),torch.squeeze(out, dim=1))

        final_out = torch.squeeze(out, dim=1)

        return final_out
        ```


  4. **Step 4:** Define Object detection decoder block
    - yolov3 custom cfg file had to be changed to omit the encoder part of the network and retain only the decoder part
    - Darknet-53 is feature extrator that extends upto the 75th layers in the yolo network, also a key point to note is there are 3 skip connection from the Darknet encoder to decoder for object detection
    - A print of the layer name with the sizes give understanding of the each layer along with their output shape -[file](https://github.com/vigneshbabupj/Project_Vision/blob/main/bbox_decoder/Actual_layers_sizes)
    - To pass the output from the encoder layers to the corresponding layer in Yolo, a 1x1 convolution was used
      - Encoder layer 2 output --> Yolo 36th layer
      - Encoder layer 3 output --> Yolo 61st layer
      - Encoder layer 4 output --> Yolo 75th layer
        ```python
        init:
        self.conv1 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=(1, 1), padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), padding=0, bias=False)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), padding=0, bias=False)

        forward:
        Yolo_75 = self.conv1(layer_4)
        Yolo_61 = self.conv2(layer_3)
        Yolo_36 = self.conv3(layer_2)

        ```

    - The Darknet layer configuration post the custom changes can ve viewed from this [updated file](https://github.com/vigneshbabupj/Project_Vision/blob/main/bbox_decoder/yolo_layer_size_vignesh)

  5. **Step 5:** Define Plane segmentation decoder block

    - Planercnn is built of MaskRcnn network which consists of resnet101 as the backbone for feature extractor and then it is followed by FPN,RPN and rest of the layers for detections
    - The first 5 layers(C1 - C5) of FPN are directly from the resnet101 block, which i changed to connect to our layers from the custom encoder block (note: C1 & C2 together form the layer 1 of our ResNext101 Encoder)
        - Encoder layer 1 output --> FPN C1 layer
        - Encoder layer 2 output --> FPN C2 layer
        - Encoder layer 3 output --> FPN C3 layer
        - Encoder layer 4 output --> FPN C4 layer
    - Key concept in Planercnn integration is that the default nms and ROI is coplied on the torch verions 0.4, which is incompatible with other decoder modules which use latest torch version, to handle this the default nms was replaced with the nms from torchvision and the ROI Align buit on pytorch([link](https://github.com/longcw/RoIAlign.pytorch)) was used

  6. **Step 6:** The Trainable model
    - The Final trainable version of the model is as below


    ```python
      class VisionNet(nn.Module):

        '''
          Network for detecting objects, generate depth map and identify plane surfaces
        '''

        def __init__(self,yolo_cfg,midas_cfg,planercnn_cfg,path=None):
          super(VisionNet, self).__init__()
          """
            Get required configuration for all the 3 models
          
          """
          self.yolo_params = yolo_cfg
          self.midas_params = midas_cfg
          self.planercnn_params = planercnn_cfg
          self.path = path

          use_pretrained = False if path is None else True

          print('use_pretrained',use_pretrained)
          print('path',path)
          
          self.encoder = _make_resnet_encoder(use_pretrained)

          self.plane_decoder = MaskRCNN(self.planercnn_params,self.encoder)

          self.depth_decoder = MidasNet_decoder(path)

          

          self.bbox_decoder =  Darknet(self.yolo_params)
          

          self.conv1 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=(1, 1), padding=0, bias=False)
          self.conv2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), padding=0, bias=False)
          self.conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), padding=0, bias=False)
          
          self.info(False)

        def forward(self,yolo_ip,midas_ip,plane_ip):

          x = yolo_ip
          #x = midas_ip

          # Encoder blocks
          layer_1 = self.encoder.layer1(x)
          layer_2 = self.encoder.layer2(layer_1)
          layer_3 = self.encoder.layer3(layer_2)
          layer_4 = self.encoder.layer4(layer_3)

          Yolo_75 = self.conv1(layer_4)
          Yolo_61 = self.conv2(layer_3)
          Yolo_36 = self.conv3(layer_2)

          if plane_ip is not None:
            plane_ip['input'][0] = yolo_ip
            # PlaneRCNN decoder
            plane_out = self.plane_decoder.forward(plane_ip,[layer_1, layer_2, layer_3, layer_4])
          else:
            plane_out = None

          if midas_ip is not None:
            # MiDaS depth decoder
            depth_out = self.depth_decoder([layer_1, layer_2, layer_3, layer_4])
          else:
            depth_out = None

          #YOLOv3 bbox decoder
          if not self.training:
            inf_out, train_out = self.bbox_decoder(Yolo_75,Yolo_61,Yolo_36)
            bbox_out=[inf_out, train_out]
          else:
            bbox_out = self.bbox_decoder(Yolo_75,Yolo_61,Yolo_36)

          return  plane_out, bbox_out, depth_out

        def info(self, verbose=False):
          torch_utils.model_info(self, verbose)

    ```



## Set up Model Training
  1. **Step 1:** Define input parameters for training
    - As each of the 3 network have their own multiple default parameters for decoder configurations and data preproccessing, I combined the Arg parser of all the 3 decoders into single file [options.py](https://github.com/vigneshbabupj/Project_Vision/blob/main/options.py)
    - This ensures we able to pass the required input parameters including weights path for each of the decoders separately.
  2. **Step 2:** 




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
