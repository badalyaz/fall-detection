[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

# Fall Detection
![Demonstration](https://user-images.githubusercontent.com/65034169/190631807-5f454de9-8e22-4ba1-9b74-f96fcb44ed78.gif)

Automated Fall Detection is an essential tool for assisting the elderly in their daily lives. By capturing fall activity, it enables timely notifications to relevant individuals or caregivers. This GitHub repository contains the necessary code and resources for implementing an automated fall detection system.

## How it works
**Description**

The fall detection system in this repository utilizes a method described in the research paper titled "Privacy Preserving Automatic Fall Detection for Elderly Using RGBD Cameras." The system focuses on analyzing the deformation of joint structures during falls, which differs significantly from regular daily activities. To achieve this, key points of the human skeleton are extracted using the OpenPifPaf model. By tracking these key points and comparing them with pre-defined vectors, the system detects and identifies falls accurately.

![Untitled](https://github.com/openpifpaf/openpifpaf/raw/main/docs/coco/000000081988.jpg.predictions.jpeg) by [openpifpaf](https://github.com/openpifpaf/openpifpaf)

**Features**

The fall detection algorithm in this repository utilizes a set of calculations to determine the likelihood of a fall event. The process involves calculating eight angles based on selected vectors, which are obtained from key points of the human skeleton. The angles are calculated in the following way:
$$\vec{\theta} = \arccos\left(\frac{{\vec{u} \cdot \vec{v}}}{{\|\vec{u}\| \|\vec{v}\|}}\right)$$

These angles are then used to derive several features that contribute to fall detection:

![Features](/assets/images/features.png)
1. **costMean:** This feature represents the average of the eight calculated angles.
2. **divisonCost:** Each angle is divided by its corresponding angle from the previous frame, and the resulting angles are summed together.
3. **differenceMean:** The absolute difference between each angle of the previous and current frames is taken, and the mean of these differences is calculated. This value is then multiplied by the current frames per second (fps), providing an approximation of the derivative.
4. **differenceSum:** Similar to differenceMean, this feature calculates the sum of the absolute differences between each angle of the previous and current frames.
5. **meanDifference:** The means of the eight angles from the previous and current frames are calculated, and the absolute difference between these means is determined.

By analyzing these features, the fall detection system can accurately identify and classify fall events.

*Note:* When analyzing falls, we focus on the last six frames (equivalent to one second) as falling is a sequential activity. The cost is determined by calculating the weighted mean of these frames.

$$C = \frac{1}{6} \sum_{i=1}^{6} c_i \cdot w_i$$

To test, run the following code:

```
python3 fall_detection.py --video video_name -m method_name --save/--no--save
```

## Citation
```
@article{
  title={Privacy Preserving Automatic Fall Detection for Elderly Using RGBD Cameras},
  author={Zhang, C., Tian, Y., Capezuti, E.},
  year={2012},
  doi={https://doi.org/10.1007/978-3-642-31522-0_95},
}
```
