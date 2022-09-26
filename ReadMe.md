**Fall Detection**

Automated fall detection is for assisting elderly in daily life. It captures fall activity and notifies about that.

**How it works**

According to the "Privacy Preserving Automatic Fall Detection for Elderly Using RGBD Cameras" paper by Chenyang Zhang, Yingli Tian, and Elizabeth Capezuti, ‘fall’ causes much more deformation of the joint structures than other daily activities.  To control the joint structures, we obtain the key points of a human skeleton. The model we used to get the key points is OpenPifPaf 0.13.4. OpenPifPaf returns seventeen key points. Having obtained the key points, we produce vectors defined in advance, which are most likely to be dramatically changed during a fall. OpenPifPaf returns negative coordinates for the key points that were not detected. We take nans instead of negative coordinates of the key points and then handle them.
In addition to the provided key points, we add extra ones that are probably important for fall detection. It includes, the midpoint of the shoulders, the midpoint of the hips, two random points with equal x coordinate that will produce a vertical vector and mean of the head points.
![Untitled](https://user-images.githubusercontent.com/65034169/190630310-2152a833-9384-43d7-bf8e-45ff065a7f55.png)

We calculate eight angles based on the specified pairs of the chosen vectors. Those eight angles are calculated by the norms of the chosen vectors and dot product of the vector pairs.
![angleFormula](https://user-images.githubusercontent.com/65034169/190630405-3c107203-25d6-4635-8f1d-94ac89af89c8.PNG)

Based on the angles several features are being calculated:
1) costMean (The mean of eight angles)



![MeanCostFormula (1)](https://user-images.githubusercontent.com/65034169/190630585-d569cd42-1010-4fea-8f01-16a5c0644bfc.PNG)



![MeanCost](https://user-images.githubusercontent.com/65034169/190630669-eda6cff9-5c6f-4d9b-85af-144c02caa5f7.PNG)


2) divisonCost (Each angle is divided on by its previous frame’s same angle, and then all resulted angles are summed up)



![DivisionCostFormula](https://user-images.githubusercontent.com/65034169/190630759-fd6e510c-bfd8-4c9b-bd1e-56a792054a7f.PNG)




![DivisionCost](https://user-images.githubusercontent.com/65034169/190630781-6f829560-43e8-4b6a-b692-2492a7d7a62d.PNG)



3) differenceMean (The absolute value of the difference between each angle of previous and current angles are taken, then mean of eight differences are calculated and multiplied by the current fps. It represents a number close to the derivative)



![DifferenceMeanCostFormula](https://user-images.githubusercontent.com/65034169/190630859-b6ee4b7c-9080-4292-88a3-5ab14c198c8f.PNG)



![DifferenceMeanCost](https://user-images.githubusercontent.com/65034169/190630888-53e694ca-4512-4b74-9e4f-02f09859be4c.PNG)


4) differenceSum (The absolute value of the difference between each angle of previous and current angles are taken, then sum of eight differences are calculated)

![DifferneceMeanSum](https://user-images.githubusercontent.com/65034169/190631136-14057044-0619-49d9-b862-a89a8c3e4db7.PNG)

![DifferenceSumCost](https://user-images.githubusercontent.com/65034169/190631154-23f76225-bbc5-4688-b540-2221acbde84e.PNG)

5) meanDifference (The means of eight angles of previous and current frames are calculated and absolute value of difference of those means is taken)



![MeanDifferenceCostFormula](https://user-images.githubusercontent.com/65034169/190631237-c6fd4cc2-e400-4141-b744-f1f460db7ed0.PNG)



![MeanDifferenceCost](https://user-images.githubusercontent.com/65034169/190631262-29f72275-da5b-49d0-9cd6-5e01714879ac.PNG)


Where <t> superscript is the previous frame, <t+1> superscript is the current frame. Alpha is the angle calculated between two vectors. We are using 8 pairs of vectors, so there are 8 angles calculated between two frames, so n=8. In the case, when more than 6 angles are not found for some frame, the frame is skipped. Moreover, if the cost is NaN, we compute nothing and assign the current cost to the previous one. If the missing one is the first cost, we take a default number.
We use weights to give respective importance to each angle calculated, since not all angles are equally important for fall prediction. Weights can be optimized using labeled train data that will produce the weights that provide right importance to the angles.

Since falling is a sequential activity and it may last a number of frames. For that reason, we keep the costs for the last 6 frames (duration of 1 second) and multiply with different weights and calculate the mean.
![cost](https://user-images.githubusercontent.com/65034169/190631666-c20903e6-1c30-482c-b49a-ff1e9c8fb1d5.PNG)
  
![CostWeighted](https://user-images.githubusercontent.com/65034169/190631717-14b4c21d-9d3a-45ba-ab11-a3798d78a899.PNG)


After getting one number from each cache as the main cost of the current frame, threshold is applied to the costs, and if they extend the threshold then a notification shows fall activity. Moreover, we remove too big and too small costs by bounding the result in [clip\_from, clip\_to] interval using the numpy clip function, which is the acceptable interval where the cost value can be.
Optimal threshold and bounding interval can be found using grid search on labeled training data using grid search.
 

  
  
![Demonstration](https://user-images.githubusercontent.com/65034169/190631807-5f454de9-8e22-4ba1-9b74-f96fcb44ed78.gif)



If you want to test it, you need to run the following code in your cmd:py FallDetection.py -video video_name -m method --save/--no--save


  
