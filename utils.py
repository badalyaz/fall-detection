#!/usr/bin/env python
# coding: utf-8

#Importing the necessary libraries

import cv2
import numpy as np
import openpifpaf
import import_ipynb
import matplotlib.pyplot as plt
import time



class KeyPoints():
    """
    class Keypoints. 
    Used to run the OpenPifPaf model and find the keypoints of an image.
    function(model) - Loads the model
    function(detectPoints) - Finds the keypoints of an image

    """
    def __init__(self):
        
        self.predictor = None
        
    def model(self, checkpoint = 'shufflenetv2k16'):    #Loads the model with provided checkpoint, which specifies the model's architecture complexity
              
        self.predictor = openpifpaf.Predictor(checkpoint = checkpoint)
        
    def detectPoints(self, frame):  #Detects the keypoints of an image
        
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   #Converts BGR image to RGB image
        
        predictions, gt_anns, meta = self.predictor.numpy_image(frameRGB)   #Finds the keypoints of the image
        
        if predictions == []:   #If no keypoints found, return an empty list
            
            predict = []
            
        else:
            
            predict = predictions[0].data[:, :2]    #If keypoints found, remove the probability column
            
        return predict     #Return the predicted points 


class FeatureExtractor():
    """
    class FeatureExtractor.
    Used to extract features from generated keypoints.
    """
    def __init__(self):
        
        self.torso_up = np.array([[5,6]])   #The slice used for generating the midpoint of the shoulders
        
        self.torso_down = np.array([[11,12]])   #The slice used for generating the midpoint of the hips
        
        self.vector_indices = np.array([[19, 17], [19,18], [6,12], [5,11], [6,8], [5,7], [12,14], [11, 13], [11,12], [13,15], [14,16], [20,21]])    #Vectors to be considered for calculating angles
        
        self.pair_indices = np.array([[4,2], [5,3], [6,10], [7,9], [8,6], [8,7], [0, 11], [1,11]])  #The pairs of vectors for angle computation
        
        self.vertical_coordinates = np.array([[1,1], [1,100]])  #A vertical vector for comparing with other vectors
        
        self.angle_weights = np.ones((8,1)) #Weights for angles
        
        self.cache_weights = np.ones((1,6)) #Weights for the cache
        
        self.keypoints = KeyPoints()    #Initialize the keypoints class
        
        self.keypoints.model()  #Call the model method of the keypoints class to load the openpifpaf model
                
        self.fps = 6    #Number of frames to consider in every second
        
        self.threshold = 10    #The threshold for fall detection
       
    def angleCalculation(self, vectors): 
        """
        Function angleCalculation.
        Used to calculate the angles between given pairs of vectors
        Takes as input the list of vector pairs, which represent two vectors with two coordinates each
        Returns the list of angles between them

        """
        difference = np.subtract(vectors[:,:,0], vectors[:,:,1])    #Subtracts the coordinates to obtain the vectors
        
        dot = (difference[:,0,:] * difference[:,1,:]).sum(axis=-1)  #Calculates the dot product between the pairs of vectors 
        
        norm = np.prod(np.linalg.norm(difference[:,:,:], axis = 2), axis=-1)    #Calculates the norm of the vectors and multiplies them, same as |a|*|b|
        
        cos_angle = np.divide(dot, norm)    #cos(x) = dot(a,b)/|a|*|b|
        
        angle = np.arccos(cos_angle)*180/np.pi    #Take arccos of the result to get the angle
        
        angle = angle.reshape(-1,1)    #Correct the shape of the output
        
        return angle

    def collectData(self, keypoints):
        """
        Function collectData.
        Calls handleMissingValues and addExtraPoints functions
        Used for handling negative predictions and adding extra points to the keypoints
        Takes as input the list of keypoints
        Returns the list of handled keypoints and added extra points

        """
        
        keypoints = self.handleMissingValues(keypoints)
        
        keypoints = self.addExtraPoints(keypoints)
        
        return keypoints
    
    def differenceMean(self, vector1_angles, vector2_angles):

        """
        Function differenceMean
        Used for calculating the feature using differenceMean method
        Takes as input previous frame angles and current frame angles
        Returns a scalar (the cost)

        """
        
        angle_difference = np.abs(vector1_angles - vector2_angles)  #Absolute difference of previous frame's angles and current frame's angles
                        
        return np.nanmean(angle_difference) * self.fps   #Returns the mean of the difference multiplied by fps
    
    def meanDifference(self, vector1_angles, vector2_angles):

        """
        Function meanDifference
        Used for calculating the feature using meanDifference method
        Takes as input previous frame angles and current frame angles
        Returns a scalar (the cost)

        """
        
        angle_difference = np.abs(np.nanmean(vector1_angles) - np.nanmean(vector2_angles))  #Absolute difference of means of previous and current angle lists
                        
        return angle_difference

    def differenceSum(self, vector1_angles, vector2_angles):

        """
        Function differenceSum
        Used for calculating the feature using differenceSum method
        Takes as input previous frame angles and current frame angles
        Returns a scalar (the cost)

        """
        
        angle_difference = np.abs(vector1_angles - vector2_angles)  #Absolute difference of previous and current frame angles
                
        return np.nansum(angle_difference)  #Returns the sum of angle differences
    
    def costMean(self, vector_angles):

        """
        Function costMean
        Used for calculating the feature using costMean method
        Takes as input the current frame angles
        Returns a scalar (the cost)
        """
        
        return np.nanmean(vector_angles)    #Return the mean of the angles for the frame
    
    def divisionCost(self, vector1_angles, vector2_angles):
        """
        Function divisionCost
        Used for calculating the feature using divisionCost method
        Takes as input the previous and current frames' angles
        Returns a scalar (the cost)
        """
        
        vector1_angles = np.where(vector1_angles == 0, np.nan, vector1_angles)    #If the angle is 0, replace it with nan to avoid division by zero
        
        angle_division = np.divide(vector2_angles, vector1_angles + 1e-6)    #Divide the current frame angles with previous frame angles
                
        return np.nansum(angle_division)    #Sum the result
    
    def handleMissingValues(self, keypoints):

        """
        Function handleMissingValues
        Used for replacing negative predictions with NaNs
        Takes as input the list of the keypoints for the current frame
        Returns corrected list of keypoints with NaNs instead of negative values
        """
        if keypoints != []:
        
            keypoints = np.where(keypoints < 0, np.nan, keypoints)    #Where the points is negative replace it with NaN
        
        return keypoints
        
    def addExtraPoints(self, keypoints):
        """
        Function addExtraPoints
        Used for adding extra points to the keypoints list
        Takes as input the keypoints for the frame
        Returns the list of keypoints with added extra points

        """
        if keypoints != []:
            
            torso_up = keypoints[self.torso_up].mean(axis = 1)    #Get the midpoint of the shoulders using the mean of left and right shoulders

            torso_down = keypoints[self.torso_down].mean(axis=1)    #Get the midpoint of the hips using the mean of left and right shoulders

            head_coordinate = np.nanmean(keypoints[:5], axis = 0)   #Get the mean of the head coordinate as one points instead of the five points

            keypoints = np.vstack((keypoints, torso_up, torso_down, head_coordinate, self.vertical_coordinates))    #Stack all the points with each other
                
        return keypoints
     
    def clip_from_to(self, costs):  
        """
        Function clip_from_to
        Used for bounding the cost list using previously defined bounds
        Takes as input the cost list
        Returns the list of the bounded costs

        """
        
        sorted_ = np.sort(costs.reshape((len(costs))), axis = -1)   #Sort the costs
        
        mean_start = np.mean(sorted_[0:int(len(sorted_)*0.1)])  #Mean of the lowest 10% values
        
        mean_end = np.mean(sorted_[(len(sorted_)-int(len(sorted_)*0.1)):len(sorted_)])  #Mean of the top 10% values
        
        result = np.clip(costs,mean_start,mean_end) #Bound the list with that values
        
        normalized = (result - mean_start)/(mean_end-mean_start)    #Normalize the costs using MinMaxScaling
        
        return normalized.reshape((len(normalized),1))  
    
    def chooseThreshold(self, cost_method):
        """
        Function chooseThreshold
        Used for choosing the threshold based on the method for cost computation
        Takes as input the cost method
        Returns the threshold for that cost method

        """
        
        if cost_method == 'DifferenceMean':
            
            self.threshold = 58
            
        elif cost_method == 'DifferenceSum':
            
            self.threshold = 55
            
        elif cost_method == 'MeanDifference':
            
            self.threshold = 5
            
        elif cost_method == 'Mean':
            
            self.threshold = 37
            
        elif cost_method == 'Division':
            
            self.threshold = 8.5
            
        return self.threshold
            
    def processVideo(self, video, cost_method):

        """
        Function processVideo
        Used for computing the cost for the entire video
        Takes as input the video and cost method
        Returns the list of the costs computed


        """
                        
        camera_video = cv2.VideoCapture(video)    #Capture the video
                
        camera_video.set(3,1280)   #Width of the video
        
        camera_video.set(4,960)    #Height of the video
        
        video_fps = camera_video.get(cv2.CAP_PROP_FPS)  #Get the fps of the video
                
        if video_fps == 30.0:   #If 30 fps
            
            pass
            
        else:
            
            return 'Video is not 30 FPS'    #If not 30 fps terminate the function
        
        frame_index = 0    #Frame Index
        
        previous_keypoints = 0   #Variable for storing the previous keypoints
        
        previous_cost = 0   #Variable for storing the previous cost
        
        step_size = video_fps // self.fps    #Step size of the frames (If 5, we consider 0th frame, then fifth, then tenth, etc.)
        
        self.costlist = []  #List for storing costs
        
        cache = []   #List for storing the cache of the costs
                        
        while camera_video.isOpened():  #While video is running 
                                    
            condition, frame = camera_video.read()  #Read every frame
                        
            if condition is False:  #If no frames left break the loop
                
                break
                
            if frame_index % step_size == 0:    #If the frame_index is divisible by step_size
                                
                current_keypoints = self.keypoints.detectPoints(frame)  #Find the keypoints for the current frame
            
                if frame_index == 0:    #If frame index is 0

                    previous_keypoints = current_keypoints  #Make the previous keypoints the current one
                    
                    previous_cost = 0   #Make the previous cost 0

                    frame_index += 1   #Add 1 to frame_index and continue

                    continue    
  
                previous_keypoints = self.collectData(previous_keypoints)    #Handle missing values and add extra ones for previous frame
        
                current_keypoints = self.collectData(current_keypoints)  #Handle missing values and add extra ones for current frame
                
                vector1_pairs = np.array(previous_keypoints[self.vector_indices][self.pair_indices])    #Get vector pairs for previous keypoints

                vector2_pairs = np.array(current_keypoints[self.vector_indices][self.pair_indices])     #Get vector pairs for current keypoints

                vector1_angles = self.angleCalculation(vector1_pairs)*self.angle_weights    #Calculate the angles for previous frame and multiply with weights

                vector2_angles = self.angleCalculation(vector2_pairs)*self.angle_weights    #Calculate the angles for current frame and multiply with weights
                                
                if np.count_nonzero(np.isnan(vector1_angles)) >= 6 or np.count_nonzero(np.isnan(vector2_angles)) >= 6:  #If more than six vectors are NaNs drop the frame and continue
                    
                    continue
                    
                start = time.time() #Calculate the time for the cost computation
                                    
                if cost_method == 'DifferenceMean':
                
                    cost = self.differenceMean(vector1_angles, vector2_angles)
                
                elif cost_method == 'DifferenceSum':
                    
                    cost = self.differenceSum(vector1_angles, vector2_angles)
                    
                elif cost_method == 'Division':
                
                    cost = self.divisionCost(vector1_angles, vector2_angles)
                    
                elif cost_method == 'Mean':
                
                    cost = self.costMean(vector2_angles)
                    
                elif cost_method == 'MeanDifference':
                    
                    cost = self.meanDifference(vector1_angles, vector2_angles)
                    
                else:
                    
                    print('Not Valid Method!! Use "DifferenceMean", "MeanDifference", "DifferenceSum", "Division" or "Mean" as cost method!!!!')
                    return False
                
                end = time.time()
                
                if np.isnan(cost):  #If cost is NaN, take previous cost instead of NaN
                    
                    cost = previous_cost
                
                cache.append(cost)  #Append the cost to cache
                            
                if frame_index >= step_size*6:  #If the cache contains more than 5 elements

                    weighted_cost = np.dot(self.cache_weights, cache) / 6    #Calculate the cost based on previous 6 costs

                    cache = cache[1:]   #Remove the last element of the cache to append the current cost

                    self.costlist.append(weighted_cost) #Append the weighted cost to the cost list
                                                                                                                                                
                previous_keypoints = current_keypoints  #Assign the current keypoints to the previous keypoints for the next frame
                
                previous_cost = cost    #Assign current cost to the previous cost for the next frame
            
            frame_index += 1   #Add 1 to frame index
                
            k = cv2.waitKey(1) & 0xff

            if(k == 27):    #If esc is pressed break
                
                break
                
        camera_video.release()
        
        cv2.destroyAllWindows()

        return np.array(self.costlist)
    

    def realTimeVideo(self, video, cost_method, save = False):

        """
        Function processVideo
        Used for computing the cost for the entire video
        Takes as input the video and cost method
        Returns the list of the costs computed


        """
        
        plot = plt.figure(figsize=(5,5))
                        
        camera_video = cv2.VideoCapture(video)    #Capture the video
                
        camera_video.set(3,1280)   #Width of the video
        
        camera_video.set(4,960)    #Height of the video
        
        if save:
        
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')

            out = cv2.VideoWriter('FallDetection.mp4', fourcc, 6.0, (int(camera_video.get(3))+500, 500))

        video_fps = round(camera_video.get(cv2.CAP_PROP_FPS))  #Get the fps of the video
                        
        if video_fps == 30.0:   #If 30 fps
            
            pass
            
        else:
            
            return 'Video is not 30 FPS'    #If not 30 fps terminate the function
        
        frame_index = 0    #Frame Index
        
        previous_keypoints = 0   #Variable for storing the previous keypoints
        
        previous_cost = 0   #Variable for storing the previous cost
        
        step_size = video_fps // self.fps    #Step size of the frames (If 5, we consider 0th frame, then fifth, then tenth, etc.)
        
        self.costlist = []  #List for storing costs
        
        cache = []   #List for storing the cache of the costs
                        
        while camera_video.isOpened():  #While video is running 
                                    
            condition, frame = camera_video.read()  #Read every frame
            
            plot.canvas.draw()
                        
            if condition is False:  #If no frames left break the loop
                
                break
                
            if frame_index % step_size == 0:    #If the frame_index is divisible by step_size
                                
                current_keypoints = self.keypoints.detectPoints(frame)  #Find the keypoints for the current frame
            
                if frame_index == 0:    #If frame index is 0

                    previous_keypoints = current_keypoints  #Make the previous keypoints the current one
                    
                    previous_cost = 0   #Make the previous cost 0

                    frame_index += 1   #Add 1 to frame_index and continue

                    continue    
                      
                previous_keypoints = self.collectData(previous_keypoints)    #Handle missing values and add extra ones for previous frame
        
                current_keypoints = self.collectData(current_keypoints)  #Handle missing values and add extra ones for current frame
                
                vector1_pairs = np.array(previous_keypoints[self.vector_indices][self.pair_indices])    #Get vector pairs for previous keypoints

                vector2_pairs = np.array(current_keypoints[self.vector_indices][self.pair_indices])     #Get vector pairs for current keypoints

                vector1_angles = self.angleCalculation(vector1_pairs)*self.angle_weights    #Calculate the angles for previous frame and multiply with weights

                vector2_angles = self.angleCalculation(vector2_pairs)*self.angle_weights    #Calculate the angles for current frame and multiply with weights
                                
                if np.count_nonzero(np.isnan(vector1_angles)) >= 6 or np.count_nonzero(np.isnan(vector2_angles)) >= 6:  #If more than six vectors are NaNs drop the frame and continue
                    
                    continue
                    
                start = time.time() #Calculate the time for the cost computation
                                    
                if cost_method == 'DifferenceMean':
                
                    cost = self.differenceMean(vector1_angles, vector2_angles)
                
                elif cost_method == 'DifferenceSum':
                    
                    cost = self.differenceSum(vector1_angles, vector2_angles)
                    
                elif cost_method == 'Division':
                
                    cost = self.divisionCost(vector1_angles, vector2_angles)
                    
                elif cost_method == 'Mean':
                
                    cost = self.costMean(vector2_angles)
                    
                elif cost_method == 'MeanDifference':
                    
                    cost = self.meanDifference(vector1_angles, vector2_angles)
                    
                else:
                    
                    print('Not Valid Method!! Use "DifferenceMean", "MeanDifference", "DifferenceSum", "Division" or "Mean" as cost method!!!!')
                    return False
                
                end = time.time()
                
                if np.isnan(cost):  #If cost is NaN, take previous cost instead of NaN
                    
                    cost = previous_cost
                
                cache.append(cost)  #Append the cost to cache
                            
                if frame_index >= step_size*6:  #If the cache contains more than 5 elements

                    weighted_cost = np.dot(self.cache_weights, cache) / 6    #Calculate the cost based on previous 6 costs

                    cache = cache[1:]   #Remove the last element of the cache to append the current cost

                    self.costlist.append(weighted_cost) #Append the weighted cost to the cost list
                                                                                                                                             
                previous_keypoints = current_keypoints  #Assign the current keypoints to the previous keypoints for the next frame
                
                previous_cost = cost    #Assign current cost to the previous cost for the next frame
                
                threshold = self.chooseThreshold(cost_method)
            
                cv2.putText(frame, 'Frame: ' + str(frame_index/5), (0, 150), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2, cv2.LINE_AA)
                
                plt.clf()    #Clear the plot

                plt.xlim(frame_index/5-15,frame_index/5+15)    #Define the limit of x axis

                plt.ylim(0,self.threshold+50)    #Define the limit of y axis

                plt.plot(self.costlist) #Plot the costlist

                x_cord = [frame_index/5-15,frame_index/5+15]    #The threshold x cord

                y_cord = [threshold, threshold]    #The threshold y cord

                plt.plot(x_cord, y_cord, color='red')    #Plot the threshold line

                plot.canvas.flush_events()   #Clears the old figure

                img = np.fromstring(plot.canvas.tostring_rgb(), dtype=np.uint8, sep='')    #Used to convert plot to image

                img  = img.reshape(plot.canvas.get_width_height()[::-1] + (3,))    #Used to convert plot to image

                img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)    #Convert the image to BGR

                h1, w1 = frame.shape[:2]

                h2, w2 = img.shape[:2]

                merged = np.zeros((max(h1, h2), w1+w2,3), dtype=np.uint8)

                merged[:,:] = (255,255,255)

                merged[:h1, :w1,:3] = frame

                merged[:h2, w1:w1+w2,:3] = img
                
                if save:
                
                    out.write(merged)

                cv2.imshow('plot', merged)


            frame_index += 1   #Add 1 to frame index

            k = cv2.waitKey(1) & 0xff

            if(k == 27):    #If esc is pressed break
                
                break
                
        camera_video.release()
        
        if save:
            
            out.release()
        
        cv2.destroyAllWindows()
        
    def plot(self, axis, cost, costmethod, fall_start, fall_end):
    
        """
        Function plot
        Used for plotting the cost list
        Takes as input the cost, starting frame of the fall and ending frame
        Returns the plot

        """
        threshold = self.chooseThreshold(costmethod)
        
        axis.plot(cost, label = 'cost')

        axis.set_title('Cost method is: ' + costmethod)

        axis.axhline(y = threshold, label = 'Threshold', color='black')

        axis.axvspan(fall_start, fall_end, alpha=0.25, color='red', label = 'Fall Frames')   

        axis.legend(loc="upper right")

    def separatePlot(self, cost, costmethod, save = False):
    
        """
        Function plot
        Used for plotting the cost list
        Takes as input the cost, starting frame of the fall and ending frame
        Returns the plot

        """
        threshold = self.chooseThreshold(costmethod)
        
        plot = plt.figure(figsize=(10,10))

        plt.plot(cost, label = 'cost')

        plt.title('Cost method is: ' + costmethod)

        # plt.axhline(y = threshold, label = 'Threshold', color='black')

        plt.legend(loc="upper right")

        plot.canvas.draw()

        img = np.fromstring(plot.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        
        img  = img.reshape(plot.canvas.get_width_height()[::-1] + (3,))

        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        if save is True:

            cv2.imwrite('FallDetection.png', img)

        cv2.imshow("plot",img)      

        k = cv2.waitKey(10000) & 0xff

        if(k == 27):    #If esc is pressed break
            
            cv2.destroyAllWindows()



if __name__ == '__main__':
    
    vid = '..//Data//Fall3.mp4'
    
    featureextractor = FeatureExtractor()
    
    cost_method = 'DifferenceMean'
    
    cost = featureextractor.processVideo(vid, cost_method)
    
    plot = plt.figure(figsize=(5,5))
    
    plt.plot(cost, label = 'cost')
    
    plt.axhline(y= featureextractor.chooseThreshold(cost_method), label = 'Threshold', color='black')
    
    plt.axvspan(17-6, 25-6, alpha=0.25, color='red')
    
    plt.legend(loc="upper left")
    
    plt.show()






