import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt


#task 1
#draw cross in the keypoint
def draw_crosses(img, keypoints):
    for kp in keypoints:
        # Get the position of the point
        x, y = map(int, kp.pt)  

        # Draw a cross
        cv2.drawMarker(img, (x, y), (0, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)

def task_one(image_path):
   img = cv2.imread(image_path)


   if img is None:
      print("invalid img")
      return
   
   #resize image to VGA size while maintaing aspect ratio
   vga_size = (600, 480)
   height, width = img.shape[:2]
   scale = min(vga_size[1]/height , vga_size[0]/width)
   new_height= int(height * scale)
   new_width = int(width * scale)
   resized_img = cv2.resize(img,(new_width, new_height))


   #gray scale
   gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

   #sift points
   sift = cv2.SIFT_create()
   kP, des = sift.detectAndCompute(gray_img,None)
   keypoint_img = gray_img.copy()
   draw_crosses(keypoint_img,kP)
   
   #draw keypoints and use flag to get radius and orientation of the keypoint
   img_1 = cv2.drawKeypoints(keypoint_img,kP,0,(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
   #combine 2 image in a single window
   hs_image=np.hstack((resized_img,img_1))
   #print the number of keypoint detected
   print("# of keypoints in",image_path,"is", len(kP))
   


   #showing images
   cv2.imshow('original img',hs_image)
   cv2.waitKey(0)
   cv2.destroyAllWindows() 


##for task two

sift_des = []
def sift(image_path):
   img = cv2.imread(image_path)


   if img is None:
      print("inbvalid img")
      return
   
   #resize image to VGA size while maintaing aspect ratio
   vga_size = (600, 480)
   height, width = img.shape[:2]
   scale = min(vga_size[1]/height , vga_size[0]/width)
   new_height= int(height * scale)
   new_width = int(width * scale)
   resized_img = cv2.resize(img,(new_width, new_height))


   #gray scale
   gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

   #sift points
   sift = cv2.SIFT_create()
   kP, des = sift.detectAndCompute(gray_img,None)

   sift_des.extend(des)
  
   
   return kP, des

def find_kmeans_clusters(sift_descriptors, num_clusters):
    #convert the list of sift descriptors to a numerical format
    sift_descriptors = np.float32(sift_descriptors)
    #define the termination criteria and flage.
    #maximum iteration 100
    #desired accuracy(epsilon) 0.2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    flag = cv2.KMEANS_RANDOM_CENTERS
    #apply k means clustering to the sift descriptors
    #here center is visual words
    _, labels, centers = cv2.kmeans(sift_descriptors, num_clusters, None, criteria, 10, flag)

    #return visual words
    return centers

def construct_hist(descriptors, k_centers):
    # initialize a histogram with zeros
    hist = np.zeros(len(k_centers))
    #iterate through each sift descriptor
    for des in descriptors:
        #calculate distance between the descriptor and all cluster centers
        distance = np.linalg.norm(des - k_centers,axis=1)
        #find the index of the nearest cluster center
        nearest_c = np.argmin(distance)
        #increment the count for the cluster(word)
        hist[nearest_c] += 1

    return hist    


def normalize_histogram(histogram):
    # normalize the histogram
    norm_hist = histogram / np.sum(histogram)
    return norm_hist

def cal_x2_distance(hist1, hist2):
    # calculate the X^2 distance between two normalized histograms, 1e-6 is added to avoid division by zero
    x2_distance = 0.5 * np.sum((hist1 - hist2) ** 2 / (hist1 +hist2 +1e-6))
    return x2_distance

def task_two(img_paths, kmeans_centers, kpercent , k):
    # list to store histograms for all images
    hist = []  
    
    for img_path in img_paths:
        # get SIFT keypoints and descriptors
        kP, des = sift(img_path)  

        if kP is None or des is None:
            continue
        
        # construct histogram for the image
        histogram = construct_hist(des, kmeans_centers)  
        # normalize the histogram
        norm_histogram = normalize_histogram(histogram)  
        hist.append(norm_histogram)
    
    #calculate no of image in img_path
    num_images = len(img_paths)
    
    
    # calculate X^2 distance between histograms for all pairs of images
    dissimilarity_matrix = np.zeros((num_images, num_images))

    for i in range(num_images):
        for j in range(i, num_images):
            x2_distance = cal_x2_distance(hist[i], hist[j])
            dissimilarity_matrix[i, j] = x2_distance
            dissimilarity_matrix[j, i] = x2_distance

    print("k=", kpercent, "% * (total number of keypoionts) = ", k)
    # display the dissimilarity matrix in a readable format
    print("Dissimilarity Matrix")
    print("        ", end="  ")
    for img_path in img_paths:
        print(f"{img_path:<10}", end="   ")
        
    print()

    for i in range(num_images):
        print(f"{img_paths[i]:<4}", end="   ")
        for j in range(num_images):
            print(f"{dissimilarity_matrix[i, j]:.2f}   ", end="       ")
        print()  
         
#taking input from terminal
#for task 1
if len(sys.argv) == 2:
    if sys.argv[1].endswith(".jpg") or sys.argv[1].endswith(".png") or sys.argv[1].endswith(".jpeg"):
     task_one(sys.argv[1])
    else:
      print("invalid input")  


#for task 2
if len(sys.argv) == 6:
    if sys.argv[1].endswith(".jpg") or sys.argv[1].endswith(".png") or sys.argv[1].endswith(".jpeg"):
     
        img_paths = sys.argv[1:]
        total_keypoints = 0

        for img_path in img_paths:
            # get SIFT keypoints and descriptors
            kP, des = sift(img_path)  

            if kP is None or des is None:
                continue
            print("# of keypoints in",img_path,"is", len(kP))
            total_keypoints += len(kP)
        
        #percentage to calculate k
        kpercent1 = 5
        kpercent2 = 10
        kpercent3 = 20 

        # define fixed values of K as a percentage of total keypoints
        k1 = int(5 * total_keypoints / 100)
        k2 = int(10 * total_keypoints / 100)
        k3 = int(20 * total_keypoints / 100)

        # compute kmeans_centers for each value of K
        kmeans_k1 =find_kmeans_clusters(sift_des, k1)
        kmeans_k2 =find_kmeans_clusters(sift_des, k2)
        kmeans_k3 =find_kmeans_clusters(sift_des, k3)

        
        print("total Keypoint ",total_keypoints)
    

        task_two(img_paths, kmeans_k1, kpercent1, k1)
        task_two(img_paths, kmeans_k2,kpercent2, k2)
        task_two(img_paths, kmeans_k3,kpercent3, k3)

    else:
      print("invalid input")    
