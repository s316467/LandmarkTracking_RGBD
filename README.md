<div align="center">
 
   <h1>LandmarkTracking_RGBD</h1>
 
   ![example workflow](https://github.com/Ziemnono/LandmarkTracking_RGBD/actions/workflows/main.yml/badge.svg)
   ![example workflow](https://github.com/Ziemnono/LandmarkTracking_RGBD/actions/workflows/codeql-analysis.yml/badge.svg)

</div> 


<h2>About the Project</h2>

The LandmarkTracking_RGBD repository is a software library built to track facial landmarks across a series of images taken with an RGBD camera. The library is written primarily in [Python](https://www.python.org/downloads/), and features machine learning and landmark detection packages from [dlib](https://github.com/davisking/dlib) and [OpenCV](https://github.com/opencv/opencv).


<h2>Applications</h2>

LandmarkTracking_RGBD contains software that can be used to detect faces and follow selected facial landmarks across a series of images taken with an RGBD camera. 

In the following example, a subject took a series of 34 images of themselves with the front facing camera of an Apple iPhone. For each image, a depth map was generated using the infrared sensors that are used in the iPhone’s Face ID facial recognition system. Using the shape predictor tools in the [dlib](https://github.com/davisking/dlib) toolkit, it is possible to calculate the locations of various facial landmarks (e.g. mouth, nose, left or right eye) for each of the 34 images and their corresponding depth maps. <strong>Table 1</strong> presents one of the 34 images, its corresponding depth map, and a GIF of LandmarkTracking_RGBD tracking the subject’s left eye across all 34 images.


<strong>Table 1: Example of inputs and output of LandmarkTracking_RGBD</strong>

<div align="center">  
 <table>
  <tr align = "middle">
     <td>1. Optical Image</td>
     <td>2. Depth Map</td>
     <td>3. Eye Tracking</td>
  </tr>
  <tr>
    <td valign="top"><img src="/resources/Optical.jpg" width=300 ></td>
    <td valign="top"><img src="/resources/Depth.jpg" width=300 ></td>
    <td valign="top"><img src="/resources/Visualisation.gif" width=300 ></td>
  </tr>
 </table>    
</div>  


<h2>Getting Started</h2>

LandmarkTracking_RGBD builds on modules from dlib and OpenCV. For more information about configuring these libraries, please see the following links:

- [dlib](http://dlib.net/compile.html) 
     > “A toolkit for making real world machine learning and data analysis applications in C++”

- [OpenCV]( https://docs.opencv.org/4.x/da/df6/tutorial_py_table_of_contents_setup.html)
     > “Open Source Computer Vision Library”