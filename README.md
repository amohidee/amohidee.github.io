Title: Parallel General Fast Radial Symmetry Transform for Image Segmentation

URL: https://amohidee.github.io/

Summary: We are going to implement and optimize the general fast radial symmetry algorithm. We will then perform post-processing to turn our detected circles into a segmented image.

Background: Fast Radial Symmetry Transform is a technique used to detect points of radial symmetry, which can be considered as generalized circles. It’s useful for identifying features that exhibit some form of circularity in the image. Some example use cases are detecting cells in biology, as well as finding locations of interest from satellite imaging. The generalized fast radial symmetry algorithm detects transformed circles (ellipses). A circle from one perspective is an ellipse from another and can be represented as a transformation. The algorithm is as follows.

The first step is calculating image gradients, generally done via a convolution. Then, you perform a transformation to calculate the orientation projection image and magnitude projection image. These are combined along with some convolutions to determine the center of the circle, which allows us to increment a vote for this position. Afterwards, we perform some processing to remove redundant and insignificant points. At this point, we have the circle centers. Now, in order to perform image segmentation, we want to paint the circles a solid color. However, we need to find the order in which to paint, as the circles can be detected through occlusion from other image objects. We can model the occlusions as a DAG, and then process the DAG in topological order in parallel as possible. 

Many of the steps in the first half will benefit greatly from parallelism. Once the topological ordering is generated, we can parallelise the segmentation painting in a number of ways. 

Challenge: The main challenge of this project is parallely creating a graph representing the layering of the objects described by the ellipses. The end goal of finding the ellipses is shading the objects to create a segmented image. However, in order to do this, we need to know what objects are layered on top of other objects. Ideally we will be doing work on each circle in parallel to determine which objects are layered on top of it. However, there are some divergent execution and load balancing issues here. Not all the ellipses will have overlapping ellipses, and even between ellipses that are occluded there are different amounts of overlapping ellipses to take care of. What we want to avoid is one thread doing no work for the highest layer ellipse and terminating, while another thread gets bogged down calculating a bunch of overlaps.



Resources:
We will be drawing the mathematical basis of the algorithm from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1217601 and https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6247768. 

We will be implementing this sequentially on a CPU, as well as optimize it parallely on a GPU and compare the results. We will have to develop the code from scratch, but will be able to use the equations from the paper as a baseline. 

Goals:

We plan to complete both a sequential and parallel version of this GFRST algorithm. 

We plan to compare the two results in terms of speedup, and break down the subcomponents of the algorithms to compare as well. 

We plan to achieve significant speedup on the GPU implementation. 

If we have time, we will look into using Halide to run this code and compare performance. 

The demo we show will be a combination of our results (speedup, etc…) as well as output images from the segmentation pipeline. If we have time, we will create a live demo, where we take an image in real time and show its outputs. 

Schedule:

Week
TODO
4/7 - 4/13
Working solution to calculating gradients of the blocked input image
4/14 - 4/20
Implement an accumulator and be able to get the basic info of each ellipse in the image and start unoptimized graph creation.
4/21 - 4/27
Continue optimizing the graph creation and GFRST algorithm.
4/28 - 5/4
Finish optimizing and record performance evaluations.




