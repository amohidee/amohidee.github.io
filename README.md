PROPOSAL

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


MILESTONE

Schedule:

Week
TODO
4/16-4/21
Implement accumulator - Anees
NMS for gradients - Kush
4/21-4/24
Naive graph creation - Anees
Sequential code - Kush
4/24-4/27
Optimise graph creation - Anees
Optimise other steps  - Kush
4/28-5/1
Gather data - Anees
make graphs for performance evaluations - Kush


5/1-5/4
Write the report


So far, we have created the framework for ingesting images, and transferring them between CPU and GPU. Additionally, we have initial kernels for calculating the gradients. One kernel will convert the color image to grayscale. The second kernel naively calculates convolutions of the derivative filters, and we have started looking into ways to optimise this step. We also have started work on the sequential version we will be comparing against for speedup. 


At this point we are on schedule but there is still a lot of work left to do. Our original ideas for this final project were lacking complexity and needed to be revised, so we only settled on this project recently. However, in regards to the schedule we devised in our proposal, we are on track. Its just that this schedule will have us do a lot of work in the final 3-4 weeks. With this in mind it is unlikely that we will finish early and have ample time to work on extra deliverables, but we should still be able to finish our original plans.


At the poster session we can show a live demonstration of our algorithm taking in an input image and outputting a modified version of that image with the shaded circles and with performance numbers.

We have not finished the entire GFRST algorithm so we do not have preliminary results.
We still haven't figured out how to parallelize the detection of overlapping circles. Part of our algorithm involves shading the detected circles, but to properly shade we need to deduce the layers of the different circles. We have a naive brute force idea of checking each pair of circles for overlap but we haven’t yet figured out our optimal parallel solution. Also we are using ImageMagick for working with images but we are not allowed to install it on the ghc machines, so we need to get that resolved or find a workaround.




