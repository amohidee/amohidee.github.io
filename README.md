# amohidee.github.io
15418 final project: Parallelizing the APSP algorithm
Title: Solving Dijkstra’s Algorithm with Parallel Priority Queues. 

URL: 

https://amohidee.github.io/

Summary:

We are going to make a parallel priority queue implementation and use it to parallelize djikstras algorithm. We will then evaluate the performance of this implementation against number of threads and graph properties (denseness, etc…)

Background:
The Shortest Path problem, finding the shortest path between two nodes in a graph, is a classic problem in graph theory. This problem has significant applications in various fields such as network analysis, geographical information systems, and routing protocols. Dijkstra’s algorithm is a well known solution to this problem. The algorithm goes as follows:
For an input graph G with V vertices, weights W, and source vertex S. 
Create an unvisited set and put all nodes into it.
Create an array to keep track of the distance to the source. Make this value for the source itself 0. 
Create a queue that we will use to keep track of the nodes that are seen but not visited.
For a node (starting with the source node): Look at all the unvisited neighbors of the node. Set their distance equal to the weights of their edges (the weights represent the distance from other nodes) + the distance of the current node. Remove the current node from the unvisited set. Add all the neighbors to the queue. Repeat set 4 with the node in the queue with the shortest distance to the source until unvisited is empty.
Each time this algorithm reaches a new node, the path that it took to reach it is the shortest path from the source to the node. Therefore, we have found the shortest path from the source to all other nodes.
However, this code seems very sequential and as such will have a very slow runtime, especially on large graphs (O(V+ElogV) runtime with min heap priority queue). The idea here is to parallelize the priority queue. The priority queue is represented by a min heap. We have two approaches for this. One is to have each processor store its own priority queue. When we want to find the min of the priority queue, each processor finds the min of its own priority queue, then a reduce operation happens across the processors to find the actual min of the priority queue. Another method is to have one priority queue but have multiple processors operate on that one priority queue, but lock the subtrees that the operations are happening in. Because the queue can be very big, there is no point locking the whole tree when only a small subtree of it can be used. Therefore we can have multiple processors operate on the tree at the same time.

Challenge: 
The challenge is that priority queues are very dependent on the data within the structure. Thus, two processors attempting to load and store at the same time have a lot of dependence on each other. This will make it difficult to parallelise as we have to consider how to manage the communication on insertions, pops, and deletions. 

The modifications we make are also constrained by the complexity constraints of the algorithm. We hope to see a speedup on dijkstra's algorithm, so we need to ensure the cost of managing communication does not outweigh the benefits of parallelism. As we scale to more processors, we want to optimize for and see the performance of our data structure. 



Resources:
https://arxiv.org/abs/1908.09378
We can use GHC for general performance and then maybe use PSC for high core count experiments. 

Goals & Deliverables : 
We plan to complete 2 implementations of a parallel priority queue(locking the whole heap vs less broadly via “subheaps”) in C++. 

We also plan to complete a parallelized version of Djikstra’s algorithm that we will use to benchmark performance of our priority queue implementations. We plan for this parallelized version to show improvements over a single threaded implementation. 

We plan to collect results for different types of graph workloads (dense vs sparse graphs, etc…) 

We hope to compare our implementation to the delta stepping algorithm if time permits.

If time permits, we hope to implement a CUDA based version of the priority queue that references (https://ieeexplore.ieee.org/document/6507490). 

Platform choice: We plan to use C++ for our coding language, OpenMP for the shared address space single priority queue implementation, and MPI for the multiple priority queue message passing solution. C++, OpenMP, and MPI are industry standards and are known to be low overhead, quick interfaces. 










Schedule:

Week
TODO
3/31 - 4/6
Design implementation of multiple priority queues
4/7 - 4/13
Working solution to maintaining multiple priority queues in parallel and finding global min
4/14 - 4/20
Solving Dijkstra’s with multiple priority queue implementation and doing performance evaluations
4/21 - 4/27
Design implementation of single priority queue and begin actual implementation.
4/28 - 5/4
Finish implementation and record performance evaluations.

