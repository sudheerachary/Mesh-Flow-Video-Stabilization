![stable-output](https://github.com/sudheerachary/Mesh-Flow-Video-Stabilization/blob/master/results/sample.gif)

[![HitCount](http://hits.dwyl.io/sudheerachary/Mesh-Flow-Video-Stabilization.svg)](http://hits.dwyl.io/sudheerachary/Mesh-Flow-Video-Stabilization)

# Mesh-Flow-Video-Stabilization

The MeshFlow is a spatial smooth sparse motion field with motion vectors only at the mesh vertexes. The MeshFlow is produced by assigning each vertex an unique motion vector via two median filters. The path smoothing is conducted on the vertex profiles, which are motion vectors collected at the same vertex location in the MeshFlow over time. The profiles are smoothed adaptively by a novel smoothing technique, namely the Predicted Adaptive Path Smoothing (PAPS), which only uses motions from the past.

## Getting Started

  - To stabilize a video execute the script `src/Stabilization.py`
  
    - ```
      python Stabilization.py <path_to_video>
      ```
  - To run experiments ipython-notebook is present in `src/mesh_flow.ipynb`
  - The stable output video is saved to `home` directory of cloned repository

### Prerequisites

  - Required Packages:
  
    - **opencv**:  `pip install --user opencv-python`
  
    - **numpy**:  `pip install --user numpy`
    
    - **scipy**: `pip install --user scipy`
    
    - **tqdm**: `pip install --user tqdm`
  
  - Optional Packages:
    
    - **cvxpy**: `pip install --user cvxpy`

## Motion Propagation

Mesh Flow only operate on a sparse regular grid of vertex profiles, such that the expensive optical flow can be replaced with cheap feature matches. For one thing, they are similar because they both encode strong spatial smoothness. For another, they are different as one is dense and the other is sparse. Moreover, the motion estimation methods are totally different. Next, we show estimatation of spacial coherent motions at mesh vertexes.

### Initial motion vectors

![initial-motion-mesh](https://github.com/sudheerachary/Mesh-Flow-Video-Stabilization/blob/master/results/old_motion_vectors/148.jpg)

  - initial motion meshes are dumped into `results/old_motion_vectors`

### Final motion vectors

![final-motion-mesh](https://github.com/sudheerachary/Mesh-Flow-Video-Stabilization/blob/master/results/new_motion_vectors/148.jpg)
  
  - final motion meshes are dumped into `results/new_motion_vectors`

## Predicted Adaptive Path Smoothing (PAPS)

A vertex profile represents the motion of its neighboring image regions. MeshFlow can smooth all the vertex profiles for the smoothed motions. It begin by describing an offline filter, and then extend it for online smoothing.

### Path smoothening

![path](https://github.com/sudheerachary/Mesh-Flow-Video-Stabilization/blob/master/results/paths/0_30.png)

  - **Green** is the optimized result
  - **Blue** is the motion vector accumulation
  - vertex profile paths are dumped into `results/paths`

## Acknowledgments

* [MeshFlow: Minimum Latency Online Video Stabilization](http://www.liushuaicheng.org/eccv2016/meshflow.pdf)
* [MeshFlow Video Denoising](https://github.com/AlbusPeter/MeshFlow_Video_Denoising)
* [SteadyFlow: Spatially Smooth Optical Flow for Video Stabilization](http://www.liushuaicheng.org/CVPR2014/SteadyFlow.pdf)
