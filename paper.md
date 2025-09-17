# PINGS: Gaussian Splatting Meets Distance Fields within a Point-Based Implicit Neural Map 

Yue Pan* Xingguang Zhong* Liren Jin* Louis Wiesmann*<br>Marija Popović ${ }^{\ddagger}$ Jens Behley* Cyrill Stachniss ${ }^{\star, \dagger}$<br>* Center for Robotics, University of Bonn, Germany<br>${ }^{\ddagger}$ MAVLab, TU Delft, the Netherlands<br>${ }^{\dagger}$ Lamarr Institute for Machine Learning and Artificial Intelligence, Germany

![img-0.jpeg](img-0.jpeg)

Fig. 1: We present PINGS, a novel LiDAR-visual SLAM system unifying distance field and radiance field mapping using an elastic pointbased implicit neural representation. On the left, we show a globally consistent neural point map overlaid on a satellite image. The map was built using PINGS from around 10,000 LiDAR scans and 40,000 images collected by a robot car driving in an urban environment for around 5 km . The estimated trajectory is overlaid on the map and colorized according to the timestep. On the right, we show a zoomed-in view of a roundabout mapped by PINGS. It illustrates from left to right the rendered image from the Gaussian splatting radiance field, neural points colorized by the principal components of their geometric features, and the reconstructed mesh from the distance field (colorized by the radiance field). The red line indicates the local trajectory of the robot car (shown as the CAD model).

Abstract-Robots benefit from high-fidelity reconstructions of their environment, which should be geometrically accurate and photorealistic to support downstream tasks. While this can be achieved by building distance fields from range sensors and radiance fields from cameras, realising scalable incremental mapping of both fields consistently and at the same time with high quality is challenging. In this paper, we propose a novel map representation that unifies a continuous signed distance field and a Gaussian splatting radiance field within an elastic and compact point-based implicit neural map. By enforcing geometric consistency between these fields, we achieve mutual improvements by exploiting both modalities. We present a novel LiDAR-visual SLAM system called PINGS using the proposed map representation and evaluate it on several challenging largescale datasets. Experimental results demonstrate that PINGS can incrementally build globally consistent distance and radiance fields encoded with a compact set of neural points. Compared to state-of-the-art methods, PINGS achieves superior photometric and geometric rendering at novel views by constraining the radiance field with the distance field. Furthermore, by utilizing dense photometric cues and multi-view consistency from the radiance field, PINGS produces more accurate distance fields, leading to improved odometry estimation and mesh reconstruction. We also provide an open-source implementation of PINGS.

## I. INTRODUCTION

The ability to perceive and understand the surroundings is fundamental for autonomous robots. At the core of this capability lies the ability to build a map - a digital twin of the robot's workspace that is ideally both geometrically accurate and photorealistic, enabling effective spatial awareness and operation of the robot [24, 41].

Previous works in robotics mainly focus on the incremental mapping of an occupancy grid or a distance field using range sensors, such as LiDAR or depth cameras, which enable localization [16], collision avoidance [14], or exploration [59]. Recently, PIN-SLAM [51] demonstrated that a compact pointbased implicit neural representation can effectively model a continuous signed distance field (SDF) for LiDAR simultaneous localization and mapping (SLAM), enabling both accurate localization and globally consistent mapping.

However, occupancy voxel grids [16], occupancy fields [91], or distance fields [49, 51] fall short of providing photorealistic novel view rendering of the scene, which is crucial for applications requiring dense photometric information. This capability can be achieved by building an additional radiance field with visual data using representations such as neural radiance field (NeRF) [45] or a 3D Gaussian splatting (3DGS) model [30]. Recent works demonstrated the potential of radiance fields, especially 3DGS, for various robotic applications including human-robot interaction [43], scene understanding [92, 97], simulation or world models for robotics learning [2, 12, 81], visual localization [4, 42], and active reconstruction [26, 27]. Nevertheless, these approaches often assume well-captured image collections in bounded scenes with offline processing, limiting their applicability for mobile robotic applications. Besides, radiance fields are not necessarily geometrically accurate, which can lead to issues in localization or planning.

In this paper, we investigate how to simultaneously build consistent, geometrically accurate, and photorealistic radiance fields as well as accurate distance fields for large-scale environments using LiDAR and camera data. Building upon PIN-SLAM's [51] point-based neural map for distance fields and inspired by Scaffold-GS [38], we propose a novel pointbased model that additionally represents a Gaussian splatting radiance field. By enforcing mutual supervision between these fields during incremental mapping, we achieve both improved rendering quality from the radiance field and more accurate distance field for better localization and surface reconstruction.

The main contribution of this paper is a novel LiDAR-visual SLAM system, called PINGS, that incrementally builds continuous SDF and Gaussian splatting radiance fields by exploiting their mutual consistency within a point-based neural map. The distance field and radiance field infered from the elastic neural points enable robust pose estimation while maintaining global consistency through loop closure correction. The compact neural point map can be efficiently stored and loaded from disk, allowing accurate surface mesh reconstruction from the distance field and high-fidelity real-time novel view rendering from the radiance field, as shown in Fig. 1.

In sum, we make four key claims: (i) PINGS achieves better RGB and geometric rendering at novel views by constraining the Gaussian splatting radiance field using the signed distance field; (ii) PINGS builds a more accurate signed distance field for more accurate localization and surface reconstruction by leveraging dense photometric cues from the radiance field; (iii) PINGS enables large-scale globally consistent mapping with loop closures; (iv) PINGS builds a more compact map than previous methods for both radiance and distance fields.

Our open-source implementation of PINGS is publicly available at: https://github.com/PRBonn/PINGS.

## II. Related Work

## A. Point-based Implicit Neural Representation

Robotics has long relied on explicit map representations with discrete primitives like point clouds [87], surfels [3, 73], meshes [66], or voxel grids [22, 46] for core tasks like localization [65] and planning [59].

Recently, implicit neural representations have been proposed to model radiance fields [45] and geometric (occupancy or distance) fields [44, 49, 52] using multi-layer perceptrons (MLP).

These continuous representations offer advantages like compact storage, and better handling of regions with sparse observations or occlusions, while supporting conversion to explicit representations for downstream tasks.

Instead of using a single MLP for the entire scene, recent methods use hybrid representations that combine local feature vectors with a shared shallow MLP. Point-based implicit neural representations [51, 79] store optimizable features in a neural point cloud, which has advantages over grid-based alternatives through its flexible spatial layout and inherent elasticity under transformations for example caused by loop closures.

Point-based implicit neural representations have been used for modeling either radiance fields or distance fields for various applications including differentiable rendering [8, 79], dynamic scene modeling [1], surface reconstruction [34], visual odometry [56, 86], and globally consistent mapping [51]. For example, PIN-SLAM [51] effectively represents local distance fields with neural points for odometry estimation and uses the elasticity of these neural points during loop closure correction.

In this paper, we propose a novel LiDAR-visual SLAM system that is built on top of PIN-SLAM [51] and encodes a Gaussian splatting radiance field within neural points while jointly optimizing it alongside the distance field. Compared to NeRF-based approaches [8, 79], this offers faster novel view rendering suitable for robotics applications.

## B. Gaussian Splatting Radiance Field

NeRF [45] pioneered the use of MLPs to map 3D positions and view directions to color and volume density, encoding radiance fields through volume rendering-based training with posed RGB images. More recently, 3DGS [30] introduced explicit 3D Gaussian primitives to represent the radiance fields, achieving high-quality novel view synthesis. Compared to NeRF-based methods, 3DGS is more efficient by using primitive-based differentiable rasterization [82] instead of raywise volume rendering. The explicit primitives also enables editing and manipulation of the radiance field. These properties make 3DGS promising for robotics applications [2, 26, 35, 42, 43]. However, two main challenges limit its usage: geometric accuracy and scalability for incremental mapping. We discuss the related works addressing geometric accuracy in the following and addressing scalable mapping in Sec. II-C.

While 3DGS achieves high-fidelity photorealistic rendering, it often lacks the geometric accuracy. To tackle this limitation, SuGaR [18] uses a hybrid representation to extract meshes from 3DGS and align the Gaussian primitives with the surface meshes. To address the ambiguity in surface description, another solution is to flatten the 3D Gaussian ellipsoids to 2D disks [11, 23, 25, 85]. The 2D disks gradually align with surfaces during training, enabling more accurate depth and normal rendering. However, extracting surface meshes from these discrete primitives still requires either TSDF fusion [46] with rendered depth or Poisson surface reconstruction [28].

Another line of works [58, 84] model discrete Gaussian opacity as a continuous field, similar to NeRF-based surface reconstruction [70]. Several works [6, 39, 83] jointly train a distance field with 3DGS and align the Gaussian primitives with the zero-level set of the distance field to achieve accurate surface reconstruction. However, these methods rely solely on image rendering supervision for both 3DGS and neural SDF training without direct 3D geometric constraints, leading to ambiguities in textureless or specular regions. The volume rendering-based SDF training also impacts efficiency.

While 3DGS originally uses structure-from-motion point clouds, robotic platforms with LiDAR can initialize primitives directly from LiDAR measurements [10, 21, 78]. Direct depth measurements can further supervise depth rendering to improve geometric accuracy and convergence speed [25, 42].

Our approach uniquely combines geometrically consistent 2D Gaussian disks with a neural distance field supervised by direct LiDAR measurements, enforcing mutual geometric consistency between the representations. This differs from GSFusion [72], which maintains decoupled distance and radiance fields without mutual supervision.

## C. Large-Scale 3D Reconstruction

This paper focuses on online large-scale 3D reconstruction. There have been numerous works for the scalable occupancy or distance field mapping in the past decade, using efficient data structures such as an Octree [22, 93], voxel hashing [33, 48, 94], an VDB [67, 77], or wavelets [53].

Scalable radiance field mapping has also made significant progress recently. For large scale scenes captured by aerial images, recent works [36, 38, 55] demonstrate promising results using level-of-detail rendering and neural Gaussian compression. For driving scenes with short sequences containing hundreds of images, both NeRF-based [54, 81] and 3DGSbased [9, 13, 19, 80, 90, 95] approaches have demonstrated high-fidelity offline radiance field reconstruction, enabling closed-loop autonomous driving simulation [9, 81].

However, radiance field mapping for even larger scenes at ground level with thousands of images remains challenging due to scene complexity and memory constraints. BlockNeRF [61] addresses this by dividing scenes into overlapping blocks, training separate NeRFs per block, and consolidating them during rendering. Similarly, SiLVR [63] employs a submap strategy for scalable NeRF mapping. For 3DGS, hierarchical 3DGS [31] introduces a level-of-detail hierarchy that enables real-time rendering of city-scale scenes. The aforementioned methods require time-consuming structure-from-motion preprocessing and offline divide-and-conquer processing, limiting their applicability for online missions.

While there are several works on incremental mapping and SLAM with NeRF [49, 56, 60] or 3DGS [29, 42, 72, 96], they primarily focus on bounded indoor scenes and struggle with our target scenarios. Our proposed system enables incremental radiance and distance field mapping at the scale of previous offline methods [31, 61], while achieving globally consistent 3D reconstruction through loop closure correction.

## III. OUR APPROACH

Our approach, called PINGS, is a LiDAR-visual SLAM system that jointly builds globally consistent Gaussian splatting radiance fields and distance fields for large-scale scenes.

Notation. In the following, we denote the transformation from coordinate frame $A$ to frame $B$ as $\Upsilon_{B A} \in \operatorname{SE}(3)$, such that point $\boldsymbol{p}_{B}=\Upsilon_{B A} \boldsymbol{p}_{A}$, with rotation $\mathcal{R}_{B A} \in \mathrm{SO}(3)$ and translation $\boldsymbol{t}_{B A} \in \mathbb{R}^{3}$, where the rotation is also parameterized by a unit quaternion $\boldsymbol{q}$. At timestep $t$, each sensor frame $S_{t}$ (LiDAR frame $L_{t}$ or camera frame $C_{t}$ ) is related to the world frame $W$ by pose $\Upsilon_{W S_{t}}$, with $\Upsilon_{W S_{0}}$ fixed as identity. We denote the rotation of a vector $\boldsymbol{v} \in \mathbb{R}^{3}$ by a quaternion $\boldsymbol{q}$ as $\boldsymbol{q} \boldsymbol{v} \boldsymbol{q}^{-1}$ and the multiplication of two quaternions as $\boldsymbol{q}_{1} \boldsymbol{q}_{2}$.

Overview. We assume the robot is equipped with a LiDAR sensor and one or multiple cameras. At each timestep $t$, the input to our system is a LiDAR point cloud $\mathcal{P}=\left\{\boldsymbol{p} \in \mathbb{R}^{3}\right\}$ and $M$ camera images $\mathcal{I}=\left\{\hat{I}_{i} \in \mathbb{R}^{H \times W \times 3} \mid i=1, \ldots, M\right\}$ collected by the robot. We assume the calibration of the LiDAR and cameras to be known but allow for the imperfect synchronization among the sensors. Our system aims to simultaneously estimate the LiDAR pose $\Upsilon_{W L_{t}}$ while updating a point-based implicit neural map $\mathcal{M}$, which models both a SDF and a radiance field, as summarized in Fig. 2.

## A. Point-based Implicit Neural Map Representation

We define our point-based implicit neural map $\mathcal{M}$ as a set of neural points, given by:

$$
\mathcal{M}=\left\{\boldsymbol{m}_{i}=\left(\boldsymbol{x}_{i}, \boldsymbol{q}_{i}, \boldsymbol{f}_{i}^{g}, \boldsymbol{f}_{i}^{o}, \tau_{i}^{c}, \tau_{i}^{n}\right) \mid i=1, \ldots, N\right\}
$$

where each neural point $\boldsymbol{m}_{i}$ is defined in the world frame $W$ by a position $\boldsymbol{x}_{i} \in \mathbb{R}^{3}$ and a quaternion $\boldsymbol{q}_{i} \in \mathbb{R}^{4}$ representing the orientation of its own coordinate frame. Each neural point stores the optimizable geometric feature vector $\boldsymbol{f}_{i}^{g} \in \mathbb{R}^{F_{g}}$ and appearance feature vector $\boldsymbol{f}_{i}^{a} \in \mathbb{R}^{F_{a}}$. In addition, we keep track of each neural point's creation timestep $\tau_{i}^{c}$ and last update timestep $\tau_{i}^{n}$ to determine its active status and associate the neural point with the LiDAR pose $\Upsilon_{W L_{t}}$ at the middle timestep $\tau_{i}=\left\lfloor\left(\tau_{i}^{c}+\tau_{i}^{n}\right) / 2\right\rfloor$ between $\tau_{i}^{c}$ and $\tau_{i}^{n}$, thus allowing direct map manipulation through pose updates.

We maintain a voxel hashing [47] data structure $\mathcal{V}$ with a voxel resolution $v_{p}$ for fast neural point indexing and neighbor search, where each voxel stores at most one active neural point.

During incremental mapping, we dynamically update the neural point map based on point cloud measurements. For each newly measured point $\boldsymbol{p}_{W}$ in the world frame, we check its corresponding voxel in $\mathcal{V}$. If no active neural point exists in that voxel, we initialize a new neural point $\boldsymbol{m}$ with the position $\boldsymbol{x}=\boldsymbol{p}_{W}$, an identity quaternion $\boldsymbol{q}=(1,0,0,0)$, and the feature vectors $\boldsymbol{f}^{g}=\mathbf{0}, \boldsymbol{f}^{o}=\mathbf{0}$. Additionally, we define a local map $\mathcal{M}_{l}$ centered at the current LiDAR position $\boldsymbol{t}_{W L_{t}}$, which contains all active neural points within radius $r_{l}$. To avoid incorporating inconsistent historical observations caused by odometry drift, both map optimization and odometry estimation are operated only within this local map $\mathcal{M}_{l}$. After the map optimization at each timestep, we reassign the local map $\mathcal{M}_{l}$ into the global map $\mathcal{M}$.

Next, we describe how the neural points map $\mathcal{M}$ models both the SDF (Sec. III-B) and the radiance field (Sec. III-C). ![img-1.jpeg](img-1.jpeg)

Fig. 2: Overview of PINGS: We take a stream of LiDAR point clouds $\mathcal{P}$ and camera images $\mathcal{I}$ as input. We initialize a neural point map $\mathcal{M}$ from $\mathcal{P}$ and maintain a training pool of SDF-labeled points $\mathcal{Q}_p$ and recent images $\mathcal{Q}_i$. The map uses a voxel hashing structure $\mathcal{V}$ where each neural point stores geometric features $\mathbf{f}^g$ and appearance features $\mathbf{f}^a$. These features are used to predict SDF values $S(\mathbf{p})$ at an arbitrary position $\mathbf{p}$ and spawn Gaussian primitives $\mathcal{G}$ through MLP decoders. We compute three kind of losses: (1) Gaussian splatting loss $\mathcal{L}_{\text{gs}}$ comparing rendered images through differentiable rasterization and reference images in the training pool, (2) SDF loss $\mathcal{L}_{\text{sdf}}$ comparing predicted SDF and labels of the sampled points in the training pool, and (3) consistency loss $\mathcal{L}_{\text{cons}}$ to align the geometry of both representations. The losses are backpropagated to optimize the neural point features $\mathbf{f}^g$ and $\mathbf{f}^a$. Meanwhile, we estimate LiDAR odometry by aligning the point cloud to current SDF and backpropagate $\mathcal{L}_{\text{gs}}$ to refine the camera poses. The final outputs are LiDAR poses $\mathbf{T}_{WL}$, camera poses $\mathbf{T}_{WC}$, and a compact neural point map $\mathcal{M}$ representing both SDF and Gaussian splatting radiance fields, enabling various robotic applications.

### *B. Neural Signed Distance Field*

For the modeling and online training of a continuous SDF using the neural points, we follow the same strategy as in PIN-SLAM [51] and present a recap in this section.

We model the SDF value $s$ at a query position $\mathbf{p}$ in the world frame $W$ conditioned on its nearby neural points. For each neural point $\mathbf{m}_j$ in the k-nearest neighborhood $\mathcal{N}_p$ of $\mathbf{p}$, we define the relative coordinate $\mathbf{d}_j = \mathbf{q}_j(\mathbf{p} - \mathbf{x}_j) \mathbf{q}_j^{-1}$ denoting $\mathbf{p}$ in the local coordinate system of $\mathbf{m}_j$. Then, we feed the geometric feature vector $\mathbf{f}_j^g$ and the relative coordinate $\mathbf{d}_j$ to a globally shared SDF decoder $D_d$ to predict the SDF $s_j$:

$$s_j = D_d(\mathbf{f}_j^g, \mathbf{d}_j). \tag{2}$$

As shown in Fig. 3 (a), the predicted SDF values $s_j$ of the neighboring neural points at the query position $\mathbf{p}$ are then interpolated as the final prediction $s = S(\mathbf{p})$, given by:

$$S(\mathbf{p}) = \sum_{j \in \mathcal{N}_p} \frac{w_j}{\sum_{k \in \mathcal{N}_p} w_k} s_j, \tag{3}$$

with the interpolation weights $w_j = \|\mathbf{p} - \mathbf{x}_j\|^{-2}$.

To optimize the neural SDF represented by the neural point geometric features $\{\mathbf{f}_j^g\}_{i=1}^N$ and the SDF decoder $D_d$, we sample points along the LiDAR rays around the measured end points and in the free space. We take the projective signed distance along the ray as a pseudo SDF label for each sample point. For incremental learning, we maintain a training data pool $\mathcal{Q}_p$ containing sampled points from recent scans, with a maximum capacity and bounded by a distance threshold from the current robot position. At each timestep, we sample from the training data pool in batches and predict the SDF value at the sample positions. The SDF training loss $\mathcal{L}_{\text{sdf}}$ is formulated as a weighted sum of the binary cross entropy loss term $\mathcal{L}_{\text{bce}}$ and the Eikonal loss term $\mathcal{L}_{\text{eik}}$, given by:

$$\mathcal{L}_{\text{sdf}} = \lambda_{\text{bce}} \mathcal{L}_{\text{bce}} + \lambda_{\text{eik}} \mathcal{L}_{\text{eik}}. \tag{4}$$

The loss term $\mathcal{L}_{\text{bce}}$ applies a soft supervision on the SDF values by comparing the sigmoid activation of both the predictions and the pseudo labels. The Eikonal loss term $\mathcal{L}_{\text{eik}}$ regularizes the SDF gradients by enforcing the Eikonal constraint [17], which requires unit-length gradients $\|\nabla S(\mathbf{x})\| = 1$ for the sampled points. For more details regarding the SDF training, we refer readers to Pan et al. [51].

The incrementally built neural SDF map can then be used for LiDAR odometry estimation and surface mesh extraction.

### *C. Neural Gaussian Splatting Radiance Field*

We use camera image streams $\mathcal{I}$ to construct a radiance field by spawning Gaussian primitives from our neural point map $\mathcal{M}$ and optimizing $\mathcal{M}$ via differentiable rasterization.

**Neural Point-based Gaussian Spawning.** Inspired by Scaffold-GS [38], we use our neural points as anchor points for spawning Gaussian primitives, see Fig. 3 (b). For each neural point $\mathbf{m}$ lying within the current camera frustum, we spawn $K$ Gaussian primitives by feeding its feature vectors ($\mathbf{f}^g$, $\mathbf{f}^a$) through globally shared MLP decoders. We parameterize each spawned Gaussian primitive $\mathbf{g}$ with its position $\mathbf{\mu} \in \mathbb{R}^3$ in the world frame, rotation $\mathbf{r} \in \mathbb{R}^4$ in the form of a unit quaternion, scale $\mathbf{s} \in \mathbb{R}^3$, opacity $\alpha \in [-1, 1]$, and RGB color $\mathbf{c} \in [0, 1]^3$. ![img-2.jpeg](img-2.jpeg)

Fig. 3: Example of neural point-based SDF prediction, Gaussian primitives spawning, and the geometric consistency of PINGS: (a) SDF prediction at a query point through weighted interpolation of predictions from neighboring neural points. (b) Neural points spawning multiple Gaussian primitives to compose the radiance field. (c) Example of an accurate SDF but geometrically inaccurate radiance field with 3D Gaussian ellipsoids in regions with dense LiDAR coverage but sparse camera views, weak texture, or poor lighting. (d) Example of a geometrically accurate radiance field but inaccurate SDF in regions with rich visual data but sparse LiDAR measurements. (e) Our solution: flattening 3D Gaussian ellipsoids to surfels and enforcing geometric consistency by aligning surfel centers with the SDF zero-level set and aligning surfel normals with SDF gradients, resulting in accurate geometry for both fields.

Each neural point spawns Gaussian primitives in its local coordinate frame defined by its position *x* and orientation *q*. The world-frame position *µ<sub>i</sub>* of each spawned primitive is:

$$
\{\boldsymbol{\mu}_i = \boldsymbol{q} \boldsymbol{o}_i \boldsymbol{q}^{-1} + \boldsymbol{x} \mid \boldsymbol{o}_i \in D_o(\boldsymbol{f}^g)\}_{i=1}^K, \tag{5}
$$

where *D<sub>o</sub>* is the offset decoder that maps the geometric feature *f<sup>g</sup>* to a set of *K* local offsets {*o<sub>i</sub>*}<sub>i=1</sub><sup>*K*</sup>, which are then transformed into the world frame through quaternion rotation and translation. Likewise, the rotation *r<sub>i</sub>* of each spawned Gaussian primitive is predicted by the rotation decoder *D<sub>r</sub>* and then rotated by quaternion *q* as:

$$
\{\boldsymbol{r}_i = \boldsymbol{q} \hat{\boldsymbol{r}}_i \mid \hat{\boldsymbol{r}}_i \in D_r(\boldsymbol{f}^g)\}_{i=1}^K. \tag{6}
$$

The scale decoder *D<sub>s</sub>* predicts each primitive's scale *s<sub>i</sub>* as:

$$
\{\boldsymbol{s}_i\}_i=1^K = D_s(\boldsymbol{f}^g). \tag{7}
$$

We predict opacity values *α* in the range [−1, 1] and treat only Gaussian primitives with positive opacity as being valid. To adaptively control spatial density of Gaussian primitives based on viewing distance, we feed the geometric feature *f<sup>g</sup>* and the view distance *δ<sub>v</sub>* = ||*x* − *t<sub>WC</sub>*||<sub>2</sub> into the opacity decoder *D<sub>α</sub>*. This implicitly encourages the network to predict fewer valid Gaussians for distant points and more for nearby points, reducing computational load. The opacity value *α<sub>i</sub>* for each Gaussian primitive is predicted as:

$$
\{\alpha_i\}_i=1^K = D_{\alpha}(\boldsymbol{f}^g, \delta_v). \tag{8}
$$

For view-dependent color prediction, we take a different approach than the spherical harmonics used in 3DGS [30]. We feed the appearance feature *f<sup>a</sup>* and the view direction *d<sub>v</sub>* = (*x* − *t<sub>WC</sub>*)/*δ<sub>v</sub>* to the color decoder *D<sub>c</sub>* to predict the color *c<sub>i</sub>* of each Gaussian primitive, given by:

$$
\{\boldsymbol{c}_i\}_i=1^K = D_c(\boldsymbol{f}^a, \boldsymbol{q}^{-1} \boldsymbol{d}_v \boldsymbol{q}), \tag{9}
$$

where the view direction *d<sub>v</sub>* is also transformed into the local coordinate system of the neural point.

Note that we treat position, rotation, scale, and opacity as geometric attributes of a Gaussian primitive, using the geometric feature *f<sup>g</sup>* for their prediction, while using the appearance feature *f<sup>a</sup>* to predict color.

**Gaussian Splatting Rasterization.** We gather all the valid Gaussians primitives *G* [30] spawned at the current viewpoint:

$$
\mathcal{G} = \{\boldsymbol{g}_i = (\boldsymbol{\mu}_i, \boldsymbol{r}_i, \boldsymbol{s}_i, \boldsymbol{c}_i, \alpha_i) \mid i = 1, \dots, N_g\}. \tag{10}
$$

The distribution of each Gaussian primitive *g<sub>i</sub>* in the world frame is represented as:

$$
\mathcal{N}(\boldsymbol{x}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i) = \exp\left(-\frac{1}{2}(\boldsymbol{x} - \boldsymbol{\mu}_i)^{\top} \boldsymbol{\Sigma}_i^{-1}(\boldsymbol{x} - \boldsymbol{\mu}_i)\right), \tag{11}
$$

where the covariance matrix *Σ<sub>i</sub>* is reparameterized as:

$$
\boldsymbol{\Sigma}_i = \boldsymbol{R}(\boldsymbol{r}_i) \boldsymbol{S}(\boldsymbol{s}_i) \boldsymbol{S}(\boldsymbol{s}_i)^{\top} \boldsymbol{R}(\boldsymbol{r}_i)^{\top}, \tag{12}
$$

where *R*(*r<sub>i</sub>*) ∈ *SO*(3) is the rotation matrix derived from the quaternion *r<sub>i</sub>* and *S*(*s<sub>i</sub>*) = diag(*s<sub>i</sub>*) ∈ ℝ<sup>3×3</sup> is the diagonal scale matrix composed of the scale *s<sub>i</sub>* on each axis.

Using a tile-based rasterizer [98], we project the Gaussian primitives to the 2D image plane and sort them according to depth efficiently. The projected Gaussian distribution is:

$$
\boldsymbol{\mu}' = \pi(\boldsymbol{\Upsilon}_{CW} \boldsymbol{\mu}), \quad \boldsymbol{\Sigma}' = \mathcal{J} \boldsymbol{W} \boldsymbol{\Sigma} \boldsymbol{W}^{\top} \mathcal{J}^{\top}, \tag{13}
$$

where *µ<sup>0</sup>* and *Σ<sup>0</sup>* are the projected mean and covariance, *π* denotes the perspective projection, *J* is the Jacobian of the projective transformation, and *W* is the viewing transformation deduced from current camera pose *T<sub>WC</sub>*. The rendered RGB image *I* at each pixel *u* is computed via alpha blending:

$$
I(\boldsymbol{u}) = \sum_{i \in \mathcal{G}(\boldsymbol{u})} w_i \boldsymbol{c}_i, \tag{14}
$$

where the weight *w<sub>i</sub>* of each of the depth-sorted Gaussian primitives *G*(*u*) covering pixel *u* is given by:

$$
w_i = T_i \sigma_i, \ T_i = \prod_{j=1}^{i-1} (1 - \sigma_j), \ \sigma_i = \mathcal{N}(\boldsymbol{u}; \boldsymbol{\mu}_i' \boldsymbol{\Sigma}_i') \alpha_i, \tag{15}
$$

where *σ<sub>i</sub>* is the projected opacity of the *i*-th Gaussian primitive, computed using the 2D Gaussian density function *N*(*u*; *µ<sub>i</sub>*, *Σ<sub>i</sub><sup>0</sup>*) evaluated at pixel *u* with the projected mean *µ<sub>i</sub><sup>0</sup>* and covariance *Σ<sub>i</sub><sup>0</sup>*.

**Gaussian Surfels Training.** To achieve accurate and multiview consistent geometry, we adopt Gaussian Surfels [11], a state-of-the-art 2DGS representation [23], by flattening 3D Gaussian ellipsoids into 2D disks (last dimension of scale $s^{z}=0$). For each pixel $\boldsymbol{u}$, we compute the surfel depth $d(\boldsymbol{u})$ as the ray-disk intersection distance, and obtain the normal $\boldsymbol{n}$ as the third column of the rotation matrix $\boldsymbol{R}(\boldsymbol{r})$. Using alpha blending, we render the depth map $\boldsymbol{D}$ and the normal map $\boldsymbol{N}$ using the weights $w_{i}$ calculated in Eq. (15):

$$
D(\boldsymbol{u})=\sum_{i \in \mathcal{G}(\boldsymbol{u})} w_{i} d_{i}(\boldsymbol{u}), \quad N(\boldsymbol{u})=\sum_{i \in \mathcal{G}(\boldsymbol{u})} w_{i} \boldsymbol{n}_{i}
$$

Given the training view with the RGB image $\widehat{I}$ and the sparse depth map $\widehat{D}$ projected from the LiDAR point cloud, we define the Gaussian splatting loss $\mathcal{L}_{\text {gs }}$ combining the photometric rendering $\mathcal{L}_{\text {photo }}$, depth rendering $\mathcal{L}_{\text {depth }}$, and area regularization $\mathcal{L}_{\text {area }}$ terms, given by:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{gs}} & =\lambda_{\text {photo }} \mathcal{L}_{\text {photo }}+\lambda_{\text {depth }} \mathcal{L}_{\text {depth }}+\lambda_{\text {area }} \mathcal{L}_{\text {area }} \\
\mathcal{L}_{\text {photo }} & =0.8 \cdot L_{1}(I, \widehat{I})+0.2 \cdot L_{\text {ssim }}(I, \widehat{I}) \\
\mathcal{L}_{\text {depth }} & =L_{1}(D, \widehat{D}) \\
\mathcal{L}_{\text {area }} & =\sum_{\boldsymbol{g}_{i} \in \mathcal{G}} s_{i}^{x} \cdot s_{i}^{y}
\end{aligned}
$$

where $L_{1}$ is the L1 loss, $L_{\text {ssim }}$ is the structural similarity index measure (SSIM) loss [71], $s_{i}^{x}$ and $s_{i}^{y}$ are the scales of the Gaussian surfel $\boldsymbol{g}_{i}$. The area loss term $\mathcal{L}_{\text {area }}$ encourages minimal overlap among the surfels covering the surface.

To handle inaccurate camera poses resulting from imperfect LiDAR odometry and camera-LiDAR synchronization, we jointly optimize the camera poses on a manifold during radiance field training [42]. We also account for real-world lighting variations by optimizing per-frame exposure parameters [31].

## D. Joint Optimization with Geometric Consistency

To enforce mutual alignment between the surfaces represented by the SDF and Gaussian splatting radiance field, we futhermore propose to jointly optimize the geometric consistency. This joint optimization helps resolve geometric ambiguities in the radiance field through the direct surface description of SDF, while simultaneously refining SDF's accuracy in regions with sparse LiDAR measurements using the dense photometric cues and multi-view consistency from the radiance field, see Fig. 3 (c), (d), and (e).

For each sampled Gaussian surfel, we randomly sample points along its normal direction $\boldsymbol{n}$ from the center $\boldsymbol{\mu}$ with random offsets $\epsilon \sim U\left(-\epsilon_{\max }, \epsilon_{\max }\right)$. We enforce geometric consistency between the SDF and Gaussian surfels through a two-part consistency loss $\mathcal{L}_{\text {cons }}$, given by:

$$
\begin{aligned}
& \mathcal{L}_{\text {cons }}=\lambda_{\text {cons }}^{\mathrm{d}} \mathcal{L}_{\text {cons }}^{\mathrm{d}}+\lambda_{\text {cons }}^{\mathrm{v}} \mathcal{L}_{\text {cons }}^{\mathrm{v}} \\
& \mathcal{L}_{\text {cons }}^{\mathrm{d}}=\sum_{\boldsymbol{g}_{i} \in \mathcal{G}}\left|S\left(\boldsymbol{\mu}_{i}+\epsilon_{i} \boldsymbol{n}_{i}\right)-\epsilon_{i}\right| \\
& \mathcal{L}_{\text {cons }}^{\mathrm{v}}=\sum_{\boldsymbol{g}_{i} \in \mathcal{G}}\left(1-\frac{\nabla S\left(\boldsymbol{\mu}_{i}+\epsilon_{i} \boldsymbol{n}_{i}\right)^{\top} \boldsymbol{n}_{i}}{\left\|\nabla S\left(\boldsymbol{\mu}_{i}+\epsilon_{i} \boldsymbol{n}_{i}\right)\right\|}\right)
\end{aligned}
$$

where $\mathcal{L}_{\text {cons }}^{\mathrm{d}}$ enforces SDF values to match sampled offsets via an L1 loss, and $\mathcal{L}_{\text {cons }}^{\mathrm{v}}$ aligns SDF gradients $\nabla S$ at the sampled points with surfel normals $\boldsymbol{n}$ using cosine distance.

We define the total loss $\mathcal{L}$ given by the sum of the SDF loss $\mathcal{L}_{\text {sdf }}$ in Eq. (4), Gaussian splatting loss $\mathcal{L}_{\mathrm{gs}}$ in Eq. (17), and the geometric consistency loss $\mathcal{L}_{\text {cons }}$ in Eq. (21):

$$
\mathcal{L}=\mathcal{L}_{\mathrm{sdf}}+\mathcal{L}_{\mathrm{gs}}+\mathcal{L}_{\mathrm{cons}}
$$

We jointly optimize the neural point features $\left\{\boldsymbol{f}_{i}^{g}, \boldsymbol{f}_{i}^{o}\right\}_{i=1}^{N}$, decoder parameters, camera poses, and exposure correction parameters to minimize the total loss $\mathcal{L}$.

## E. PINGS LiDAR-Visual SLAM System

We devise a LiDAR-visual SLAM system called PINGS using the proposed map representation, built on top of the LiDAR-only PIN-SLAM [51] system. PINGS alternates between two main steps: (i) mapping: incremental learning of the local neural point map $\mathcal{M}_{l}$, which jointly models the SDF and Gaussian splatting radiance field, and (ii) localization: odometry estimation using the learned SDF. In addition, loop closure detection and pose graph optimization run in parallel.

We initialize PINGS with 600 iterations of SDF training using only the first LiDAR scan. At subsequent timesteps, we jointly train the SDF and radiance field for 100 iterations. To prevent catastrophic forgetting during incremental mapping, we freeze decoder parameters after 30 timesteps and only update neural point features. We found the decoders converge on learning the interpretation capability within these 30 frames.

We maintain sliding window-like training pools $\mathcal{Q}_{p}$ and $\mathcal{Q}_{i}$ containing SDF-labeled sample points and image data whose view frustum overlaps with the local map $\mathcal{M}_{l}$, respectively. Each training iteration samples one image from $\mathcal{Q}_{i}$ and 8192 points from $\mathcal{Q}_{p}$ for optimization.

We estimate LiDAR odometry by aligning each new scan to the SDF's zero level set using an efficient Gauss-Newton optimization [74] that requires only SDF values and gradients queried at source point locations, eliminating the need for explicit point correspondences. Initial camera poses are derived from the LiDAR odometry and extrinsic calibration, then refined via gradient descent during the radiance field optimization to account for imperfect camera-LiDAR synchronization, as described in Sec. III-C.

In line with PIN-SLAM, we detect loop closures using the layout and features of the local neural point map. We then conduct pose graph optimization to correct the drift of the LiDAR odometry and get globally consistent poses. We move the neural points along with their associated LiDAR frames to keep a globally consistent map. Suppose $\Delta \mathrm{T}$ is the pose correction matrix of LiDAR frame $L_{i}$ after pose graph optimization, we update the position $\boldsymbol{x}$ and orientation $\boldsymbol{q}$ of each neural point associated with $L_{i}$ as:

$$
\boldsymbol{x} \leftarrow \Delta \mathrm{~T} \boldsymbol{x}, \quad \boldsymbol{q} \leftarrow \Delta \boldsymbol{q} \boldsymbol{q}
$$

where $\Delta \boldsymbol{q}$ is the rotation part of $\Delta \mathrm{T}$ in the form of a quaternion. Since the positions, rotations, and colors of the spawned Gaussian primitives are predicted in the local frames of their anchor neural points, see Eq. (5), Eq. (6), and Eq. (9), they automatically transform with their anchor neural points, thus maintaining the global consistency of the radiance field.

PINGS aims to build static distance and radiance fields without artifacts from dynamic objects. Since measured points with large SDF values in stable free space likely correspond to dynamic objects [57], we identify neural points representing dynamic objects through SDF thresholding. We disable Gaussian primitive spawning for these points, effectively preventing dynamic objects from being rendered from the radiance field.

## IV. EXPERIMENTAL EVALUATION

The main focus of this paper is an approach for LiDAR-visual SLAM that unifies Gaussian splatting radiance fields and signed distance fields by leveraging their mutual consistency within a point-based implicit neural map representation.

We present our experiments to show the capabilities of our method called PINGS. The results of our experiments support our key claims, which are: (i) PINGS achieves better RGB and geometric rendering at novel views by constraining the Gaussian splatting radiance field using the SDF; (ii) PINGS builds a more accurate SDF for more accurate localization and surface reconstruction by leveraging dense photometric cues from the radiance field; (iii) PINGS enables large-scale globally consistent mapping with loop closures; (iv) PINGS builds a more compact map than previous methods for both radiance and distance fields.

### A. Experimental Setup

1) Datasets: We evaluate PINGS on self-collected in-house car datasets and the Oxford Spires dataset [64]. Our in-house car datasets were collected using a robot car equipped with four Basler Ace cameras providing $360^{\circ}$ visual coverage and an Ouster OS1-128 LiDAR ($45^{\circ}$ vertical FOV, 128 beams) mounted horizontally, both operating at 10 Hz . We calibrate the LiDAR-camera system using the method proposed by Wiesmann et al. [75] and generate reference poses through offline LiDAR bundle adjustment [76], incorporating RTKGNSS data, point cloud alignment as well as constraints from precise geo-referenced terrestrial laser scans.

We evaluate the SLAM localization accuracy and scalability of PINGS on two long sequences from our dataset: a 5 km sequence with around 10,000 LiDAR scans and 40,000 images, and a second sequence which is a bit shorter. Both sequences traverse the same area in opposite directions on the same day. For better quantitative evaluation of the radiance field mapping quality, we select five subsequences of 150 LiDAR scans and 600 images each, as shown in Fig. 4. Having sequences captured in opposite driving directions and lane-level lateral displacement allows us to evaluate novel view rendering from substantially different viewpoints from the training views (out-of-sequence testing views), which is a critical capability for downstream tasks such as planning and simulation.

We evaluate surface reconstruction accuracy on the Oxford Spires dataset [64], which provides a millimeter-accurate reference map from a Leica RTC360 terrestrial laser scanner. The data was collected using a handheld system equipped with three global-shutter cameras and a 64-beam LiDAR.

2) Parameters and Implementation Details: For mapping parameters, we set the local map radius $r_{l}$ to 80 m , voxel resolution $v_{p}$ to 0.3 m , and maximum sample offset for consistency loss $\epsilon_{\text{max}}$ to $0.5~{}{v_{p}}$. The training data pool $\mathcal{Q}_{d}$ has a capacity of $2 \cdot 10^{7}$ SDF-labeled sample points, and $\mathcal{Q}_{i}$ has a capacity of 200 images. During map optimization, we use Adam [32] with learning rates of 0.002 for neural point features, 0.001 for decoder, camera poses and exposure corrections parameters. The neural point feature dimensions $F_{g}$ and $F_{a}$ are set to 32 and 16 , respectively. All decoders use shallow MLPs with one hidden layer of 128 neurons. Each neural point spawns $K=8$ Gaussian primitives. For decoder activations, we use sigmoid for SDF decoder $D_{d}$ and color decoder $D_{c}$, tanh for offset decoder $D_{o}$ and opacity decoder $D_{\alpha}$, and exponential for scale decoder $D_{s}$. The Gaussian spawning offset is scaled to $\left[-2 v_{p}, 2 v_{p}\right]$, and the scale output is clamped to a maximum of $2 v_{p}$. The rotation decoder $D_{r}$ output is normalized to valid unit quaternions. The weights for different loss terms are set to: $\lambda_{\text{bce}}=1.0, \lambda_{\text{eik}}=0.5, \lambda_{\text{photo}}=1.0, \lambda_{\text{depth}}=0.01$, $\lambda_{\text{area}}=0.001$, and $\lambda_{\text{cons}}^{d}=\lambda_{\text{cons}}^{v}=0.02$.

For training and testing, we use image resolutions of $512 \times$ 1,032 for the in-house car dataset and $540 \times 720$ for the Oxford Spires dataset. The experiments are carried out on a single NVIDIA A6000 GPU.

### B. Novel View Rendering Quality Evaluation

We evaluate novel view rendering quality on five subsequences from the in-house car dataset. For quantitative evaluation, we employ standard metrics: PSNR [45], SSIM [71], and LPIPS [88] to assess photorealism, along with Depth-L1 error to measure geometric accuracy. We compute these metrics TABLE I: Quantitative comparison of rendering quality on the in-house car dataset. We evaluate rendering photorealism using PSNR, SSIM, and LPIPS metrics, and geometric accuracy using Depth-L1 error (in m). Best results are shown in bold, second best are underscored.

| Sequence | Method | In-Sequence Testing View | | | | Out-of-Sequence Testing View | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | PSNR $\uparrow$ | SSIM $\uparrow$ | LPIPS $\downarrow$ | Depth-L1 $\downarrow$ | PSNR $\uparrow$ | SSIM $\uparrow$ | LPIPS $\downarrow$ | Depth-L1 $\downarrow$ |
| Church | 3DGS | 18.02 | 0.62 | 0.52 | 1.45 | 16.37 | 0.59 | 0.52 | 1.22 |
| | GSS | 18.04 | 0.63 | 0.51 | 1.37 | 16.44 | 0.60 | 0.52 | 1.25 |
| | Neural Point + 3DGS | 20.89 | 0.71 | 0.41 | 0.80 | 19.56 | 0.70 | 0.42 | 0.78 |
| | Neural Point + GGS | 22.56 | 0.75 | 0.36 | 0.43 | 20.48 | 0.74 | 0.38 | 0.47 |
| | PINGS (Ours) | 22.93 | 0.78 | 0.33 | 0.43 | 20.79 | 0.76 | 0.34 | 0.46 |
| Residential Area | 3DGS | 17.60 | 0.58 | 0.52 | 2.22 | 14.63 | 0.53 | 0.56 | 2.78 |
| | GSS | 17.56 | 0.59 | 0.51 | 2.26 | 14.80 | 0.54 | 0.55 | 2.68 |
| | Neural Point + 3DGS | 21.10 | 0.71 | 0.38 | 0.89 | 18.34 | 0.65 | 0.42 | 0.93 |
| | Neural Point + GSS | 22.33 | 0.73 | 0.35 | 0.53 | 19.31 | 0.69 | 0.38 | 0.69 |
| | PINGS (Ours) | 22.67 | 0.77 | 0.30 | 0.53 | 19.48 | 0.71 | 0.34 | 0.68 |
| Street | 3DGS | 16.39 | 0.56 | 0.55 | 2.09 | 15.73 | 0.57 | 0.53 | 2.30 |
| | GSS | 16.85 | 0.59 | 0.53 | 1.87 | 16.01 | 0.59 | 0.52 | 2.15 |
| | Neural Point + 3DGS | 19.74 | 0.68 | 0.42 | 0.81 | 18.02 | 0.64 | 0.44 | 0.79 |
| | Neural Point + GSS | 22.13 | 0.75 | 0.35 | 0.29 | 19.09 | 0.69 | 0.40 | 0.49 |
| | PINGS (Ours) | 22.45 | 0.78 | 0.32 | 0.28 | 19.34 | 0.71 | 0.37 | 0.47 |
| Campus | 3DGS | 17.38 | 0.57 | 0.52 | 2.70 | 14.88 | 0.49 | 0.58 | 3.60 |
| | GSS | 17.34 | 0.59 | 0.51 | 2.36 | 14.96 | 0.51 | 0.57 | 3.42 |
| | Neural Point + 3DGS | 20.04 | 0.67 | 0.40 | 1.06 | 17.83 | 0.60 | 0.44 | 1.19 |
| | Neural Point + GSS | 21.82 | 0.72 | 0.35 | 0.65 | 18.71 | 0.64 | 0.41 | 0.79 |
| | PINGS (Ours) | 22.40 | 0.76 | 0.31 | 0.64 | 18.91 | 0.66 | 0.38 | 0.80 |
| Roundabout | 3DGS | 21.20 | 0.71 | 0.39 | 0.87 | 18.97 | 0.69 | 0.40 | 0.85 |
| | GSS | 21.74 | 0.72 | 0.38 | 0.55 | 19.10 | 0.69 | 0.40 | 0.60 |
| | Neural Point + 3DGS | 21.44 | 0.75 | 0.35 | 0.72 | 19.23 | 0.72 | 0.37 | 0.78 |
| | Neural Point + GSS | 23.54 | 0.82 | 0.28 | 0.47 | 20.22 | 0.78 | 0.31 | 0.55 |
| | PINGS (Ours) | 23.45 | 0.82 | 0.28 | 0.47 | 20.23 | 0.77 | 0.30 | 0.54 |

![img-3.jpeg](img-3.jpeg)

Fig. 5: Qualitative comparison of rendering quality on the in-house car dataset. Left: Bird's eye view rendering of the Church scene, showing the training view trajectory (black line) and the test viewpoint for comparison (green camera frustum). Right: RGB and normal map renderings from different methods at the test viewpoint, with detailed comparison of curb and sidewalk rendering in the highlighted box.

for both in-sequence and out-of-sequence testing views. We consider the following methods for comparison:

- 3DGS [30]: An incremental training variant of 3DGS initialized with LiDAR measurements and supervised with the depth rendering loss $\mathcal{L}_{\text{depth}}$ as defined in Sec. III-C.
- GSS [11]: Gaussian surfels splatting, a state-of-the-art 2D Gaussian representation using surfels instead of 3D ellipsoids. It uses the same setup as the 3DGS baseline but adds the depth-normal consistency loss from GSS [11].
- Neural Point+3DGS: Our extension of Scaffold-GS [38] that enables incremental training and adds supervision of neural point geometric features through the SDF branch, as detailed in Sec. III-C.
- Neural Point+GSS: A variant that replaces the 3D Gaussian in Neural Point+3DGS with 2D Gaussian surfels.
- PINGS: Our complete framework that extends Neural Point+GSS by introducing geometric consistency loss $\mathcal{L}_{\text{cons}}$ into the joint training, as described in Sec. III-D.

For fair comparison, we disable the localization part and use the ground truth pose for all the compared methods. For 3DGS and GSS, we initialize their Gaussian primitive density to match the total number of Gaussians spawned by PINGS.

We show the quantitative comparison on five sequences in Tab. I as well as show the qualitative comparison of the RGB and normal map rendering results on the church scene at a novel view in Fig. 5. Our method PINGS achieves TABLE II: Quantitative evaluation of surface reconstruction quality on the Oxford-Spires dataset. We use the metrics include accuracy error (in m), completeness error (in m), and Chamfer distance (in m), as well as precision, recall and F-score (with 0.1 m threshold). $\dagger$ denotes methods requiring offline batch processing. Best results are shown in bold, second best are underscored.

| Sequence | Method | Accuracy $\downarrow$ | Completeness $\downarrow$ | Chamfer Distance $\downarrow$ | Precision $\uparrow$ | Recall $\uparrow$ | F-score $\uparrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Blenheim Palace 05 | OpenMVS^{†} | 0.126 | 1.045 | 0.586 | 0.574 | 0.381 | 0.458 |
| | Nerfacto^{†} | 0.302 | 0.676 | 0.489 | 0.388 | 0.257 | 0.309 |
| | GSS | 0.204 | 0.254 | 0.229 | 0.271 | 0.261 | 0.266 |
| | VDB-Fusion | 0.098 | 0.123 | 0.111 | 0.646 | 0.746 | 0.692 |
| | PIN-SLAM | 0.078 | 0.136 | 0.107 | 0.768 | 0.712 | 0.739 |
| | PINGS (Ours) | 0.072 | 0.133 | 0.102 | 0.794 | 0.726 | 0.758 |
| Christ Church 02 | OpenMVS^{†} | 0.046 | 5.381 | 2.714 | 0.886 | 0.266 | 0.410 |
| | Nerfacto^{†} | 0.219 | 4.435 | 2.327 | 0.532 | 0.254 | 0.343 |
| | GSS | 0.174 | 0.292 | 0.233 | 0.407 | 0.301 | 0.346 |
| | VDB-Fusion | 0.098 | 0.243 | 0.171 | 0.655 | 0.524 | 0.582 |
| | PIN-SLAM | 0.069 | 0.252 | 0.160 | 0.812 | 0.497 | 0.617 |
| | PINGS (Ours) | 0.067 | 0.251 | 0.159 | 0.815 | 0.502 | 0.622 |
| Keble College 04 | OpenMVS^{†} | 0.067 | 0.342 | 0.205 | 0.918 | 0.718 | 0.806 |
| | Nerfacto^{†} | 0.137 | 0.150 | 0.144 | 0.654 | 0.709 | 0.680 |
| | GSS | 0.171 | 0.162 | 0.167 | 0.424 | 0.518 | 0.466 |
| | VDB-Fusion | 0.103 | 0.101 | 0.102 | 0.639 | 0.821 | 0.719 |
| | PIN-SLAM | 0.096 | 0.108 | 0.102 | 0.701 | 0.793 | 0.744 |
| | PINGS (Ours) | 0.093 | 0.106 | 0.099 | 0.705 | 0.799 | 0.749 |
| Observatory Quarter 01 | OpenMVS^{†} | 0.048 | 0.622 | 0.335 | 0.902 | 0.618 | 0.734 |
| | Nerfacto^{†} | 0.197 | 0.398 | 0.298 | 0.587 | 0.598 | 0.592 |
| | GSS | 0.179 | 0.184 | 0.181 | 0.377 | 0.443 | 0.407 |
| | VDB-Fusion | 0.123 | 0.109 | 0.116 | 0.573 | 0.737 | 0.645 |
| | PIN-SLAM | 0.105 | 0.129 | 0.117 | 0.654 | 0.677 | 0.665 |
| | PINGS (Ours) | 0.102 | 0.124 | 0.113 | 0.659 | 0.705 | 0.681 |

![img-4.jpeg](img-4.jpeg)

Fig. 6: Qualitative results of the surface mesh reconstruction by PINGS on the Oxford-Spires dataset. The meshes are extracted using marching cubes algorithm from the SDF with a resolution of 0.1 m.

Superior performance in both photorealistic rendering quality and depth rendering accuracy on the in-house car dataset, and consistently outperforms the baselines for both in-sequence and out-of-sequence testing views. Analysis of the results reveals several insights: (i) The adoption of GSS over 3DGS leads to improved geometric rendering quality and enhanced out-of-sequence rendering photorealism; (ii) Our approach of spawning Gaussians from neural points and jointly training with the distance field provides better optimization control and reduces floating Gaussians in free space, resulting in superior rendering quality; (iii) The addition of geometric consistency constraints from SDF enables better surface alignment of Gaussian surfels, further enhancing both geometric accuracy and photorealistic rendering quality, as evidenced by the smoother normal maps produced by PINGS compared to Neural Point+GSS. These improvements are less significant in the Roundabout scene, where the dense viewpoint coverage from the vehicle's circular trajectory provides strong multi-view constraints, reducing the benefit of additional geometric constraints from the SDF.

In sum, this experiment validates that PINGS achieves better RGB and geometric rendering at novel views by constraining the Gaussian splatting radiance field using the SDF.

### V-C Surface Reconstruction Quality Evaluation

We evaluate surface reconstruction quality on four sequences from the Oxford-Spires dataset. We follow the benchmark [64] to report the metrics including accuracy error, completeness error, and Chamfer distance, as well as precision, recall and F-score calculated with a threshold of 0.1 m. We compare the performance of PINGS with five state-of-the-art methods, including OpenMVS [5], Nerfacto [62], GSS [11], VDB-Fusion [67], and PIN-SLAM [51]. To ensure fair comparison of geometric mapping quality, we disable the localization modules of PIN-SLAM and PINGS and use ground truth poses across all methods. For GSS, after completing the radiance field mapping, we render depth maps at each frame and apply TSDF fusion [67] for mesh extraction. Results of the vision-based offline processing methods (OpenMVS and Nerfacto) are taken from the benchmark [64]. For the remaining methods (GSS, VDB-Fusion, PIN-SLAM, and PINGS), TABLE III: Localization performance comparison of PINGS against state-of-the-art odometry/SLAM methods on the in-house car dataset. We report average relative translation error (ARTE) [%] and absolute trajectory error (ATE) [m]. Odometry methods are shown above the midrule, SLAM methods below. Best results are shown in bold, second best are underscored.

| Method | Seq. 1 (5.0 km) | | Seq. 2 (3.7 km) | |
| | ARTE [%] $\downarrow$ | ATE [m] $\downarrow$ | ARTE [%] $\downarrow$ | ATE [m] $\downarrow$ |
| --- | --- | --- | --- | --- |
| F-LOAM [69] | 1.96 | 28.52 | 1.93 | 27.00 |
| KISS-ICP [68] | 1.49 | 8.17 | 1.38 | 8.22 |
| PIN odometry [51] | 0.95 | 4.51 | 0.98 | 5.64 |
| PINGS odometry | 0.73 | 5.17 | 0.59 | 4.78 |
| SuMa [3] | 5.55 | 39.90 | 4.42 | 44.78 |
| MULLS [50] | 2.23 | 40.37 | 1.64 | 33.82 |
| PIN-SLAM [51] | 1.00 | 3.17 | 0.98 | 4.44 |
| PINGS (Ours) | 0.68 | 1.99 | 0.58 | 3.47 |

we extract surface meshes from their SDFs using marching cubes [37] at a resolution of 0.1 m.

We show the qualitative results of PINGS on the four sequences in Fig. 6. Quantitative comparisons in Tab. II demonstrate that PINGS achieves superior performance, particularly in terms of Chamfer distance and F-score metrics. Notably, when using identical neural point resolution, PINGS consistently outperforms PIN-SLAM across all metrics through its joint optimization of the radiance field and geometric consistency constraints. This improvement validates that incorporating dense photometric cues and multi-view consistency from the radiance field improves the SDF accuracy, ultimately enabling surface mesh reconstruction with higher quality.

### IV-D SLAM Localization Accuracy Evaluation

We compare the pose estimation performance of PINGS against state-of-the-art LiDAR odometry/SLAM systems on two full sequences of the in-house car dataset. The compared methods include F-LOAM [69], KISS-ICP [68], SuMa [3], MULLS [50], and PIN-SLAM [51]. For evaluation metrics, we use average relative translation error (ARTE) [15] to assess odometry drift and absolute trajectory error (ATE) [89] to measure the global pose estimation accuracy. The results shown in Tab. III demonstrate that PINGS achieves both lower odometry drift and superior global localization accuracy than the compared approaches. Compared to PIN-SLAM, the improvement stems from the refined SDF obtained through joint optimization with the radiance field and geometric consistency constraints. The improved SDF leads to more accurate LiDAR odometry and relocalization during loop closure correction.

### IV-E Large-Scale Globally Consistent Mapping

Fig. 1 demonstrates the globally consistent SLAM capabilities of PINGS on a challenging 5 km sequence from our in-house car dataset. In Fig. 7, we show the effect of loop closure correction. Without loop closure correction, odometry drift accumulates over time, causing neural points to be inconsistently placed when revisiting previously mapped regions. This results in visual artifacts in the radiance field rendering, such as duplicate objects and trees incorrectly appearing on the road. After conducting loop closure correction and updating the map, both the neural point map and RGB rendering achieve

![img-5.jpeg](img-5.jpeg)

Fig. 7: Demonstration of the effect of loop closure correction on the in-house car dataset. When the vehicle revisits a previously mapped region, we compare the neural point map (colored by timestep) and RGB rendering (viewed from the green frustum) with and without correcting the loop closure. Without loop closure correction, the misaligned neural points create visual artifacts like trees appearing on the road. After applying loop closure correction and map update, we achieve globally consistent neural point map and RGB rendering.

![img-6.jpeg](img-6.jpeg)

Fig. 8: Map memory efficiency analysis comparing mapping quality versus memory usage. Left: Radiance field comparison between PINGS and GSS on the in-house car dataset using LPIPS metric. Right: Distance field comparison between PINGS and VDB-Fusion on the Oxford Spires dataset using Chamfer distance. Points represent results at different map resolution, with points closer to the bottom-left corner indicating better quality-memory trade-off.

global consistency. These results validate that PINGS can build globally consistent maps at large scale through loop closure correction by leveraging the elasticity of neural points.

### IV-F Map Memory Efficiency Evaluation

Fig. 8 depicts the map memory usage in relation to the rendering quality for the radiance field and the surface reconstruction quality for the distance field. Experiment results validate that storing the neural points and decoder MLPs in PINGS is more memory-efficient than directly storing the Gaussian primitives or the discrete SDF in voxels. With equivalent memory usage, PINGS achieves superior performance across both metrics: better novel view rendering photorealism (lower LPIPS) compared to GSS [11] on the in-house car dataset, and better surface reconstruction accuracy (lower Chamfer distance) compared to the discrete TSDF-based method VDB-Fusion [67] on the Oxford Spires dataset. Moreover, while GSS and VDB-Fusion each model only a single field type, our PINGS framework efficiently represents both radiance field and SDF within a single map. The efficiency of PINGS comes from the globally-shared decoder MLPs that learn common patterns, and locally-defined neural points that compactly encode multiple Gaussian primitives and continuous SDF values through feature vectors instead of storing them explicitly.

## V. LIMITATIONS

Our current approach has three main limitations. First, although our SDF mapping and LiDAR odometry modules operate at sensor frame rate, the computationally intensive radiance field mapping results in an overall processing time of around five seconds per frame on an NVIDIA A6000 GPU. This performance bottleneck could potentially be addressed through recent advances in efficient optimization schemes for Gaussian splatting training [20, 40]. Second, PINGS relies solely on online per-scene optimization without using any pretrained priors. Incorporating such priors [7] into our unified map representation could improve both mapping quality and convergence speed. Finally, though PINGS can filter dynamic objects using the distance field, it lacks explicit 4D modeling capabilities. This limitation is noticeable in highly dynamic environments and when objects transition between static and dynamic states. Future work could address this challenge by incorporating object detection priors [9, 80] to enable accurate 4D mapping of both radiance and distance fields.

## VI. CONCLUSION

In this paper, we present a new LiDAR-visual SLAM system making use of a novel map representation that unifies a continuous signed distance field and a Gaussian splatting radiance field within an elastic and compact set of neural points. By introducing mutual geometric consistency constraints between these fields, we jointly improve both representations. The distance field provides geometric structure to guide radiance field optimization, while the radiance field's dense photometric cues and multi-view consistency enhance the distance field's accuracy. Our experimental results on challenging large-scale datasets show that our method can incrementally construct globally consistent maps that outperform baseline methods in the novel view rendering fidelity, surface reconstruction quality, odometry estimation accuracy, and map memory efficiency.