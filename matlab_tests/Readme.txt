Sparse Representation for Phase Contrast Image
                           

ABOUT
The source code implements preconditioning for phase contrast microscopy image, which provides two different algorithms for use to choose:
1) preconditioning based on a linear imaging model: The source code implements a restoration method for phase contrast image by understanding its optics. We derive a linear imaging model corresponding to the phase contrast optics and formulate a quadratic optimization function to restore the authentic phase contrast image without halos or shade-off artifacts. With artifacts removed, high quality segmentation can be achieved by simply thresholding the restored images.  
2) a modified method based on dictionary representation: we construct a dictionary based on diffraction patterns of phase contrast microscopes. By formulating and solving a min-l1 optimization problem, each pixel is restored into a feature vector corresponding to the dictionary representation. Cells in the images are then segmented by the feature vector clustering. 

It is written in MATLAB for ease of use, and detailed documentation throughout. It supports
Windows and Linux.

The documentation is available online at
http://www.celltracking.ri.cmu.edu/

Please refer the following papers if you happen to use this source code.

[1] Hang Su, Zhaozheng Yin, Takeo Kanade, Seungil Huh: Phase Contrast Image Restoration via Dictionary Representation of Diffraction Patterns. The 15th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), 2012: 615-622

[2] Zhaozheng Yin, Takeo Kanade, Mei Chen: Understanding the phase contrast optics to restore artifact-free microscopy images for segmentation. Medical Image Analysis 16(5): 1047-1062 (2012) 

QUICK START AND FILE LIST

To start using preconditioning source code, download the latest version from: http://www.celltracking.ri.cmu.edu/

1.	Sparse_Resp_GUI.m: provides an interface to adjust the corresponding parameters, and implements the restoration with sparse representation. The process is as follows: 
	> Open a phase contrast image from the menu.
        > select the corresponging method 
	> Set the optimization and kernel parameters
	> Run restoration by press the sparse representation button
	> Run binary convert to get a binary cell image  
2.	Demo.m: demo code to demonstrate the restoration procedure
3.  precondition_phase_contrast.m: main function for preconditioning, and the description of input and output arguments are detailed in the documents.   
4.	precondition_sparse_respresent.m: preconditioning based on sparse respresentatin algorithms in [1]
5.  precondition_linear_model.m: preconditioning based on linear imaging model in [2]
6.	folder func: functions implement some sub-functions, e.g., background removal, kernel generation, etc.  
7.  folder GUI function: functions implements some GUI sub-functions, e.g., image selection, parameter setup. 
8.  folder seq_training: functions for traing better initailization
