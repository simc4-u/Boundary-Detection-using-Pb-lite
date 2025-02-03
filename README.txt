Implementation Details (January 16, 2025)


Phase 1:

For phase 1, Please input the path of input directory and output director in main function. 
image_dir - input directory
output_dir - output_directory
You can find all Phase 1 results in the phaseoutput folder. Please run wrapper.py to review these results.
Please find wrapper_fromscratch.py under phaseoutput folder. Itis completely working in case i am not allowed to use basic cv function.

In this also just change the path.

Phase2:

MY deep learning model implementation is organized as follows:
I have five different class for each neural network(BasicNet, ImprovedNet, ResNet, ResNext and DenseNet). You can find detailed instructions for switching between these networks in the Network.py file.

The train.py script automatically tracks and plots important model metrics including the number of parameters, training accuracy, and testing accuracy.
Important note: When using ResNet, ResNeXt, or DenseNet, please uncomment AdamW to SGD optimizer for optimal performance.

The system saves checkpoints automatically for each model type with their respective names. All code includes detailed comments to help understand the implementation.



All files have been thoroughly documented with comments to make them easy to understand and modify. 


