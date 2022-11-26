# Photo To NPR Image Generator

## Description

Automatically generates a non-photo-realistic image out of a real world photograph by producing a line sketch and reducing the palette of original colors. 

The baseline model achieves this by using Canny Edge Detection to produce the line sketch and K Means Clustering of the colors.

## Setup and Installation

- To install the dependancies, run

        pip install -r requirements.txt

- baselineModel.py depends on cannyLineSketch.py, kMeansColorPalette.py and depthBlend.py. To run the model on an image whose filename is <code>img.jpg</code>, the full path is not necessary. The input is relative to the data folder, and the output is saved as <code>Baseline_img.jpg</code> in the data folder.  

        python3 baselineModel.py img.jpg


## Sample Outputs

### Baseline Model

Input | Line Sketch | Colors | Output
:-:|:-:|:-:|:-:
![Tristan Gretzky](data/inputs/Tristan_Gretzky_0001.jpg) | ![Tristan Gretzky](data/outputs/Canny_Tristan_Gretzky_0001.jpg) | ![Tristan Gretzky](data/outputs/KMeans_Tristan_Gretzky_0001.jpg) | ![Tristan Gretzky](data/outputs/Baseline_Tristan_Gretzky_0001.jpg)
![TJ Ford](data/inputs/TJ_Ford_0001.jpg) | ![TJ Ford](data/outputs/Canny_TJ_Ford_0001.jpg) | ![TJ Ford](data/outputs/KMeans_TJ_Ford_0001.jpg) | ![TJ Ford](data/outputs/Baseline_TJ_Ford_0001.jpg)
![Victoria Clarke](data/inputs/Victoria_Clarke_0004.jpg) | ![Victoria Clarke](data/outputs/Canny_Victoria_Clarke_0004.jpg) | ![Victoria Clarke](data/outputs/KMeans_Victoria_Clarke_0004.jpg) | ![Victoria Clarke](data/outputs/Baseline_Victoria_Clarke_0004.jpg)
![Zhang Wenkang](data/inputs/Zhang_Wenkang_0002.jpg) | ![Zhang Wenkang](data/outputs/Canny_Zhang_Wenkang_0002.jpg) | ![Zhang Wenkang](data/outputs/KMeans_Zhang_Wenkang_0002.jpg) | ![Zhang Wenkang](data/outputs/Baseline_Zhang_Wenkang_0002.jpg)

### Final Model

Input | Line Sketch | Colors | Output
:-:|:-:|:-:|:-:
![Aishwarya Rai](data/inputs/Aishwarya_Rai_0001.jpg) | ![Aishwarya Rai](data/outputs/LIC_Aishwarya_Rai_0001.jpg) | ![Aishwarya Rai](data/outputs/Region_Aishwarya_Rai_0001.jpg) | ![Aishwarya Rai](data/outputs/Final_Aishwarya_Rai_0001.jpg)
![Arnold Schwarzenegger](data/inputs/Arnold_Schwarzenegger_0004.jpg) | ![Arnold Schwarzenegger](data/outputs/LIC_Arnold_Schwarzenegger_0004.jpg) | ![Arnold Schwarzenegger](data/outputs/Region_Arnold_Schwarzenegger_0004.jpg) | ![Arnold Schwarzenegger](data/outputs/Final_Arnold_Schwarzenegger_0004.jpg)
![Sachin Tendulkar](data/inputs/Sachin_Tendulkar_0001.jpg) | ![Sachin Tendulkar](data/outputs/LIC_Sachin_Tendulkar_0001.jpg) | ![Sachin Tendulkar](data/outputs/Region_Sachin_Tendulkar_0001.jpg) | ![Sachin Tendulkar](data/outputs/Final_Sachin_Tendulkar_0001.jpg)
![Aung San Suu Kyi](data/inputs/Aung_San_Suu_Kyi_0001.jpg) | ![Aung San Suu Kyi](data/outputs/LIC_Aung_San_Suu_Kyi_0001.jpg) | ![Aung San Suu Kyi](data/outputs/Region_Aung_San_Suu_Kyi_0001.jpg) | ![Aung San Suu Kyi](data/outputs/Final_Aung_San_Suu_Kyi_0001.jpg)


## Team Information
Team Name: iShowSpeed
- Sai Sriram Yannamani &emsp; 2019101029
- Raghav Raj Dwivedi &emsp; 2019101008
- Rishin Chakraborty &emsp; 2019112008
- Astitva Ranjan &emsp; 2019112025
- Rishu Anand &emsp; 2022802007

## Biblography:
- Original Paper: https://web.stanford.edu/class/ee368/Project_Winter_1819/Reports/teo_colas_deng.pdf
- Dataset: https://drive.google.com/file/d/1W782sqp0B4vFXX4Vo4Uux5zEk2jcpk3i/view?usp=sharing
