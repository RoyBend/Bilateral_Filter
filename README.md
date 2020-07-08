# Bilateral_Filter
The bilateral filter is a technique to smooth images while preserving edges,  
It replaces the intensity of each pixel with a weighted average of intensity values from nearby pixels.   
This weight is based on a Gaussian distribution. Crucially, the weights depend not only on Euclidean distance of pixels, but also on the radiometric differences (e.g., range   differences, such as color intensity, depth distance, etc.). This preserves sharp edges. (taken from MIT & Wikipedia)  

In this project implemention of the algorithm in  Python language with reference to the formulas and calculations written in the article by Tomasi & Manduchi.
Implemention of this algorithm for both B&W and colored pictures.  

In order to improve performance, calculate the gaussian differences in advance, for the domain.  

Bilateral_FilterMulty file: improving the algorithm performance by using multy proccess
