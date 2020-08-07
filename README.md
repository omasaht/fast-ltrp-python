# fast-ltrp-python
Fast Implementation of Local Tetra Pattern \[1] in Python. It uses Adrian Ungureanu's python implementation as a reference for comparison. On my system, I have been able to achieve upto **80x** improved computation speed compared to original. Similar approach could be used to work with other Local Patterns such as LBP, LDP, etc. One downside of this approach is the memory footprint it leaves which could be problematic depending upon input size.

## Required Libraries
1. OpenCV (4.2.0)
2. NumPy (1.18.5)
3. Scikit-Image (0.17.2)

## Results
### LTrP feature maps
Input:

![input](images/input.png?raw=true)

Output:

![output](images/feature-maps.jpg?raw=true)

### Speed Comparison
![console](images/console.gif?raw=true)

| Implementation | Image Size | Time (ms) |
| --- | --- | --- |
| Adrian | 341x341  | 2356 |
| Omar  | 341x341  | 29 |

## License
Copyright (c) 2020, Omar Hassan. (MIT License)

See LICENSE for more info.

## Acknowledgements
Code is partially reused from Adrian Ungureanu's implementation of LTrP: [Github Repository](https://github.com/AdrianUng/palmprint-feature-extraction-techniques)

## References
\[1] S. Murala, R. P. Maheshwari and R. Balasubramanian, "Local Tetra Patterns: A New Feature Descriptor for Content-Based Image Retrieval," in IEEE Transactions on Image Processing, vol. 21, no. 5, pp. 2874-2886, May 2012.
doi: 10.1109/TIP.2012.2188809. [IEEE-Xplore link](https://ieeexplore.ieee.org/abstract/document/6175124)
