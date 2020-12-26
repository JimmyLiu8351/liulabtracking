# Eye Tracking Log

### Nov 5th
##### Links
- [Basics on how to compile and debug C++ in VSCode](https://code.visualstudio.com/docs/cpp/config-linux)
- Look up Bjarne Stroustrup Youtube videos
- [C vs C++](https://softwareengineering.stackexchange.com/questions/16390/what-are-the-fundamental-differences-between-c-and-c)
    - [Smart pointers](https://docs.microsoft.com/en-us/cpp/cpp/smart-pointers-modern-cpp?view=msvc-160) and [RAII](https://docs.microsoft.com/en-us/cpp/cpp/object-lifetime-and-resource-management-modern-cpp?view=msvc-160)
    - [Constructor and Destructor](https://www.tutorialspoint.com/cplusplus/cpp_constructor_destructor.htm)
    - [Templates](http://www.cplusplus.com/doc/oldtutorial/templates/) - haven't read yet

##### Random Knowledge

##### Tasks
- Choose a few frames to do analysis on
- Find center of mass and a point in the eye and then do starburst
- Plot differentials of the different rays
- Ask teledyne person about preferred language for drivers + CUDA version
    - Install latest CUDA 9.X/10.X version
- Run speed tests for graphics card loading


### Nov 12th
##### Links
- [Installing opencv for c++](http://techawarey.com/programming/install-opencv-c-c-in-ubuntu-18-04-lts-step-by-step-guide/)
    - Remember to move the opencv2 file out of opencv4 file in /usr/local/include so that it doesn't give an error in VSCode
    - Command to compile: 
    ```bash
    g++ input.cpp -o outputfilename `pkg-config --cflags --libs opencv`
    ```

##### Tasks
- Look at *fast radial symmetry transform*
- Try out code from tracker app on [github](http://github.com/coxlab/eyetracker)


### Nov 17th
##### FRST Implementation
- [FRST Python Implementation](https://github.com/Xonxt/frst)
- Mouse video is 320 x 240 at 100FPS
    - Frame is 35cm across on my monitor -> 9.14pixels/cm
    - Eye is 2.6cm diameter on my monitor
    - Eye is ~12 pixels radius
- Parameters:
    - N/radii - radius in pixels of the circle to be detected
    - alpha - strictness of of frst for how close the circle has to be in radius to N
        - 2 is a good place to start
    - beta - gradient threshold, ex: 0.2, gradients 20% or less of the maximum amplitude are ignored
        - increases performances the higher beta is
    - stdFactor - spreads the effect of each point, paper uses 0.5n
    - dark/bright - only detect circles that are dark/bright

### Nov 19th
##### Links
- [Drawing functions](https://docs.opencv.org/master/dc/da5/tutorial_py_drawing_functions.html)
- [Video Capture](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a85b55cf6a4a50451367ba96b65218ba1)
- [How to find maximum in 2D numpy array](https://stackoverflow.com/questions/55284090/how-to-find-maximum-value-in-whole-2d-array-with-indices)
- [Gaussian Blurring](https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html)
##### Notes
- Testing with a black circle in a white background first
    - Image is 400 x 400 pixels / 11.2cm on screen
    - Circle is 7.8cm of diameter -> 139 pixels radius
    - Weird glitch where the focus is on the top left - fixed, coordinate error

### Nov 23rd
##### Links
 - [OpenCV thresholding](https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html)
 
##### Notes
- frst is working but the corneal glare is too strong, going to try and mitigate it

##### Tasks
- Try and contact the author of the libraries - get his code for research purposes

### December 3rd
##### Links
- [Create your own contour](https://stackoverflow.com/questions/14161331/creating-your-own-contour-in-opencv-using-python)

##### Notes
- Having trouble with starburst, most likely because of different coordinate systems between opencv and numpy, looking into this
- It seems that opencv functions take in (x, y) while indexing is (y, x), kind of confusing
- Top left is zero though
- Don't need to use Sobel for intensity, can just use comparisons so that should save some computation power
- Starburst could definitely still be optimized, will look into it later
- Added a blur to after the rough corneal remove is very effective for frst

##### Meeting Notes
- 3 scenarios, when corneal reflection is at the edge, ignore the ones at the edge
    - Other scenario is corneal reflection is at the center
    - No corneal reflection at all
- Algorithm: fill up gap if pupil is smaller than starburst size thresholdex
    - Backup algorithm use simple thresholding to cover up the gap
- Using derivates on the starburst to better counter the starburst
    - If corneal reflection in the center, you see one positive slope
    - If corneal reflection in the middle, two slopes, use the value of the slope to determine which one is from the corneal reflection
    - If no corneal reflection, you see one positive slope 
    - If corneal reflection at the boundary, one positive slope, is very high
- Parallelization of starburst algorithm


### December 11th
##### Links
- [Find point by angle and length](https://stackoverflow.com/questions/22252438/draw-a-line-using-an-angle-and-a-point-in-opencv)
- [Line iterator](https://stackoverflow.com/questions/32328179/opencv-3-0-lineiterator)
##### Notes
- If there is a corneal reflection, there should be a positive peak and then negative peak, if it's the edge of the pupil then it should be just a positive peak

### December 12th
##### Notes
- Tried to multiprocess the starburst by line but each iteration of starburst is inconsistent in the number of points that come from each iteration of starburst
- Need some way for if there's not enough points in the contour then use back-up algorithm
- REMEMBER TO BLUR THE IMAGE BEFORE USING STARBURST SO IMPORTANT!!!
- Need to redo the thing to redo frst every 5 frames


### December 17th
##### Links
- [Multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
- [Video writer on ubuntu](https://answers.opencv.org/question/182488/not-able-to-save-video-using-cv2videowriter-on-ubuntu-1604/)
##### Notes
- Doing speed testing on the lab computer today
- Non-realtime scheduler, 16 threads around 360 iterations/second
- Realtime scheduler only doing about 60 iterations/second
    - Did some profiling, frst and rough corneal remove taking the longest time
    - Did optimization on rough corneal remove, improved to around 75 iterations/second
    - Realtime scheduler is hot garbage, got better results just running single-threaded
- Maybe look to optimize non realtime scheduler, I think I can push it even further with thread optimizations
- I don't know how but my single-thread thing is running at 200 FPS
##### Next
- Realtime processing, maybe some kind of system where frames get processed as they come in
- Measure real absolute time of functions
- Add condition with derivatives, second positive


### December 26th
##### Links
- [Queue and pipe time](https://stackoverflow.com/questions/8463008/multiprocessing-pipe-vs-queue)
- [Timing](https://stackoverflow.com/questions/1938048/high-precision-clock-in-python#:~:text=Python%20on%20Windows%20uses%20%2B%2D,you're%20measuring%20execution%20time.)