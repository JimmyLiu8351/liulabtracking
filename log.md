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
    - Weird glitch where the focus is on the top left