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
