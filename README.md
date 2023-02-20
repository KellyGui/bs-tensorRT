# bs-tensorRT
This project implements the deeplabv3+ netowrk with TensorRT API for saving time and resources.
The ArgMax is made as a plugin which supports both dynamic shape and fixed shape.

## How to build
### 1. presiquisite
       TensorRT: TensorRT-7.2.3.4 is uesd in this project.
       Opencv: opencv-3.3 in this project, but you can use opencv 4.x instead.
### 2. build
    ```
mkdir build&&cd build
cmake ..
make
```
       
