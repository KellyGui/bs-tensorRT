#include "deeplabv3plus.h"

//gdb
//set args /home/zhongxy/projects/tensorrtx-master/yolov5/build/batch64car.engine /home/zhongxy/projects/tensorrtx-master/yolov5/data 1000 1000 0.8 /disk0/gmp/datasets/car/0114output

bool inference_parse_args(int argc, char** argv, std::string& engine, std::string& img_dir, std::string &savePath){
    if(argc<4) return false; 
    engine = std::string(argv[1]);
    img_dir = std::string(argv[2]);
    savePath = std::string(argv[3]);
   
    return true;
}

int main(int argc, char** argv){
    cudaSetDevice(DEVICE);
    std::string engine_file_name = "";
    std::string img_dir = "";
    std::string savePath="";
    bool flag=false;
    
    std::cout<<"parser args begin...\n";
    if(!inference_parse_args(argc, argv, engine_file_name, img_dir, savePath)){
       std::cerr<<"arguments not right!"<<std::endl;
       std::cerr<<"./inference engine_file_name image_source_dir savePath"<<std::endl;
       return -1;
    }
    std::cout<<"parser args Done.\n";

    //创建Inference实例对象
    Inference* infer = new Inference(engine_file_name);
  
    
    //get all input images
    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }
   
    int fcount = 0;
   
    
    while(1){
        std::string imageName;
        std::cin>>imageName;
        auto big_start = std::chrono::steady_clock::now();
        auto mask = infer->inferMat(imageName);
        auto big_end = std::chrono::steady_clock::now();
        std::cout<<"inference time is "<<std::chrono::duration_cast<std::chrono::milliseconds>(big_end-big_start).count()<<"ms\n";

    }
    

}