#include "deeplabv3plus.h"

int main(int argc, char** argv){
    std::string engine_name =  "deeplabv3plus.engine";
    std::string weights_name = "./bareSimple2.wts";
    std::string fir_name = "/home/ylc/GMP/trt7/baresoil_v2/build/output";
    
    // std::string weights_name = "/data/gmp/TensorRT_EVN/projects/baresoil/deeplabv3plus/build/bareSimple2.wts";
    // std::string fir_name = "/data/gmp/TensorRT_EVN/projects/baresoil/deeplabv3plus/build/output";
    
    char *trtModelStream{nullptr};
    size_t size{0};
    if(argc == 2 && std::string(argv[1]) == "-s"){
        //serialize engine file
        IHostMemory* modelStream{nullptr};
        APIToModel(BATCH_SIZE, &modelStream, weights_name);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }
    else{
         std::cerr << "./serialize -s // serialize model to plan file" << std::endl;
         return -1;
    }
}