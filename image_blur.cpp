#include <iostream>
#include <vector>
#include <cstring>
#include <ctime>
#include <ratio>
#include <chrono>
#include <fstream>
#include <CL/sycl.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace cl::sycl;
using namespace std;

int main(int agrc, char** argv) {
    int channels;
    static constexpr auto filterSize = 7;
    unsigned char* inputData = nullptr;
    unsigned char* outputData_sycl = nullptr;
    unsigned char* outputData_native = nullptr;
    int inputWidth, inputHeight;

    //Load image
    inputData = stbi_load(argv[1], &inputWidth, &inputHeight, &channels, 0);
    if(inputData == NULL) {
        std::cout << "Error in loading the image" << std::endl;
        exit(1);
    }

    std::cout << "Loaded image with a width of " << inputWidth << "px, a height of " << inputHeight << "px and " 
    << channels << " channels" << std::endl;

    // Create filters
    vector<vector<double> > filter(filterSize, vector<double>(filterSize));
    for (int i = 0; i < filter.size(); i++) {
        for (int j = 0; j < filter[i].size(); j++) {
            filter[i][j] = 1.0 / (filterSize * filterSize);
        }
    }

    int pixCount = inputWidth * inputHeight;
    int imgSize = pixCount * channels;

    // rgbrgb..rgb -> rr..rgg..gbb..b
    unsigned char *rgb = new unsigned char[imgSize];
    for(size_t i = 0; i < pixCount; i++) {
        *(rgb + i) = (uint8_t)(*(inputData + i * channels));
        *(rgb + pixCount + i) = (uint8_t)(*(inputData + i * channels + 1));
        *(rgb + 2 * pixCount + i) = (uint8_t)(*(inputData + i * channels + 2));
    }

    /*
        Native method
    */
    outputData_native = new unsigned char[imgSize];
    unsigned char *r_curr_pix, *g_curr_pix, *b_curr_pix;
    int capacity = (filterSize - 1) / 2;
    auto start_time = chrono::high_resolution_clock::now();
    for(int i = 0; i <= inputHeight - 1; i++) {
        for(int j = 0; j <= inputWidth - 1; j++) {
            vector<double> sum(3, 0);
            for(int k = (-1) * capacity; k <= capacity; k++) {
                for(int t = (-1) * capacity; t <= capacity; t++) {
                    if((j + t) < 0 || (i + k) < 0) {
                        continue;
                    }
                    r_curr_pix = rgb + i * inputWidth + j;
                    g_curr_pix = (rgb + pixCount) + i * inputWidth + j;
                    b_curr_pix = (rgb + 2* pixCount) + i * inputWidth + j;
                    sum[0] += (uint8_t)(double(*(r_curr_pix + k * inputWidth + t)) * filter[capacity + k][capacity + t]);
                    sum[1] += (uint8_t)(double(*(g_curr_pix + k * inputWidth + t)) * filter[capacity + k][capacity + t]);
                    sum[2] += (uint8_t)(double(*(b_curr_pix + k * inputWidth + t)) * filter[capacity + k][capacity + t]);
                }
            }
            *(outputData_native + (i * inputWidth + j) * channels) = (uint8_t)(sum[0]);
            *(outputData_native + (i * inputWidth + j) * channels + 1) = (uint8_t)(sum[1]);
            *(outputData_native + (i * inputWidth + j) * channels + 2) = (uint8_t)(sum[2]);
            sum.clear();
        }
    }
    auto end_time = chrono::high_resolution_clock::now();
    auto native_time = end_time - start_time;
    stbi_write_jpg("native_methods.jpg", inputWidth, inputHeight, channels, outputData_native, 100);

    delete[] rgb;
    delete[] outputData_native;

    /*
        SYCL implementation
    */
    outputData_sycl = new unsigned char[imgSize];

    // Create flilter range
    range<2> filterRange(filterSize, filterSize);

    std::string MySelector = argv[2];

    queue myQueue;

    queue cpuQueue( cpu_selector{});
    queue gpuQueue( gpu_selector{});

    if (std::strcmp(argv[2], "cpu") == 0) {
        myQueue = cpuQueue;
    } else {
        myQueue = gpuQueue;
    }


    std::cout << "\nRunning on "
                << myQueue.get_device().get_info<cl::sycl::info::device::name>()
                << "\n";

    vector<int> imgData(imgSize);
    for(int i = 0; i < imgSize; i++) {
        imgData[i] = (int(*(inputData + i)));
    }

    {
        /*
            Create filter's kernel
        */  
        buffer<float, 2> blur(filterRange);
        myQueue.submit([&](cl::sycl::handler& cgh) {
            auto globalBlur = blur.get_access<access::mode::discard_write>(cgh);
            cgh.parallel_for<class blurFilter>(filterRange, [=](cl::sycl::item<2> i) {
                globalBlur[i] = 1. / (filterSize * filterSize);
            });
        });
        cl::sycl::buffer<int> imgData_buffer(imgData.data(), cl::sycl::range<1>(imgSize));
        cl::sycl::buffer<int> RGB_buffer(imgSize);

        myQueue.submit([&](cl::sycl::handler& cgh) {
            auto globalImgBuf = imgData_buffer.get_access<access::mode::read>(cgh);
            auto globalRGB = RGB_buffer.get_access<access::mode::discard_write>(cgh);
            cgh.parallel_for<class to_rgb>(cl::sycl::range<1>(pixCount), [=](cl::sycl::nd_item<1> item) {
                int wiID = item.get_global_id()[0];
                globalRGB[wiID] = globalImgBuf[wiID * channels];
                globalRGB[wiID + pixCount] = globalImgBuf[wiID * channels + 1];
                globalRGB[wiID + 2 * pixCount] = globalImgBuf[wiID * channels + 2];
            });
        });

        imgData.clear();

        std::vector<int> finalIMG_vector(imgSize, 0);
        buffer<int> finalIMG_buffer(finalIMG_vector.data(), cl::sycl::range<1>(imgSize));
        start_time = chrono::high_resolution_clock::now();
        std::cout << "Start calculation..." << std::endl;
        try {
            myQueue.submit([&](cl::sycl::handler& cgh) {
                auto globalRGB = RGB_buffer.get_access<access::mode::read>(cgh);
                auto globalBlur = blur.get_access<access::mode::read>(cgh);
                auto globalFinalImg = finalIMG_buffer.get_access<access::mode::discard_write>(cgh);

                cgh.parallel_for<class create_final_image>(cl::sycl::range<1>(pixCount), [=](cl::sycl::nd_item<1> item) {
                    size_t wiID = item.get_global_id()[0];
                    auto r = 0;
                    auto g = 0;
                    auto b = 0;
                    constexpr auto offset = (filterSize - 1) / 2;
                    for (int x = -offset; x <= offset; x++) {
                        for (int y = -offset; y <= offset; y++) {
                            if(((wiID % inputWidth) + y) < 0 || (size_t(wiID / inputWidth) + x) < 0) {
                                continue;
                            }
                            if(((wiID % inputWidth) + y) >= inputWidth || (size_t(wiID / inputWidth) + x) >= inputHeight) {
                                continue;
                            }
                            size_t r_curr_pix = wiID;
                            size_t g_curr_pix = wiID + pixCount;
                            size_t b_curr_pix = wiID + 2 * pixCount;
                            r += globalRGB[r_curr_pix + x * inputWidth + y] * globalBlur[offset + x][offset + y];
                            g += globalRGB[g_curr_pix + x * inputWidth + y] * globalBlur[offset + x][offset + y];
                            b += globalRGB[b_curr_pix + x * inputWidth + y] * globalBlur[offset + x][offset + y];
                        }
                    }
                    globalFinalImg[wiID * channels] = r;
                    globalFinalImg[wiID * channels + 1] = g;
                    globalFinalImg[wiID * channels + 2] = b;
                });
            });
            myQueue.wait_and_throw();
        } catch (std::exception e) {
            std::cout << e.what() << std::endl;
        }

        std::cout << "Finish calculation..." << std::endl;
        end_time = chrono::high_resolution_clock::now();
        auto sycl_time = end_time - start_time;
        std::cout << std::endl << "Native method's time: " << chrono::duration_cast<chrono::microseconds>(native_time / 1000).count() <<
        " microseconds" << std::endl;
        std::cout << "Parallel method's time: " << chrono::duration_cast<chrono::microseconds>(sycl_time / 1000).count() << 
        " milliseconds" << std::endl;

        auto globalFinalImg = finalIMG_buffer.get_access<access::mode::read>();
        for(int i = 0; i < pixCount; i++) {
            *(outputData_sycl + i * channels) = globalFinalImg[i * channels];
            *(outputData_sycl + i * channels + 1) = globalFinalImg[i * channels + 1];
            *(outputData_sycl + i * channels + 2) = globalFinalImg[i * channels + 2];
        }
        stbi_write_jpg("nature_sycl.jpg", inputWidth, inputHeight, channels, outputData_sycl, 100);
    }

    // free memory
    delete[] outputData_sycl;
    delete[] inputData;
    return 0;
}
