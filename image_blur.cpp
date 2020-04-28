#include <iostream>
#include <vector>
#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>
#include <CL/sycl.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace cl::sycl;
using namespace std;

range<2> get_optimal_local_range(cl::sycl::range<2> globalSize,
                                 cl::sycl::device d) {
    range<2> optimalLocalSize(0, 0);

    if (d.is_gpu()) {
        optimalLocalSize = range<2>(64, 1);
    } else {
        optimalLocalSize = range<2>(4, 1);
    }

    for (int i = 0; i < 2; ++i) {
        while (globalSize[i] % optimalLocalSize[i]) {
            optimalLocalSize[i] = optimalLocalSize[i] >> 1;
        }
    }
    return optimalLocalSize;
}

int main(int, char**) {
    int width, height, channels;
    //Load image
    unsigned char *img = stbi_load("wot.jpg", &width, &height, &channels, 0);
    if(img == NULL) {
        std::cout << "Error in loading the image" << std::endl;
        exit(1);
    }
    std::cout << "Loaded image with a width of " << width << "px, a height of " << height << "px and " 
    << "and " << channels << " channels" << std::endl;

    // Create filters
    size_t filter_size = 7;
    vector<vector<double> > filter(filter_size, vector<double>(filter_size));
    for (int i = 0; i < filter.size(); i++) {
        for (int j = 0; j < filter[i].size(); j++) {
            filter[i][j] = 1.0 / (filter_size * filter_size);
        }
    }

    size_t pixels_count = width * height;
    size_t img_size = pixels_count * channels;

    // rgbrgb..rgb -> rr..rgg..gbb..b
    unsigned char *rgb = new unsigned char[img_size * sizeof(unsigned char)];
    for(size_t i = 0; i < pixels_count; i++) {
        *(rgb + i) = (uint8_t)(*(img + i * channels));
        *(rgb + pixels_count + i) = (uint8_t)(*(img + i * channels + 1));
        *(rgb + 2 * pixels_count + i) = (uint8_t)(*(img + i * channels + 2));
    }
    /*
        Native method
    */
    unsigned char *final_image = new unsigned char[img_size * sizeof(unsigned char)];
    unsigned char *r_curr_pix, *g_curr_pix, *b_curr_pix;
    int capacity = (filter_size - 1) / 2;
    auto start_time = chrono::high_resolution_clock::now();
    for(int i = 0; i <= height - 1; i++) {
        for(int j = 0; j <= width - 1; j++) {
            vector<double> sum(3, 0);
            for(int k = (-1) * capacity; k <= capacity; k++) {
                for(int t = (-1) * capacity; t <= capacity; t++) {
                    if((j + t) < 0 || (i + k) < 0) {
                        continue;
                    }
                    r_curr_pix = rgb + i * width + j;
                    g_curr_pix = (rgb + pixels_count) + i * width + j;
                    b_curr_pix = (rgb + 2* pixels_count) + i * width + j;
                    sum[0] += (uint8_t)(double(*(r_curr_pix + k * width + t)) * filter[capacity + k][capacity + t]);
                    sum[1] += (uint8_t)(double(*(g_curr_pix + k * width + t)) * filter[capacity + k][capacity + t]);
                    sum[2] += (uint8_t)(double(*(b_curr_pix + k * width + t)) * filter[capacity + k][capacity + t]);
                }
            }
            *(final_image + (i * width + j) * channels) = (uint8_t)(sum[0]);
            *(final_image + (i * width + j) * channels + 1) = (uint8_t)(sum[1]);
            *(final_image + (i * width + j) * channels + 2) = (uint8_t)(sum[2]);
            sum.clear();
        }
    }
    auto end_time = chrono::high_resolution_clock::now();
    std::cout << "Native method's time: " << chrono::duration_cast<chrono::microseconds>((end_time - start_time) / 1000).count() <<
    " microseconds" << std::endl;
    stbi_write_jpg("wot_native_methods.jpg", width, height, channels, final_image, 100);


    /*
        SYCL implementation
    */
    static constexpr auto filterSize = 7;
    unsigned char* inputData = nullptr;
    unsigned char* outputData = nullptr;
    int inputWidth, inputHeight;
    // Load image
    inputData = stbi_load("wot.jpg", &inputWidth, &inputHeight, &channels, 0);

    int imgSize = inputWidth * inputHeight * channels;
    int pixCount = inputWidth * inputHeight;
    // Create image and flilter range
    range<2> imgRange(inputHeight, inputWidth);
    range<2> filterRange(filterSize, filterSize);
    // Final image
    outputData = new unsigned char[imgSize];


    // Create queue
    queue Queue([](cl::sycl::exception_list l) {
    for (auto ep : l) {
        try {
            std::rethrow_exception(ep);
        } catch (const cl::sycl::exception& e) {
            std::cout << "Async exception caught:\n" << e.what() << "\n";
            throw;
        }
    }
    });

    vector<int> imgData(imgSize);
    for(int i = 0; i < pixCount; i++) {
        imgData[i * channels] = (int(*(inputData + i * channels)));
        imgData[i * channels + 1] = (int(*(inputData + i * channels + 1)));
        imgData[i * channels + 2] = (int(*(inputData + i * channels + 2)));
    }

    {
        /*
            Create filter's kernel
        */  
        buffer<float, 2> blur(filterRange);
        Queue.submit([&](cl::sycl::handler& cgh) {
            auto globalBlur = blur.get_access<access::mode::discard_write>(cgh);
            cgh.parallel_for<class blurFilter>(filterRange, [=](cl::sycl::item<2> i) {
                globalBlur[i] = 1. / (filterSize * filterSize);
            });
        });
        cl::sycl::buffer<int> imgData_buffer(imgData.data(), cl::sycl::range<1>(imgSize));
        cl::sycl::buffer<int> RGB_buffer(imgSize);

        Queue.submit([&](cl::sycl::handler& cgh) {
            auto globalImgBuf = imgData_buffer.get_access<access::mode::read>(cgh);
            auto globalRGB = RGB_buffer.get_access<access::mode::write>(cgh);
            cgh.parallel_for<class to_rgb>(cl::sycl::range<1>(pixCount), [=](cl::sycl::nd_item<1> item) {
                int wiID = item.get_global_id()[0];
                globalRGB[wiID] = globalImgBuf[wiID * channels];
                globalRGB[wiID + pixCount] = globalImgBuf[wiID * channels + 1];
                globalRGB[wiID + 2 * pixCount] = globalImgBuf[wiID * channels + 2];
            });
        });
        
        std::vector<int> finalIMG_vector(img_size, 0);
        buffer<int> finalIMG_buffer(finalIMG_vector.data(), cl::sycl::range<1>(imgSize));
        start_time = chrono::high_resolution_clock::now();
        Queue.submit([&](cl::sycl::handler& cgh) {
            cl::sycl::stream kernelout(320000, 1024, cgh);
            auto globalRGB = RGB_buffer.get_access<access::mode::read>(cgh);
            auto globalBlur = blur.get_access<access::mode::read>(cgh);
            auto globalFinalImg = finalIMG_buffer.get_access<access::mode::write>(cgh);

            cgh.parallel_for<class create_final_image>(cl::sycl::range<1>(pixCount), [=](cl::sycl::nd_item<1> item) {
                int wiID = item.get_global_id()[0];
                constexpr auto offset = (filterSize - 1) / 2;
                for (int x = -offset; x <= offset; x++) {
                    for (int y = -offset; y <= offset; y++) {
                        if(((wiID % inputWidth) + y) < 0 || (int(wiID / inputWidth) + 1 + x) < 0) {
                            continue;
                        }
                        auto r_curr_pix = wiID;
                        auto g_curr_pix = wiID + pixCount;
                        auto b_curr_pix = wiID + 2 * pixCount;
                        globalFinalImg[wiID * channels] += globalRGB[r_curr_pix + x * inputWidth + y] * globalBlur[offset + x][offset + y];// for R
                        globalFinalImg[wiID * channels + 1] += globalRGB[g_curr_pix + x * inputWidth + y] * globalBlur[offset + x][offset + y];// for G
                        globalFinalImg[wiID * channels + 2] += globalRGB[b_curr_pix + x * inputWidth + y] * globalBlur[offset + x][offset + y];// for B
                    }
                }
            });
        });
        end_time = chrono::high_resolution_clock::now();
        std::cout << "Parallel method's time: " << chrono::duration_cast<chrono::microseconds>((end_time - start_time) / 1000).count() << 
        " microseconds" << std::endl;


        auto globalFinalImg = finalIMG_buffer.get_access<access::mode::read>();
        for(int i = 0; i < pixCount; i++) {
            *(outputData + i * channels) = globalFinalImg[i * channels];
            *(outputData + i * channels + 1) = globalFinalImg[i * channels + 1];
            *(outputData + i * channels + 2) = globalFinalImg[i * channels + 2];
        }
            stbi_write_jpg("wot_sycl.jpg", inputWidth, inputHeight, channels, outputData, 100);

    } 
}
