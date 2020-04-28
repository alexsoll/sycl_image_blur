#include <iostream>
#include <vector>
#include <CL/sycl.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace cl::sycl;
using namespace std;

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
    unsigned char *r = new unsigned char[pixels_count * sizeof(unsigned char)];
    unsigned char *g = new unsigned char[pixels_count * sizeof(unsigned char)];
    unsigned char *b = new unsigned char[pixels_count * sizeof(unsigned char)];
    unsigned char *rgb = new unsigned char[img_size * sizeof(unsigned char)];
    unsigned char *(r_) = r, *(g_) = g, *(b_) = b;
    for(size_t i = 0; i < pixels_count; i++, (r_)++, (g_)++, (b_)++) {
        *(r_) = (uint8_t)(*(img + i * channels));
        *(g_) = (uint8_t)(*(img + i * channels + 1));
        *(b_) = (uint8_t)(*(img + i * channels + 2));
    }
    memcpy(rgb, r, pixels_count * sizeof(unsigned char));
    unsigned char *tmp = rgb + pixels_count;
    memcpy(tmp, g, pixels_count * sizeof(unsigned char));
    *tmp = *(tmp + pixels_count);
    memcpy(tmp, b, pixels_count * sizeof(unsigned char));
    /*
        Native method
    */
    unsigned char *final_image = new unsigned char[img_size * sizeof(unsigned char)];
    unsigned char *final_image_tmp = final_image;
    unsigned char *r_curr_pix, *g_curr_pix, *b_curr_pix;
    int capacity = (filter_size - 1) / 2;
    for(int i = capacity; i <= height - capacity - 1; i++) {
        for(int j = capacity; j <= width - capacity - 1; j++) {
            vector<double> sum(3, 0);
            for(int k = (-1) * capacity; k <= capacity; k++) {
                for(int t = (-1) * capacity; t <= capacity; t++) {
                    r_curr_pix = r + i * width + j;
                    g_curr_pix = g + i * width + j;
                    b_curr_pix = b + i * width + j;
                    sum[0] += (uint8_t)(double(*(r_curr_pix + k * width + t)) * filter[capacity + k][capacity + t]);
                    sum[1] += (uint8_t)(double(*(g_curr_pix + k * width + t)) * filter[capacity + k][capacity + t]);
                    sum[2] += (uint8_t)(double(*(b_curr_pix + k * width + t)) * filter[capacity + k][capacity + t]);
                }
            }
            *final_image_tmp++ = (uint8_t)(sum[0]);
            *final_image_tmp++ = (uint8_t)(sum[1]);
            *final_image_tmp++ = (uint8_t)(sum[2]);
            sum.clear();
        }
    }

    // for(unsigned char *p = img, *pg = gray_img; p != img + img_size; p += channels, pg += gray_channels) {
    //     *pg = (uint8_t)((*p + *(p + 1) + *(p + 2))/3.0);
    // }

    stbi_write_jpg("test_test_test.jpg", width - filter_size + 1, height - filter_size + 1, channels, final_image, 100);

}
