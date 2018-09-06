#include "Cache.h"
#include "Cache_impl.h"

template<>
Cache<cv::Mat>* Cache<cv::Mat>::instance = nullptr;

template<>
size_t Cache<cv::Mat>::getSize(const cv::Mat& obj)
{
   return obj.total() * obj.elemSize();
}
