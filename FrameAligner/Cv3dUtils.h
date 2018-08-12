#pragma once
#include <opencv2/opencv.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/binary_object.hpp>
#include <numeric>
extern const int X;
extern const int Y;
extern const int Z;


cv::Mat extractSlice(const cv::Mat& m, int slice);
int area(const cv::Mat& m);

void writeScaledImage(const std::string&, const cv::Mat& m);

double correlation(cv::Mat &image_1, cv::Mat &image_2, cv::Mat &mask);
void interpolatePoint3d(const std::vector<cv::Point3d>& Ds, std::vector<cv::Point3d>& D);

void inpaint3d(const cv::Mat& input, const cv::Mat& mask, cv::Mat& output);

BOOST_SERIALIZATION_SPLIT_FREE(cv::Mat)

namespace boost {
   namespace serialization {

      /** Serialization support for cv::Mat */
      template<class Archive>
      void save(Archive & ar, const cv::Mat& m, const unsigned int version)
      {
         size_t elem_size = m.elemSize();
         int elem_type = m.type();

         std::vector<int> size;
         size.assign(m.dims, 0);
         for (int i = 0; i < m.dims; i++)
            size[i] = m.size[i];

         ar & size;
         ar & elem_size;
         ar & elem_type;

         const size_t data_size = m.total() * elem_size;
         boost::serialization::binary_object mat_data(m.data, data_size);
         ar & mat_data;
      }

      /** Serialization support for cv::Mat */
      template<class Archive>
      void load(Archive & ar, cv::Mat& m, const unsigned int version)
      {
         size_t elem_size;
         int elem_type;
         std::vector<int> size;

         ar & size;
         ar & elem_size;
         ar & elem_type;

         m.create(size, elem_type);

         const size_t data_size = m.total() * elem_size;
         boost::serialization::binary_object mat_data(m.data, data_size);
         ar & mat_data;
      }

   }
}
/*
namespace boost {
   namespace serialization {

      template<class Archive>
      void serialize(Archive &ar, cv::Mat& mat, const unsigned int)
      {
         int type, dims;
         std::vector<int> size;
         bool continuous;

         if (Archive::is_saving::value) {
            type = mat.type();
            dims = mat.dims;
            size.assign(dims, 0);
            for (int i = 0; i < dims; i++)
               size[i] = mat.size[i];
            continuous = mat.isContinuous();
         }

         ar & type & continuous & size;

         if (Archive::is_loading::value)
            mat.create(size, type);

         if (continuous) {
            const size_t data_size = mat.total() * mat.elemSize();
            ar & boost::serialization::make_array(mat.ptr(), data_size);
         }
         else {
            throw std::runtime_error("must be continuous");
            //const size_t row_size = cols * mat.elemSize();
            //for (int i = 0; i < rows; i++) {
            //   ar & boost::serialization::make_array(mat.ptr(i), row_size);
           // }
         }

      }

   } // namespace serialization
} // namespace boost
*/