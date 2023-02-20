#ifndef __JSON_HELPER_H
#define __JSON_HELPER_H
 
#include <time.h>
#include <string>
#include <vector>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <boost/shared_ptr.hpp>
 

template<typename T>
class jsoner {
 public:
  static std::string to_json (std::string const& name, T const& value) {
    std::stringstream stream;
    stream << "{\"" << name << "\":";
    stream << value.to_json();
    stream << "}";
    return stream.str();
  }
};


template<>
class jsoner<std::vector<std::string> > {
 public:
  static std::string to_json (std::string const& name, std::vector<std::string> const & value) {
    std::vector<std::string>::const_iterator itor, last = value.end();
    std::stringstream stream;
    stream << "{\"" << name << "\":[";
    int i = 0;
    for (itor = value.begin(); itor != last; ++itor) {
      stream << "{";
      stream << "\"index\":" << "\"" << i << "\",";
      stream << "\"value\":" << "\"" << *itor << "\"";
      stream << "}";
      if(itor != last -1) {
	        stream << ",";
      }
      ++i;
    }
    stream << "]}";
    return stream.str();
  }
 
};

template<>
class jsoner<std::vector<cv::Point> > {
public:
  static std::string to_json (std::string const& name, std::vector<cv::Point>const & value) {
    std::vector<cv::Point>::const_iterator it, last = value.end();
    std::stringstream stream;
    stream << "{\"" << name << "\":[";
    int i = 0;
    for(it=value.begin();it!=last;++it){   
        stream << "["<<it->x<<","<<it->y<<"]";
        if(it!=(last-1))
          stream<<",";   
    }
    stream<<"]}";

  return stream.str();
};

// template<>
// class jsoner<std::vector<std::vector<cv::Point>>> {
// public:
//   static std::string to_json (std::string const& name, std::vector<std:vector<cv::Point>>const & value) {
//     std::vector<std:vector<cv::Point>>::const_iterator it, last = value.end();
//     std::stringstream stream;
//     stream << "{\"" << name << "\":[";
//     int i = 0;
    
//     for(it=value.begin();it!=last;++it){
//       stream << "[";
//       std:vector<cv::Point>::const_iterator itor, end = *it.end();
//       for (itor = *it.begin(); itor != end; ++itor) {    
//         stream << "["<<itor->x<<","<<itor->y<<"]";
//         if(itor!=(end-1))
//           stream<<",";
//       }
//       stream << "]";
//       if(it!=(last-1))
//         stream<<","
//     }
//     stream<<"]}";

//   return stream.str();
// };

 
template<typename T>
class jsoner<std::vector<boost::shared_ptr<T> > > {
 public:
  static std::string to_json (std::string const& name, std::vector<boost::shared_ptr<T> > const & value) {
    typename std::vector<boost::shared_ptr<T> >::const_iterator itor, last = value.end();
    std::stringstream stream;
    stream << "{\"" << name << "\":[";
    int i = 0;
    for (itor = value.begin(); itor != last; ++itor) {
      stream << "{";
      stream << "\"index\":" << "\"" << i << "\",";
      stream << "\"value\":" << (*itor)->to_json();
      stream << "}";
      if(itor != last -1) {
	stream << ",";
      }
      ++i;
    }
    stream << "]}";
    return stream.str();
  }
 
};
 
 
#endif
 