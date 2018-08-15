#pragma once
#include "Cache.h"
#include <opencv2/core.hpp>
#include <iostream>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include "lz4_filter.hpp"

template<typename T>
CachedObject<T>::~CachedObject()
{
   cache->remove(id);
   if (!filename.empty())
   {
      std::cout << "Deleting:" << boost::filesystem::canonical(filename).string() << "\n";
      boost::filesystem::remove(filename);
   }
}

template<typename T>
CachedObject<T>::CachedObject(Cache<T>* cache, size_t id, generator_fcn fcn) :
   cache(cache), id(id), fcn(fcn)
{
}

template<typename T>
CachedObject<T>::CachedObject(Cache<T>* cache, size_t id, const T& obj) :
   cache(cache), id(id)
{
   std::string x = "galene-" + std::to_string(id) + "-%%%%-%%%%-%%%%-%%%%.bin";
   filename = boost::filesystem::temp_directory_path() / boost::filesystem::unique_path(x);
   write(obj);
}

template<typename T>
T CachedObject<T>::get() const
{
   if (filename.empty())
      return cache->get(id, fcn);
   else
      return cache->get(id, std::bind(&CachedObject<T>::read, this));
}

namespace bio = boost::iostreams;
namespace ext { namespace bio = ext::boost::iostreams; }

template<typename T>
T CachedObject<T>::read() const
{
   std::ifstream ifs(filename.c_str(), std::ifstream::binary);
   bio::filtering_streambuf<bio::input> in;
   in.push(bio::zlib_decompressor()); 
   in.push(ifs);
   boost::archive::binary_iarchive ia(in);

   T obj;
   ia >> obj;
   return obj;
}

template<typename T>
void CachedObject<T>::write(const T& obj)
{
   std::ofstream ofs(filename.c_str(), std::ofstream::binary);
   bio::filtering_streambuf<bio::output> out;
   out.push(bio::zlib_compressor(bio::zlib::best_speed)); 
   out.push(ofs);
   boost::archive::binary_oarchive oa(out);
   oa << obj;
}


// Cache Impl
// =========================

template<class T>
Cache<T>* Cache<T>::getInstance()
{
   if (instance == nullptr)
      instance = new Cache<T>();
   return instance;
}

template <typename T>
Cache<T>::Cache(size_t cache_size_) :
   cache_size(cache_size_), current_size(0)
{
   next_id = 0;
   if (cache_size == 0)
      cache_size = static_cast<size_t>(1024 * 1024 * 512); // 512Mb
}

template <typename T>
std::shared_ptr<CachedObject<T>> Cache<T>::add(generator_fcn fcn)
{
   auto cache_obj = std::make_shared<CachedObject<T>>(this, next_id++, fcn);
   return cache_obj;
}

template <typename T>
std::shared_ptr<CachedObject<T>> Cache<T>::add(const T& obj)
{
   std::lock_guard<std::recursive_mutex> lk(m);
   size_t id = next_id++;
   auto cache_obj = std::make_shared<CachedObject<T>>(this, id, obj);
   insert(id, obj);
   return cache_obj;
}

template <typename T>
void Cache<T>::insert(size_t id, const T& obj)
{
   store[id] = obj;
   queue.push_front(id);

   current_size += getSize(obj);

   while (current_size > cache_size)
   {
      size_t id = queue.back();
      queue.pop_back();
      auto it = store.find(id);
      current_size -= getSize(it->second);
      store.erase(it);
   }
}

template <typename T>
T Cache<T>::get(size_t id, generator_fcn fcn)
{
   std::unique_lock<std::recursive_mutex> lk(m);
   T obj;

   auto it = store.find(id);
   if (it != store.end())
   {
      obj = it->second;
      queue.remove(id);
      queue.push_front(id);
   }
   else
   {
      lk.unlock();
      obj = fcn();
      lk.lock();
      insert(id, obj);
   }
   return obj;
}

template <typename T>
void Cache<T>::remove(size_t id)
{
   auto it = store.find(id);
   if (it != store.end()) store.erase(it);

   auto qit = std::find(queue.begin(), queue.end(), id);
   if (qit != std::end(queue)) queue.erase(qit);
}

template <typename T>
size_t Cache<T>::getSize(const T& obj)
{
   static_assert(false, "Please provide a concrete implementation for getSize");
}
