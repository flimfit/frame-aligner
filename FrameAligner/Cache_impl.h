#pragma once
#include "Cache.h"
#include <opencv2/core.hpp>
#include <iostream>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include "lz4_filter.hpp"

template<typename T>
CachedObject<T>::CachedObject() :
   cache(nullptr)
{
   initRef();
}

template<typename T>
CachedObject<T>::CachedObject(const CachedObject& other)
{
   (*this) = other;
}

template<typename T>
CachedObject<T>& CachedObject<T>::operator=(const CachedObject& other)
{
   cache = other.cache;
   id = other.id;
   fcn = other.fcn;
   filename = other.filename;
   refs = other.refs;
   (*refs)++;
   return *this;
};

template<typename T>
CachedObject<T>::CachedObject(CachedObject&& other)
{
   (*this) = other;
};

template<typename T>
CachedObject<T>& CachedObject<T>::operator=(CachedObject&& other)
{
   cache = other.cache;
   id = other.id;
   fcn = other.fcn;
   filename = other.filename;
   refs = other.refs;
   other.refs = nullptr;
   return *this;
};

template<typename T>
CachedObject<T>::~CachedObject()
{
   if (!refs) return;

   (*refs)--;
   if ((*refs) == 0)
   {
      cache->remove(id);
      if (!filename.empty())
      {
         std::cout << "Deleting:" << boost::filesystem::canonical(filename).string() << "\n";
         boost::filesystem::remove(filename);
      }
      delete refs;
   }
}

template<typename T>
CachedObject<T>::CachedObject(Cache<T>* cache, size_t id, generator_fcn fcn) :
   cache(cache), id(id), fcn(fcn)
{
   initRef();
}

template<typename T>
CachedObject<T>::CachedObject(Cache<T>* cache, size_t id, const T& obj) :
   cache(cache), id(id)
{
   initRef();
   safe_to_clear = false;
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
   //in.push(ext::bio::lz4_decompressor());
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
   //out.push(ext::bio::lz4_compressor());
   out.push(bio::zlib_compressor(bio::zlib::best_speed)); 
   out.push(ofs);
   boost::archive::binary_oarchive oa(out);
   oa << obj;
   safe_to_clear = true;
}

template<typename T>
void CachedObject<T>::initRef()
{
   refs = new size_t;
   *refs = 1;
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
Cache<T>::Cache(precache_t precache, size_t cache_size_) :
   precache(precache), cache_size(cache_size_), current_size(0)
{
   next_id = 0;
   if (cache_size == 0)
      cache_size = static_cast<size_t>(getMemorySize() / 4);
}

template <typename T>
CachedObject<T> Cache<T>::add(generator_fcn fcn)
{
   std::lock_guard<std::mutex> lk(m);
   CachedObject<T> cache_obj(this, next_id++, fcn);
   return cache_obj;
}

template <typename T>
CachedObject<T> Cache<T>::add(const T& obj)
{
   std::lock_guard<std::mutex> lk(m);
   size_t id = next_id++;
   CachedObject<T> cache_obj(this, id, obj);
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
   std::lock_guard<std::mutex> lk(m);

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
      obj = fcn();
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
