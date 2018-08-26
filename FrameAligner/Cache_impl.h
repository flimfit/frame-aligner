#pragma once
#include "Cache.h"
#include <opencv2/core.hpp>
#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <ctime>

template<typename T>
CachedObject<T>::~CachedObject()
{
   cache->remove(id);
}

template<typename T>
CachedObject<T>::CachedObject(Cache<T>* cache, size_t id, generator_fcn fcn) :
   cache(cache), id(id), fcn(fcn)
{
}

template<typename T>
T CachedObject<T>::get() const
{
   return cache->get(id, fcn);
}

namespace bio = boost::iostreams;
namespace fs = boost::filesystem;

template<typename T>
T Cache<T>::read(const std::string& filename) const
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
void Cache<T>::write(const std::string& filename, const T& obj)
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


inline bool is_older_than(const boost::filesystem::path& path, int hrs)
{
   return std::difftime(std::time(0), boost::filesystem::last_write_time(path)) > (hrs * 3600);
}

inline std::vector<boost::filesystem::path> files_older_than(boost::filesystem::path dir, int hrs)
{
   std::vector<boost::filesystem::path> result;

   for (const auto& p : boost::filesystem::recursive_directory_iterator(dir))
      if (boost::filesystem::is_regular_file(p) && is_older_than(p, hrs)) result.push_back(p);

   return result;
}

inline std::pair<int, bool> remove_files_older_than(boost::filesystem::path dir, unsigned int days)
{
   int cnt = 0;
   try
   {
      for (const auto& p : files_older_than(dir, days * 24))
      {
         boost::filesystem::remove(p);
         ++cnt;
      }
      return { cnt, true };
   }
   catch (const std::exception&) { return { cnt, false }; }
}

template <typename T>
Cache<T>::Cache(size_t cache_size_) :
   cache_size(cache_size_), current_size(0)
{
   next_id = 0;
   if (cache_size == 0)
      //      cache_size = 1LL << 31; // 2Gb
      cache_size = 0.5 * getMemorySize();

   // Create temporary folder
   temp_path = boost::filesystem::temp_directory_path() / "galene";
   if (!boost::filesystem::exists(temp_path))
      boost::filesystem::create_directory(temp_path);

   // Clean up any old left over temporary files from crash
   remove_files_older_than(temp_path, 1);

   deletor_thread = std::thread(&Cache<T>::deletor, this);
}




template <typename T>
Cache<T>::~Cache()
{
   terminate = true;
   if (deletor_thread.joinable())
      deletor_thread.join();
}

template <typename T>
std::shared_ptr<CachedObject<T>> Cache<T>::add(generator_fcn fcn)
{
   auto cache_obj = std::make_shared<CachedObject<T>>(this, next_id++, fcn);

   if (current_size < 0.8 * cache_size)
      cache_obj->get();

   return cache_obj;
}

template <typename T>
std::shared_ptr<CachedObject<T>> Cache<T>::add(const T& obj)
{
   std::lock_guard<std::recursive_mutex> lk(m);
   size_t id = next_id++;

   std::string f = "galene-" + std::to_string(id) + "-%%%%-%%%%-%%%%-%%%%.bin";
   std::string filename = (temp_path / boost::filesystem::unique_path(f)).string();
   filestore[id] = filename;
   auto fcn = std::bind(&Cache<T>::read, this, filename);

   auto cache_obj = std::make_shared<CachedObject<T>>(this, id, fcn);

   insert(id, obj);
   return cache_obj;
}

template <typename T>
void Cache<T>::insert(size_t id, const T& obj)
{
   std::unique_lock<std::recursive_mutex> lk(m);

   store[id] = obj;
   queue.push_front(id);

   current_size += getSize(obj);

   std::unique_lock<std::mutex> del_lk(deletor_mutex);
   while (current_size > cache_size)
   {
      size_t id = queue.back();
      queue.pop_back();

      auto it = store.find(id);
      if (it != store.end())
      {
         deletor_queue.push(id);
         current_size -= getSize(it->second);
      }
   }
   del_lk.unlock();
   deletor_cv.notify_one();
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
   std::unique_lock<std::recursive_mutex> lk(m);
   std::unique_lock<std::mutex> del_lk(deletor_mutex);


   auto qit = std::find(queue.begin(), queue.end(), id);
   if (qit != std::end(queue))
   {
      deletor_queue.push(*qit);
      queue.erase(qit);
   }

   auto fit = filestore.find(id);
   if (fit != filestore.end())
   {
      CachedObjectFileStore& file = fit->second;
      if (file.commited)
         fs::remove(file.filename);
      filestore.erase(fit);
   }

   del_lk.unlock();
   deletor_cv.notify_one();
}

template <typename T>
void Cache<T>::deletor()
{
   while (!(terminate && deletor_queue.empty()))
   {
      std::unique_lock<std::mutex> del_lk(deletor_mutex);
      deletor_cv.wait(del_lk, [&] { return !deletor_queue.empty(); });

      while (!deletor_queue.empty())
      {
         size_t id = deletor_queue.front();
         deletor_queue.pop();
         del_lk.unlock();

         auto it = store.find(id);

         if (it != store.end())
         {
            auto fit = filestore.find(id);
            if (fit != filestore.end())
               if (!fit->second.commited)
               {
                  write(fit->second.filename, it->second);
                  fit->second.commited = true;
               }

            std::unique_lock<std::recursive_mutex> lk(m);
            store.erase(it);
            lk.unlock();
         }

         del_lk.lock();
      }
   }
}

template <typename T>
size_t Cache<T>::getSize(const T& obj)
{
   static_assert(false, "Please provide a concrete implementation for getSize");
}

