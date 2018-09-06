#pragma once
#include <memory>
#include <mutex>
#include <map>
#include <list>
#include <queue>
#include <atomic>
#include <algorithm>
#include <functional>
#include <thread>
#include <boost/filesystem.hpp>
#include <iostream>
#include <boost/filesystem/fstream.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

extern size_t getMemorySize();

class CachedObjectFileStore
{
public:
   CachedObjectFileStore(std::string filename = "") : filename(filename) {}

   std::string filename;
   bool commited = false;
};

template<typename T>
class CachedObject;

/*
   T should be a reference counting shared object, e.g. a shared_ptr<U>, cv::Mat
   Otherwise things are going to get ugly !
*/
template<typename T>
class Cache
{
   typedef std::function<T()> generator_fcn;

public:

   static Cache<T>* getInstance();
   ~Cache<T>();

   std::shared_ptr<CachedObject<T>> add(generator_fcn fcn);
   std::shared_ptr<CachedObject<T>> add(const T& fcn);

protected:

   Cache(size_t cache_size = 0);
   T get(size_t id, generator_fcn fcn);
   void remove(size_t id);
   void insert(size_t id, const T& obj);
   void eraseBack();
   void deletor();

   void write(const std::string& filename, const T& obj);
   T read(const std::string& filename) const;

   size_t getSize(const T& obj);

   size_t cache_size;
   size_t current_size;
   std::map<size_t, T> store;
   std::map<size_t, CachedObjectFileStore> filestore;
   std::list<size_t> queue;
   std::queue<size_t> deletor_queue;
   std::atomic<size_t> next_id;
   std::recursive_mutex m;

   std::thread deletor_thread;
   std::condition_variable deletor_cv;
   std::mutex deletor_mutex;
   bool terminate = false;
   
   boost::filesystem::path temp_path;

   static Cache<T>* instance;

   friend class CachedObject<T>;
};


template<typename T>
class CachedObject
{
   typedef std::function<T()> generator_fcn;

public:

   CachedObject(Cache<T>* cache, size_t id, generator_fcn fcn);
   ~CachedObject();

   T get() const;

private:
   Cache<T>* cache;
   size_t id;
   generator_fcn fcn;
   friend class Cache<T>;
};
