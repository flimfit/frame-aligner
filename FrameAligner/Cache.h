#pragma once
#include <memory>
#include <mutex>
#include <map>
#include <list>
#include <atomic>
#include <algorithm>
#include <functional>
#include <boost/filesystem.hpp>
#include <iostream>
#include <boost/filesystem/fstream.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

extern size_t getMemorySize();

template<typename T>
class CachedObject;

/*
   T should be a reference counting shared object, e.g. a shared_ptr<U>, cv::Mat
   Otherwise things are going to get ugly !
*/
template<typename T>
class Cache
{
   enum precache_t { PRECACHE, NO_PRECACHE };

   typedef std::function<T()> generator_fcn;

public:

   static Cache<T>* getInstance();

   CachedObject<T> add(generator_fcn fcn);
   CachedObject<T> add(const T& fcn);

protected:

   Cache(precache_t precache = NO_PRECACHE, size_t cache_size = 0);
   T get(size_t id, generator_fcn fcn);
   void remove(size_t id);
   void insert(size_t id, const T& obj);
   
   size_t getSize(const T& obj);

   size_t cache_size;
   size_t current_size;
   std::map<size_t, T> store;
   std::list<size_t> queue;
   std::atomic<size_t> next_id;
   precache_t precache;
   std::mutex m;

   static Cache<T>* instance;

   friend class CachedObject<T>;
};


template<typename T>
class CachedObject
{
   typedef std::function<T()> generator_fcn;

public:

   CachedObject();

   CachedObject(const CachedObject& other);
   CachedObject& operator=(const CachedObject& other);

   CachedObject(CachedObject&& other);
   CachedObject& operator=(CachedObject&& other);
   ~CachedObject();

   T get() const;

   operator T() const { return get(); }

protected:

   CachedObject(Cache<T>* cache, size_t id, generator_fcn fcn);
   CachedObject(Cache<T>* cache, size_t id, const T& obj);

   void write(const T& obj);
   T read() const;

   void initRef();


private:
   Cache<T>* cache;
   size_t id;
   generator_fcn fcn;
   boost::filesystem::path filename;
   size_t* refs;
   friend class Cache<T>;
   bool safe_to_clear = true;
};
