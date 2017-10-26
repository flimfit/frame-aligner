#pragma once 

#include <memory>
#include <stack>
#include <mutex>
#include <condition_variable>


template<class T, class U>
class Pool
{
private:
   struct External_Deleter 
   {
     explicit External_Deleter(std::weak_ptr<Pool<T,U>*> pool_ = std::weak_ptr<Pool<T, U>*>())
         : pool(pool_) {}
 
     void operator()(T* ptr) 
     {
       if (auto pool_ptr = pool.lock()) 
       {
         try {
           (*pool_ptr.get())->release(std::unique_ptr<T>{ptr});
           return;
         } catch(...) {}
       }
       std::default_delete<T>{}(ptr);
     }
    private:
     std::weak_ptr<Pool<T,U>* > pool;
   };

public:
   
   using ptr_type = std::unique_ptr<T, External_Deleter >;

   Pool() :
   this_ptr(new Pool<T,U>*(this)) 
   {};

   Pool(const U& init) : 
      init(init), this_ptr(new Pool<T,U>*(this))
   { 
      pool.push( std::make_unique<T>(init) );
   }

   void setInit(const U& init_)
   {
      init = init_;
   }

   ptr_type get()
   {
      std::unique_lock<std::mutex> lk(m);
      
      if (pool.empty())
      {
         try
         {
            pool.push( std::make_unique<T>(init) );                     
         } catch (...)
         {
            cv.wait(lk,[&]{ return !pool.empty(); });            
         }
      }
      
      ptr_type tmp(pool.top().release(),
         External_Deleter{std::weak_ptr<Pool<T,U>*>{this_ptr}});
      
      pool.pop();   
      return std::move(tmp);
   }

   void release(std::unique_ptr<T> el)
   {
      std::unique_lock<std::mutex> lk(m);
      pool.push(std::move(el));
      lk.unlock();

      cv.notify_all();
   }

private:
   std::shared_ptr<Pool<T,U>* > this_ptr;
   std::stack<std::unique_ptr<T>> pool;
   std::mutex m;
   std::condition_variable cv;
   U init;
};