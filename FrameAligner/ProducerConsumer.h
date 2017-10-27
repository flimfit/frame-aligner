#pragma once

#include "Pool.h"
#include <future>
#include <queue>
#include <functional>

template<typename T, typename FP, typename FC>
void ProducerConsumer(size_t n_producer, FP producer, FC consumer, size_t n)
{
   std::map<int, T> buffer;
   std::mutex m;
   std::condition_variable cv;
   std::atomic<int> pi = 0;
   std::vector<std::future<void>> t_producer;

   for(int p=0; p<n_producer; p++)
      t_producer.push_back(std::async([&]() {
         int i;
         while((i = pi++) < n)
         {
            T obj = producer(i);

            std::unique_lock<std::mutex> lk(m);
            buffer[i] = obj;
            lk.unlock();
            cv.notify_all();

            // Keep the buffer from getting too large
            //lk.lock();
            //cv.wait(lk, [&] { return buffer.size() < 5; });
            //lk.unlock();
         }
      }));

   auto t_consumer = std::async([&]() {
      for (size_t i = 0; i < n; i++)
      {
         std::unique_lock<std::mutex> lk(m);
         cv.wait(lk, [&] { return buffer.count(i) == 1; });
         T obj = buffer[i];
         buffer.erase(i);
         lk.unlock();
         cv.notify_all();

         consumer(i, obj);
      }
   });

   for(auto& t : t_producer)
      t.get();

   t_consumer.get();
}