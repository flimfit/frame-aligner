#pragma once

#include "Pool.h"
#include <future>
#include <queue>
#include <functional>

template<typename T, typename FP, typename FC>
void ProducerConsumer(FP producer, FC consumer, size_t n)
{
   std::queue<T> buffer;
   std::mutex m;
   std::condition_variable cv;

   auto t_producer = std::async([&]() {
      for (size_t i = 0; i < n; i++)
      {
         T obj = producer(i);

         std::unique_lock<std::mutex> lk(m);
         cv.wait(lk, [&] { return buffer.size() < 5; });
         buffer.push(obj);
         lk.unlock();
         cv.notify_all();
      }
   });

   auto t_consumer = std::async([&]() {
      for (size_t i = 0; i < n; i++)
      {
         std::unique_lock<std::mutex> lk(m);
         cv.wait(lk, [&] { return !buffer.empty(); });
         T obj = buffer.front();
         buffer.pop(); 
         lk.unlock();
         cv.notify_all();

         consumer(i, obj);
      }
   });

   t_producer.get();
   t_consumer.get();
}