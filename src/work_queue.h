// Copyright 2013 15418 Course Staff.
// modified by Ken Ling

#ifndef __WORKER_WORK_QUEUE_H__
#define __WORKER_WORK_QUEUE_H__

#include <pthread.h>
#include <vector>


template <class T>
class WorkQueue {
private:
  std::vector<T> storage;
  pthread_mutex_t queue_lock;
  // pthread_cond_t queue_cond;

public:

  WorkQueue() {
    // pthread_cond_init(&queue_cond, NULL);
    pthread_mutex_init(&queue_lock, NULL);
  }

  T get_work(bool &rt) {
    pthread_mutex_lock(&queue_lock);
    if (storage.size() == 0) {
      rt = false;
      pthread_mutex_unlock(&queue_lock);
      return T();
    } else {
      rt = true;
      T item = storage.front();
      storage.erase(storage.begin());
      pthread_mutex_unlock(&queue_lock);
      return item;
    }
    // while (storage.size() == 0) {
    //   pthread_cond_wait(&queue_cond, &queue_lock);
    // }

    // T item = storage.front();
    // //storage.pop_front();
    // storage.erase(storage.begin());

    // pthread_mutex_unlock(&queue_lock);
    // return item;
  }

  void put_work(const T& item) {
    pthread_mutex_lock(&queue_lock);
    storage.push_back(item);
    pthread_mutex_unlock(&queue_lock);
    // pthread_cond_signal(&queue_cond);
  }
};

#endif  // WORKER_WORK_QUEUE_H_