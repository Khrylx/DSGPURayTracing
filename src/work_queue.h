#ifndef __WORK_QUEUE_H__
#define __WORK_QUEUE_H__

#include <condition_variable>
#include <mutex>
#include <vector>

template <class T>
class WorkQueue {
private:
    std::vector<T> storage;
    std::mutex lock;

public:

    WorkQueue() {}

    bool is_empty() {
        lock.lock();
        bool result = storage.empty();
        lock.unlock();
        return result;
    }

    T get_work() {
        lock.lock();
        T item = storage.front();
        storage.erase(storage.begin());
        lock.unlock();
        return item;
    }

    void put_work(const T& item) {
        lock.lock();
        storage.push_back(item);
        lock.unlock();
    }

    void clear() {
      lock.lock();
      storage.clear();
      lock.unlock();
  }
};

#endif  // WORK_QUEUE_H_
