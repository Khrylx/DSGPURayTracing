#include <atomic>
#include <cstdio>
#include <pthread.h>
#include <sys/socket.h>
#include <cstdlib>
#include <semaphore.h>
#include "csapp.h"
#include "work_queue.h"
// thread based master node

void *listen_thread(void *vargp);
void *process(void *vargp);
void echo(int connfd);

WorkQueue<int> workQueue;
std::atomic<int> threadCount;
sem_t complete_sem;

int main(int argc, char *argv[])
{
	threadCount = 0;
	sem_init(&complete_sem, false, 1);

	pthread_t tid;
	pthread_create(&tid, NULL, listen_thread, (void*)(argv[1]));
	
	for (int i = 0; i < 100; i++) {
		workQueue.put_work(i);
	}

	// put work into queue
	int k = 3;
	while(1) { // work there is still work in work queue
		// process work
		bool rt;
		int val = workQueue.get_work(rt);
		if (!rt) {
			break;
		}
		printf("main: %d\n", val);
		if (k >= 0) {
			k--;
			sleep(2);
		}

	}
	while (threadCount != 0) {
		sem_wait(&complete_sem);
	}
	// all threads have been exited; all work are done
	// output image
	return 0;
}



void *listen_thread(void *vargp) {
	int listenfd, *connfdp;
	socklen_t clientlen;
	struct sockaddr_storage clientaddr;
	pthread_t tid;
	printf("%s\n", (char*)vargp);
	listenfd = open_listenfd((char*)vargp);

	while (true) {
		clientlen = sizeof(struct sockaddr_storage);
		connfdp = (int*)malloc(sizeof(int));
		*connfdp = accept(listenfd, (SA*)&clientaddr, &clientlen);
		pthread_create(&tid, NULL, process, connfdp);
	}

	return NULL;
}

void *process(void *vargp) {
	threadCount++;

	int connfd = *((int *)vargp);
	pthread_detach(pthread_self());
	free(vargp);
	// echo(connfd);
	while(1) {
		bool rt;
		int val = workQueue.get_work(rt);
		if (!rt) {
			break;
		}
		printf("thread: %d\n", val);
		sleep(10);
	}
	close(connfd);

	threadCount--;
	if (threadCount == 0) {
		sem_post(&complete_sem);
	}
	return NULL;
}

void echo(int connfd) {
	size_t n;
	char buf[MAXLINE]; rio_t rio;
	Rio_readinitb(&rio, connfd);
	while((n = Rio_readlineb(&rio, buf, MAXLINE)) != 0) {
		printf("server received %d bytes\n", (int)n); Rio_writen(connfd, buf, n);
	}
}