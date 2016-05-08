#include <atomic>
#include <cstdio>
#include <pthread.h>
#include <sys/socket.h>
#include <cstdlib>
#include <semaphore.h>
#include "csapp.h"
#include "task_queue.h"
#include "common.h"
// thread based master node

void *listen_thread(void *vargp);
void *process(void *vargp);
void generate_work();
void master_process_request(Request req);

uint32_t output[scene_height][scene_width];
int sizeRequest = sizeof(struct Request);
int sizeResult = sizeof(struct Result);

TaskQueue<Request> workQueue;
std::atomic<int> threadCount;
sem_t complete_sem;

int main(int argc, char *argv[])
{
	threadCount = 0;
	sem_init(&complete_sem, false, 1);

	pthread_t tid;
	pthread_create(&tid, NULL, listen_thread, (void*)(argv[1])); // listening for worker connection
	
	// put work into queue
	generate_work();
	
	int k = 3; // simulate some processing time

	while(1) {
		// process work
		bool rt;
		Request req = workQueue.get_work(rt);
		if (!rt) { // work queue is empty
			break;
		}
		printf("master process START [x: %d, y: %d, xRange: %d, yRange: %d]\n", req.x, req.y, req.xRange, req.yRange);
		master_process_request(req);

		if (k >= 0) { // simulate some processing time
			k--;
			sleep(2);
		}
		printf("master process  Done[ x: %d, y: %d, xRange: %d, yRange: %d]\n", req.x, req.y, req.xRange, req.yRange);
	}
	while (threadCount != 0) {
		sem_wait(&complete_sem);
	}
	// all threads have been exited; all work are done
	// output image
	printf("Output image\n");
	for (int r = 0; r < scene_height; r++) {
		for (int c = 0; c < scene_width; c++) {
			printf("%u ", output[r][c]);
		}
		printf("\n");
	}
	return 0;
}

void generate_work() {
	int rowNum = (scene_height + tileSize - 1) / tileSize;
	int colNum = (scene_width + tileSize - 1) / tileSize;

	for (int r = 0; r < rowNum; r++) {
		for (int c = 0; c < colNum; c++) {
			Request req;
			req.x = c * tileSize;
			req.y = r * tileSize;
			req.xRange = std::min(tileSize, scene_width - req.x);
			req.yRange = std::min(tileSize, scene_height - req.y);
			workQueue.put_work(req);
		}
	}
}

void master_process_request(Request req) {
	for (int y = req.y; y < req.y + req.yRange; y++) {
		for (int x = req.x; x < req.x + req.xRange; x++) {
			output[y][x] = y * scene_width + x;
		}
	}
}

void *listen_thread(void *vargp) {
	int listenfd, *connfdp;
	socklen_t clientlen;
	struct sockaddr_storage clientaddr;
	pthread_t tid;

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

	char requestBuf[sizeRequest];
	char resultBuf[sizeResult];
	Result result;

	rio_t rio;
	rio_readinitb(&rio, connfd);

	while(1) {
		bool rt;
		Request req = workQueue.get_work(rt);
		if (!rt) { // work queue is empty
			break;
		}
		printf("thread process START [x: %d, y: %d, xRange: %d, yRange: %d]\n", req.x, req.y, req.xRange, req.yRange);
		memcpy(requestBuf, &req, sizeRequest);
		rio_writen(connfd, requestBuf, sizeRequest);

		// worker node process request and return result

		// compute how much result to receive
		int dataSize = req.xRange * req.yRange;
		int k = 0;

		// receive result from worker
		for (int y = req.y; y < req.y + req.yRange; y++) {
			for (int x = req.x; x < req.x + req.xRange; x++) {
				if (k % DATA_SIZE == 0) {
					rio_readnb(&rio, resultBuf, sizeResult);
					memcpy(&result, resultBuf, sizeResult);
				}
				output[y][x] = result.data[k % DATA_SIZE];
				k++;
			}
		}

		printf("thread process  DONE [x: %d, y: %d, xRange: %d, yRange: %d]\n", req.x, req.y, req.xRange, req.yRange);
		printf("thread result\n");
		for (int i = 0; i < DATA_SIZE; i++) {
			printf("%d ", result.data[i]);
		}
		printf("\n");
	}
	close(connfd);

	threadCount--;
	if (threadCount == 0) {
		sem_post(&complete_sem);
	}
	return NULL;
}
