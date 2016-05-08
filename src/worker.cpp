#include <atomic>
#include <cstdio>
#include <pthread.h>
#include <sys/socket.h>
#include <cstdlib>
#include <semaphore.h>
#include "csapp.h"
#include "work_queue.h"
#include "common.h"

int main(int argc, char *argv[])
{
	int clientfd;
	char *host, *port, buf[MAXLINE], tmpBuf[MAXLINE];
	rio_t rio;

	host = argv[1];
	port = argv[2];

	clientfd = open_clientfd(host, port);
	Rio_readinitb(&rio, clientfd);
	printf("Connected to server\n");

	int sizeRequest = sizeof(struct Request);
	int sizeResult = sizeof(struct Result);

	char requestBuf[sizeRequest];
	char resultBuf[sizeResult];
	
	Request req;
	Result result;

	while(rio_readnb(&rio, requestBuf, sizeRequest) > 0) {
		memcpy(&req, requestBuf, sizeRequest);
		int dataSize = req.xRange * req.yRange;
		int k = 0;
		printf("worker process START [x: %d, y: %d, xRange: %d, yRange: %d]\n", req.x, req.y, req.xRange, req.yRange);
		// process request
		for (int y = req.y; y < req.y + req.yRange; y++) {
			for (int x = req.x; x < req.x + req.xRange; x++) {
				result.data[k % DATA_SIZE] = y * scene_width + x;
				k++;
				if (k % DATA_SIZE == 0 || k == dataSize) {
					memcpy(resultBuf, &result, sizeResult);
					rio_writen(clientfd, resultBuf, sizeResult);
				}
			}
		}
		printf("worker process  DONE [x: %d, y: %d, xRange: %d, yRange: %d]\n", req.x, req.y, req.xRange, req.yRange);
		printf("worker result\n");
		for (int j = 0; j < DATA_SIZE; j++) {
			printf("%d ", result.data[j]);
		}
		printf("\n");
		sleep(1);
	}
	close(clientfd);
	return 0;
}
