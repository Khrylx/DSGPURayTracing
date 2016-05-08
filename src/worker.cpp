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

	while(rio_readnb(&rio, buf, MAXLINE) > 0) {
		float *floatBuf = (float*)buf;
		// printf("client read: %s\n", buf);
		printf("client read: %f\n", floatBuf[0]);
		sleep(1);
		// sprintf(tmpBuf, "processed %s", buf);
		sprintf(tmpBuf, "processed %f", floatBuf[0]);

		rio_writen(clientfd, tmpBuf, MAXLINE);
	}
	close(clientfd);
	return 0;
}
