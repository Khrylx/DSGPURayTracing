#include <atomic>
#include <cstdio>
#include <pthread.h>
#include <sys/socket.h>
#include <cstdlib>
#include <semaphore.h>
#include "csapp.h"
#include "work_queue.h"

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

	while(rio_readlineb(&rio, buf, MAXLINE) > 0) {
		printf("client read: %s\n", buf);
		sleep(1);
		sprintf(tmpBuf, "processed %s", buf);
		rio_writen(clientfd, tmpBuf, strlen(tmpBuf));
	}
	close(clientfd);
	return 0;
}
