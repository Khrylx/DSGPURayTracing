#include <cstdio>
#include <pthread.h>

void *thread(void *vargp);

int main(int argc, char const *argv[])
{
	pthread_t tid[5];
	for (int i = 0; i < 5; i++) {
		pthread_create(tid + i, NULL, thread, (void*)i);
	}
	for (int i = 0; i < 5; i++) {
		pthread_join(tid[i], NULL);
	}
	printf("done\n");
	return 0;
}

void *thread(void *vargp) {
	printf("%ld Hello World\n", (long long)vargp);
	return NULL;
}