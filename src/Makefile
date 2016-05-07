#
# Makefile for Proxy Lab 
#
# You may modify is file any way you like (except for the handin
# rule). Autolab will execute the command "make" on your specific 
# Makefile to build your proxy from sources.
#
CC = g++
CFLAGS = -g -std=c++11
# CFLAGS = -g -Wall
# LDFLAGS = -lpthread
# The original flag is -lpthread!!!
LDFLAGS = -lpthread

all: master

csapp.o: csapp.c
	$(CC) $(CFLAGS) -c csapp.c

#work_queue.o: work_queue.h
#	$(CC) $(CFLAGS) -c work_queue.h

master.o: master.cpp
	$(CC) $(CFLAGS) -c master.cpp

master: csapp.o master.o

# Creates a tarball in ../proxylab-handin.tar that you should then
# hand in to Autolab. DO NOT MODIFY THIS!

clean:
	rm -f *~ *.o proxy core *.tar *.zip *.gzip *.bzip *.gz

