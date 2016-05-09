#pragma once
#define DATA_SIZE 4096

#define tileSize 256

struct Request
{
	Request() {};
	int seq;
	int x;
	int y;
	int xRange;
	int yRange;
};

struct Result
{
	uint32_t data[DATA_SIZE];
};
