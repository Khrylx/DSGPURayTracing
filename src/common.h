#pragma once
#define DATA_SIZE 4096

#define tileSize 256

struct Request
{
	Request() {};
	float a;
	float b;
	float c;
	int x;
	int y;
	int xRange;
	int yRange;
};

struct Result
{
	uint32_t data[DATA_SIZE];
};
