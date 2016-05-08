#define DATA_SIZE 256

#define scene_width 1000
#define scene_height 1000
#define tileSize 16

struct Request
{
	Request() {};

	int x;
	int y;
	int xRange;
	int yRange;
};

struct Result
{
	uint32_t data[DATA_SIZE];
};
