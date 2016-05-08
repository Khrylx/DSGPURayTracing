#define DATA_SIZE 16

#define scene_width 10
#define scene_height 10
#define tileSize 4

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
