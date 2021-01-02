#include "iostream"
#include "gpuDAVertexer.h"

int main()
{
	gpuDAVertexer::DAVertexer demoVertexer(5.5);
	demoVertexer.makeAsync(nullptr);

	return 0;
}
