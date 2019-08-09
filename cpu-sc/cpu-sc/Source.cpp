#include <mkl.h>
#include <iostream>
#include "cpu-sc.h"
#include "test-cases.h"

int main()
{
	 float a[4]{1 ,2 ,3, 4};
	 float b[4]{1, 2, 3, 4};
   
	 auto res = cblas_sdot(4, a, 1, b, 1);
	 std::cout << res << std::endl;

   // testEigen1();
	 // testLaplacian1();
   // 
	 // testLoadData1();
	 // testLoadData2();
	 // testLoadData3();
   testKMeans1();
   // testGroup();
}