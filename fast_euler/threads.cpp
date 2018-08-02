/*
g++ threads.cpp -o threads -std=c++11
./threads
*/

// thread example
#include <iostream>       // std::cout
#include <thread>         // std::thread

using namespace std;

void foo()
{
  cout << "random function!! \n" << endl;
}

void bar(int *x)
{
  cout << "got pointer with address " << x << endl;
  cout << "x == " << *x << endl;
}

int main()
{
  int* num;
  *num = 0;
  std::thread first (foo);     // spawn new thread that calls foo()
  std::thread second (bar, num);  // spawn new thread that calls bar(0)

  std::cout << "main, foo and bar now execute concurrently...\n";

  // synchronize threads:
  first.join();                // pauses until first finishes
  second.join();               // pauses until second finishes

  std::cout << "foo and bar completed.\n";

  return 0;
}
