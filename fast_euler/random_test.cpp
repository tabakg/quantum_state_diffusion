/*
g++ random_test.cpp -o random_test -std=c++11
./random_test
*/


#include <iostream>
#include <string>
#include <random>
#include <chrono>
#include <eigen3/Eigen/Sparse>

using namespace std;
using namespace chrono;
using namespace Eigen;

typedef std::complex<float> comp;
const std::complex<float> imag_unit(0.0,1.0);

void get_new_randoms(vector<vector<comp>> & randoms,
                     int size1,
                     int size2,
                     std::default_random_engine & gen erator,
                     std::normal_distribution<float> & distribution){
  for (int i=0; i<size1; ++i) {
    for (int j=0; j<size2; ++j) {
      randoms[i][j] = distribution(generator) + imag_unit * distribution(generator);
    }
  }
}


int main()
{
  int seed = 10;
  std::default_random_engine generator (seed);
  std::normal_distribution<float> distribution(0., 1.);
  int size1 = 1000;
  int size2 = 10;

  std::vector<std::vector<comp>> randoms(size1, std::vector<comp>(size2));
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  get_new_randoms(randoms, size1, size2, generator, distribution);

  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast< duration<double> >(t2 - t1);
  cout << "Number of experiments: " << size1*size2 << endl;
  cout << "It took me " << time_span.count() *1000000<< " nano seconds." << endl;
  cout << "Average rate per experiment: " << time_span.count() *1000000 / (size1*size2)<< " nano seconds.";
  cout << endl;
  cout << endl;

  return 0;
}
