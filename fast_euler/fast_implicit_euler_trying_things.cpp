/*
g++ fast_implicit_euler.cpp -o fast_implicit_euler -std=c++11
./fast_implicit_euler
*/

#include <iostream>
#include <eigen3/Eigen/Sparse>
// #include <pybind11.h>
#include <vector>
#include <chrono>
#include <thread>
#include <future>

using namespace std;
using namespace Eigen;
using namespace chrono;

const complex<float> i(0.0,1.0);
typedef complex<float> comp;
typedef vector<comp> comp_vec;
typedef vector<float> float_vec;
typedef SparseMatrix< complex<float> > SpMat; // declares a column-major sparse matrix type of double
typedef SparseVector< complex<float> > SpVec; // declares a sparse vector type of double
typedef vector<SpVec> vec_of_SpVec;


// int add(int i, int j) {
//     return i + j;
// }
//
// namespace py = pybind11;
//
// PYBIND11_MODULE(python_example, m) {
//
//     m.def("add", &add, R"pbdoc(
//         Add two numbers
//     )pbdoc");
//
// }

SpMat f(SpMat& H, SpVec& psi){
  return -i*H*psi;
}

float norm(SpVec& psi){
  float s = 0.;
  for (int i = 0; i < psi.size(); i++){
    s += norm(psi.coeff(i));
  }
  return sqrt(s);
}

void run_simulation(SpMat& H, vec_of_SpVec& traj, float& delta_t, int& timesteps){
  for(int i=0; i < timesteps-1; i++){
    traj[i+1] = traj[i] + delta_t * f(H, traj[i]);
    traj[i+1] /= norm(traj[i+1]);
  }
}

void adds_vecs(comp_vec& in_1, comp_vec& in_2, comp_vec& out){
  // plies two array component-wise.
  transform(in_1.begin(), in_1.end(), in_2.begin(), out.begin(), plus<comp>() );
}

void mult_vecs(comp_vec* in_1, comp_vec* in_2, comp_vec* out){
  // Multiplies two array component-wise.
  transform(in_1->begin(), in_1->end(), in_2->begin(), out->begin(), multiplies<comp>() );
}


void mult_vecs_offset_upper(comp_vec& diag, comp_vec& vec, comp_vec& out, int& offset){
  /* Multiplies arrays with an offset.

  Should be equivalent to multiplying a sparse matrix with offset upper diagonal.

  IMPORTANT: This does not populate the last `offset` components of out for efficiency; they should be zero.
  */
  transform(diag.begin(), diag.end(), vec.begin()+offset, out.begin(), multiplies<comp>() );
}

void mult_vecs_offset_lower(comp_vec& diag, comp_vec& vec, comp_vec& out, int& offset){
  /* Multiplies arrays with an offset.

  Should be equivalent to multiplying a sparse matrix with offset lower diagonal.

  IMPORTANT: This does not populate the first `offset` components of out for efficiency; they should be zero.
  */
  transform(diag.begin(), diag.end(), vec.begin(), out.begin()+offset, multiplies<comp>() );
}

int main()
{

  // int timesteps = 10;
  // float delta_t = 0.1;
  // int dim = 3;
  //
  // SpMat H(dim,dim);
  // H.insert(0,0) = 0.1;
  // H.insert(1,1) = 0.2;
  // H.insert(2,2) = 0.3;
  //
  // vec_of_SpVec traj(timesteps);
  //
  // traj[0] = SpVec(dim);
  //
  // traj[0].insert(0) = 1.;
  // traj[0].insert(1) = 0.;
  // traj[0].insert(2) = 0.;
  //
  // run_simulation(H, traj, delta_t, timesteps);
  // cout << "H = " << H << endl;
  //
  // cout << "traj_norm = " << endl;
  // for(int i = 0; i < timesteps; i++){
  //   cout << norm(traj[i]) << endl;
  // }

  // initialize vector for inputs

  int size = 2500;

  float_vec v;
  float_vec* p_v = &v;
  v.reserve(size);
  for(int n=0; n < size; n++)
    v.push_back(n);

  comp_vec u;
  comp_vec* p_u = &u;
  u.reserve(size);
  for(int n=0; n < size; n++)
    u.push_back(n);

  comp_vec w;
  comp_vec* p_w = &w;
  w.reserve(size);


  // int num_threads = 5;
  // vector <comp_vec*> ws;
  // for(int i=0; i< num_threads; i++){
  //   // initialize vector for outputs
  //   comp_vec w;
  //   comp_vec* p_w = &w;
  //   w.reserve(size);
  //   ws.push_back(p_w);
  // }

  // // initialize vector for summing outputs
  // comp_vec y;
  // y.resize(size);

  // vector<std::thread> mult_workers;
  // mult_workers.resize(num_threads);

  for(int k=0; k < 30; k++){

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // mult_workers.clear();

    // mult_vecs(p_v, p_v, p_w);
    // std::thread mult_worker_w(mult_vecs, p_v, p_v, p_w);
    // std::thread mult_worker_x(mult_vecs, p_v, p_v, p_x);
    //
    // mult_worker_w.join();
    // mult_worker_x.join();
    mult_vecs(p_u, p_u, p_w);

    // for(int i=0; i< num_threads; i++){
    //   mult_workers.push_back(std::thread(mult_vecs, p_v, p_v, ws[i]));
    // }
    //
    // for(int i=0; i< num_threads; i++){
    //   mult_workers[i].join();
    // }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast< duration<double> >(t2 - t1);
    cout << "It took me " << time_span.count() *1000000<< " nano seconds.";
    cout << endl;
    }
}
