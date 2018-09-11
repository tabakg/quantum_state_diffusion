
/*

// Worked on Mac (used on my local machine):
g++ fast_sim.cpp -o fast_sim -std=c++11

spec_file="/scratch/users/tabakg/qsd_output/json_spec/tmp_file.json"
psis_out="/scratch/users/tabakg/qsd_output/fast_out/tmp_output.json"
expects_out="/scratch/users/tabakg/qsd_output/fast_out/tmp_output_expects.json"

./fast_sim "$spec_file" "$psis_out" "$expects_out"

// Worked on Linux (i.e. Sherlock):
g++ fast_sim.cpp -o fast_sim -std=c++11 -pthread

spec_file="/Users/gil/Google Drive/repos/quantum_state_diffusion/num_json_specifications/tmp_file.json"
psis_out="/Users/gil/Google Drive/repos/quantum_state_diffusion/num_json_specifications/tmp_output.json"
expects_out="/Users/gil/Google Drive/repos/quantum_state_diffusion/num_json_specifications/tmp_output_expects.json"

./fast_sim "$spec_file" "$psis_out" "$expects_out"
*/

// basic file operations
#include <iostream>
#include <fstream>
#include <json.hpp>
#include <random>
#include <thread>
#include <math.h>       /* sqrt */
#include <assert.h>     /* assert */
#include <chrono>
#include <complex>      // std::complex
#include <stdlib.h>     /* exit, EXIT_FAILURE */
#include <iomanip>

#ifdef DEBUG
#define DEBUG_MSG(str) do { std::cout << str << std::endl; } while( false )
#else
#define DEBUG_MSG(str) do { } while ( false )
#endif

using json = nlohmann::json;
using namespace std::chrono;

typedef double num_type;
typedef std::string string;
typedef std::complex<num_type> comp;
const std::complex<num_type> imag_unit(0.0,1.0);
typedef std::vector<comp> comp_vec;

/*
The following vectors inputs_one_system and inputs_two_systems are
used to check the input type to make sure it is compatible. */

std::vector<string> inputs_one_system = {"num_systems",
                                         "H_eff",
                                         "Ls",
                                         "psi0",
                                         "duration",
                                         "delta_t",
                                         "sdeint_method",
                                         "obsq",
                                         "downsample",
                                         "ntraj",
                                         "seeds"};

std::vector<string> inputs_two_systems = {"num_systems",
                                          "H_eff",
                                          "Ls",
                                          "L2_dag",
                                          "L2_dag_L1",
                                          "psi0",
                                          "duration",
                                          "delta_t",
                                          "sdeint_method",
                                          "obsq",
                                          "downsample",
                                          "ntraj",
                                          "seeds",
                                          "R",
                                          "T",
                                          "eps",
                                          "n",
                                          "lambda"};

// struct to hold operators, represented by diagonals and offsets.
// This is used for H_eff (multiplied by -i) and the Ls.
// In the two system case we will hold L2_dag and L2_dag_L1 as well.

typedef struct{
  std::vector<comp_vec> diags;
  std::vector<int> offsets;
  int size; // number of diagonals
} diag_op;

// When using one system we contain the data in the following struct.

typedef struct{
  // psi0
  comp_vec psi0;
  int dimension;

  // Other parameters
  num_type duration;
  num_type delta_t;
  string sdeint_method;
  int downsample;
  int traj_num;

  // H_eff
  diag_op H_eff;
  std::vector<comp_vec> H_eff_x_psi;

  // Ls
  std::vector<diag_op> Ls;
  std::vector<std::vector<comp_vec>> Ls_diags_x_psi;
  std::vector<comp> Ls_expectations;

  // obsq
  std::vector<diag_op> obsq;
} one_system;

// When using two systems we contain the data in the following struct.

typedef struct{
  // psi0
  comp_vec psi0;
  int dimension;

  // Other parameters
  num_type duration;
  num_type delta_t;
  string sdeint_method;
  int downsample;
  int traj_num;

  num_type R; // reflectivity
  num_type T;
  num_type eps; // classical transmission amplification
  num_type n; // fictitious noise transmission amplification
  num_type lambda; // Filtering parameter

  // H_eff
  diag_op H_eff;
  std::vector<comp_vec> H_eff_x_psi;

  // Ls
  std::vector<diag_op> Ls;
  std::vector<std::vector<comp_vec>> Ls_diags_x_psi;
  std::vector<comp> Ls_expectations;

  // other operators for transmission
  diag_op L2_dag;
  std::vector<comp_vec> L2_dag_x_psi;

  diag_op L2_dag_L1;
  std::vector<comp_vec> L2_dag_L1_x_psi;

  //transmission between two systems
  comp alpha_t; // alpha(t)= R*l_1(t) + dW_t^{(2)}
  comp alpha_old; // alpha(t-1)
  comp alpha_t_filtered; // filtered version
  // alpha_t_filtered = (1-lambda) * alpha(t) + lambda * alpha(t-1)

  // obsq
  std::vector<diag_op> obsq;
} two_system;


void write_to_file(json j, string output_file){
  // Write json object j to file output_file
    std::ofstream o(output_file);
    o << std::setw(4) << j << std::endl;
}


void read_from_file(string & input_file, json & j){
  // Read from input file input_file to json object j.
  string line;
  std::ifstream myfile (input_file);
  if (myfile.is_open())
  {
    while ( getline (myfile, line) )
    {
      j.merge_patch(json::parse(line));
    }
    myfile.close();
  }
  else std::cout << "Unable to open file: " << input_file << std::endl;
}


bool matches_inputs(json & j, std::vector<string> inputs){
  // check if all inputs are fields of the json j.
  json entry;
  for(int i=0; i < std::min(inputs.size(), j.size()); i++){
    entry = inputs[i];
    if (j.find(entry) == j.end()) {
      std::cout << "Non-matching entry found: " << entry << std::endl;
      return false;
    }
  }
  return true;
}


int get_num_systems(json & j){
  // figures out the number of systems encoded based on what the inputs are.
  // Assumes that there are no inputs matching BOTH one and two systems.
  if (j["num_systems"] == 1 and matches_inputs(j, inputs_one_system))
    return 1;
  else if (j["num_systems"] == 2 and matches_inputs(j, inputs_two_systems))
    return 2;
  else return -1;
}


comp_vec json_to_complex_array(json & j_arr, num_type scaling = 1.){
  // Converts JSON object to complex vector.

  // Assumes a json serialization with fields "real" and "imag".
  // These should be arrays of the same size containing num_types.
  // Output is a std::vector with complex<num_type> entries.
  comp_vec arr;
  int size = j_arr["real"].size();
  arr.reserve(size);
  num_type real_part;
  num_type imag_part;
  for(int n=0; n < size; n++){
    real_part = j_arr["real"][n];
    imag_part = j_arr["imag"][n];
    arr.push_back(real_part * scaling + imag_unit * imag_part * scaling);
  }
  return arr;
}


template <class iterable>
json complex_array_to_json(iterable vec){
  // Converts object to JSON.

  // When using a vector of complex vectors or vector of vectors of complex vectors,
  // this template results in calling each member recursively.
  json j;
  int size = vec.size();
  for(int i=0; i<size; i++){
    j.push_back(complex_array_to_json(vec[i]));
  }
  return j;
}


template<>
json complex_array_to_json(comp_vec vec){
  // Converts complex vector to JSON.

  // output JSON object has two fields "real" and "imag", with values
  // corresponding to two numerical arrays of the same size.
  int size = vec.size();
  std::vector<num_type> real_part(size);
  std::vector<num_type> imag_part(size);
  for(int i=0; i<size; i++){
    real_part[i] = std::real(vec[i]);
    imag_part[i] = std::imag(vec[i]);
  }
  json j_real(real_part);
  json j_imag(imag_part);
  json j_arr;
  j_arr["real"] = j_real;
  j_arr["imag"] = j_imag;
  return j_arr;
}


void load_diag_operator(diag_op & op, json & diag_json, num_type scaling=1.){
  // Loads operators from JSON with fields "data" and "offsets" representing a
  // sparse matrix.
  json json_data = diag_json["data"];
  json json_offsets = diag_json["offsets"];
  int size = json_data.size();
  assert(size == json_offsets.size());

  op.size = size;
  for (int i = 0; i < size; i++){
    op.diags.push_back(json_to_complex_array(json_data[i], scaling=scaling));
    op.offsets.push_back(json_offsets[i]);
  }
}


void load_diag_operator_sequence(std::vector<diag_op> & ops,
                                 json & diag_json,
                                 num_type scaling=1.){
  // Loads a vector of operators from a JSON object.
  int size = diag_json.size();
  ops = std::vector<diag_op>(size);
  for (int i = 0; i < size; i++){
    load_diag_operator(ops[i], diag_json[i], scaling=scaling);
  }
}


void get_new_randoms(std::vector<std::vector<comp>> & randoms,
                     int size1,
                     int size2,
                     std::default_random_engine & generator,
                     std::normal_distribution<num_type> & distribution){
  // Generate complex random numbers with shape (size1, size2).
  for (int i=0; i<size1; ++i) {
    for (int j=0; j<size2; ++j) {
      randoms[i][j] = distribution(generator) + imag_unit * distribution(generator);
    }
  }
}


void add_second_to_first(comp_vec& in_1, comp_vec& in_2){
  // Adds second to first
  int size = in_1.size();
  for (int i=0; i<size; i++){
    in_1[i] += in_2[i];
  }
}


void add_second_to_first(comp_vec& in_1, comp_vec& in_2, comp scalar){
  // Adds scalar * second to first
  int size = in_1.size();
  for (int i=0; i<size; i++){
    in_1[i] += in_2[i] * scalar;
  }
}


void add_second_to_first(comp_vec& vec, std::vector<comp_vec> & vecs_to_add){
  // Adds every vector in vecs_to_add to vec
  for (int i=0; i < vecs_to_add.size(); i++){
    add_second_to_first(vec, vecs_to_add[i]);
  }
}

void add_second_to_first(comp_vec& vec, std::vector<comp_vec> & vecs_to_add, comp scalar){
  // Adds every vector in vecs_to_add to vec (multiplied by scalar)
  for (int i=0; i < vecs_to_add.size(); i++){
    add_second_to_first(vec, vecs_to_add[i], scalar);
  }
}


void mult_vecs_offset_upper(comp_vec& diag, comp_vec& vec, comp_vec& out, int& offset){
  /* Multiplies arrays with an offset.

  Should be equivalent to multiplying a sparse matrix with offset upper diagonal.

  IMPORTANT: This adds values to existing "out" vector; may have to zero them first.
  */
  DEBUG_MSG(std::cout << "offset " << offset << std::endl);
  DEBUG_MSG(std::cout << "diag.size() " << diag.size() << std::endl);

  transform(diag.begin()+offset, diag.end(), vec.begin()+offset, out.begin(), std::multiplies<comp>() );
}


void mult_vecs_offset_lower(comp_vec& diag, comp_vec& vec, comp_vec& out, int& offset){
  /* Multiplies arrays with an offset.

  Should be equivalent to multiplying a sparse matrix with offset lower diagonal.

  IMPORTANT: This adds values to existing "out" vector; may have to zero them first.
  */
  DEBUG_MSG(std::cout << "offset " << offset << std::endl);
  DEBUG_MSG(std::cout << "diag.size() " << diag.size() << std::endl);

  transform(diag.begin(), diag.end()-offset, vec.begin(), out.begin()+offset, std::multiplies<comp>() );
}


comp dot(comp& z1, comp& z2){
  // returns z1.conj() * z2, NOT standard Euclidean dot product.
  num_type a, b, c, d;
  a = std::real(z1); b = std::imag(z1);
  c = std::real(z2); d = std::imag(z2);
  return comp(a * c + b * d, a * d - b * c);
}


comp dot_vecs(comp_vec& v1, comp_vec& v2){
  // computes v1.dag() * v2
  comp val(0., 0.);
  int size = v1.size();
  for (int i=0; i<size; i++){
    val += dot(v1[i], v2[i]);
  }
  return val;
}


comp norm(std::vector<comp>& vec){
  // Computes norm of vec.
  comp val = dot_vecs(vec, vec);
  return sqrt(val);
}


void normalize(std::vector<comp>& vec, comp& total){
  // normalize vector vec by total
  int size = vec.size();
  for(int i=0; i<size; i++){
    vec[i] /= total;
  }
}


void normalize(std::vector<comp>& vec){
  // normalize vector vec by its norm.
  comp total = norm(vec);
  normalize(vec, total);
}


void update_products(diag_op& op,
                     comp_vec& current_psi,
                     std::vector<comp_vec>& products){
  // Compute the products of each operator diagonal multiplied by psi,
  // taking into account the offset of each diagonal.
  // Should be idempotent.

  // TODO: separate data for positive and negative offsets to clean this up

  for (int i=0; i<products.size(); i++){
    std::fill(products[i].begin(), products[i].end(), comp(0., 0.));
    int offset = op.offsets[i];
    if (offset >= 0){
      mult_vecs_offset_upper(op.diags[i],
                             current_psi,
                             products[i],
                             offset);
    }
    else{
      offset *= -1;
      mult_vecs_offset_lower(op.diags[i],
                             current_psi,
                             products[i],
                             offset);
    }
  }
}


void update_products_sequence(std::vector<diag_op>& ops,
                              comp_vec& current_psi,
                              std::vector<std::vector<comp_vec>>& products){
  // Applies update_products to a sequence of operators and products.
  // Should be idempotent.

  for (int i=0; i<ops.size(); i++){
    update_products(ops[i], current_psi, products[i]);
  }
}


comp expectation_from_vecs(std::vector<comp_vec>& product,
                            comp_vec& current_psi){
  // Computes the expectation values <psi|op|psi> from the products op|psi>
  // and the vector psi.
  // Should be idempotent.

  comp expectation(0., 0.);
  for (int i=0; i<product.size(); i++){
    expectation += dot_vecs(current_psi, product[i]);
    DEBUG_MSG(std::cout << "current expectation: " << expectation << std::endl);
  }
  return expectation;
}


void update_expects_from_vecs(std::vector<std::vector<comp_vec>>& products,
                              std::vector<comp>& expectations,
                              comp_vec& current_psi){
  // Update the expectation values <psi|op|psi> from the products op|psi>
  // for a sequence of operators.
  // Should be idempotent.

  int size = products.size();
  for (int i=0; i<size; i++){
    DEBUG_MSG(std::cout << "old expectations[i]: " << expectations[i] << std::endl);
    expectations[i] = expectation_from_vecs(products[i], current_psi);
    DEBUG_MSG(std::cout << "new expectations[i]: " << expectations[i] << std::endl);
  }
}


std::vector<comp> expectations_from_ops(std::vector<diag_op>& ops, comp_vec& psi){
  // Compute the expectation values <psi|op|psi> for each op in ops,
  // directly from ops and psi.
  // Should be idempotent.

  int num_operators = ops.size();
  int dimension = psi.size();
  std::vector<std::vector<comp_vec>> products(num_operators);
  for (int i=0; i<num_operators; i++){
    products[i] = std::vector<comp_vec>(ops[i].size,
      std::vector<comp>(dimension));
  }

  std::vector<comp> expectations(num_operators);
  update_products_sequence(ops, psi, products);
  for (int i=0; i<num_operators; i++){
    expectations[i] += expectation_from_vecs(products[i], psi);
  }
  return expectations;
}


void update_alpha(two_system& system, std::vector<comp>& noise){
  // find alpha_t and alpha_t_filtered, but does NOT update alpha_old.
  // should be idempotent.

  system.alpha_t = system.R * system.Ls_expectations[0] + noise[1];
  system.alpha_t_filtered = (1.0 - system.lambda) * system.alpha_t + system.lambda * system.alpha_old;
}


void step_alpha(two_system& system){
  // Update alpha_old to current filtered alpha.
  // Note this is NOT idempotent.
  system.alpha_old = system.alpha_t_filtered;
}


void update_structures(one_system& system,
                       std::vector<comp>& current_psi){
  // Update H_eff * psi and for each L, L*psi and l = <psi|L|psi>
  // Should be idempotent.

  update_products(system.H_eff, current_psi, system.H_eff_x_psi);
  update_products_sequence(system.Ls, current_psi, system.Ls_diags_x_psi);
  update_expects_from_vecs(system.Ls_diags_x_psi, system.Ls_expectations, current_psi);
}


void update_structures(two_system& system,
                       std::vector<comp>& noise,
                       std::vector<comp>& current_psi){
  // Update H_eff * psi and for each L, L*psi and l = <psi|L|psi>.
  // Also update L2_dag * psi and L2_dag_L1 * psi, and the current alpha.
  // Should be idempotent.

  update_products(system.H_eff, current_psi, system.H_eff_x_psi);
  update_products(system.L2_dag, current_psi, system.L2_dag_x_psi);
  update_products(system.L2_dag_L1, current_psi, system.L2_dag_L1_x_psi);
  update_products_sequence(system.Ls, current_psi, system.Ls_diags_x_psi);
  update_expects_from_vecs(system.Ls_diags_x_psi, system.Ls_expectations, current_psi);
  update_alpha(system, noise);
}


void update_psi(one_system& system, std::vector<comp>& noise, std::vector<comp>& current_psi){
  // Update psi for one system using the system state and noise.
  // IMPORTANT: this is NOT idempotent.

  // Add Hamiltonian component
  add_second_to_first(current_psi, system.H_eff_x_psi);

  // Add L components, including noise terms
  comp mult_L_by;
  for (int i=0; i<system.Ls_diags_x_psi.size(); i++){
    mult_L_by = std::conj(system.Ls_expectations[i]) + noise[i];
    add_second_to_first(current_psi, system.Ls_diags_x_psi[i], mult_L_by);
  }
}


void update_psi(two_system& system, std::vector<comp>& noise, std::vector<comp>& current_psi){
  // Update psi for one system using the system state and noise.
  // IMPORTANT: this is NOT idempotent.

  // Add Hamiltonian component
  add_second_to_first(current_psi, system.H_eff_x_psi);

  comp coefficient; // coefficient for each term to add to psi

  // Add L1 component
  coefficient = (std::conj(system.Ls_expectations[0])
                + system.T * std::conj(system.Ls_expectations[1])
                + system.T * noise[0]
                + system.R * std::conj(noise[1]));
  add_second_to_first(current_psi, system.Ls_diags_x_psi[0], coefficient);

  // Add L2 component
  coefficient = (std::conj(system.Ls_expectations[1])
                + system.T * std::conj(system.Ls_expectations[0])
                + noise[0]
                + system.eps * std::conj(system.alpha_t_filtered));
  add_second_to_first(current_psi, system.Ls_diags_x_psi[1], coefficient);

  // Add L2_dag component
  coefficient = - system.eps * system.alpha_t_filtered;
  add_second_to_first(current_psi, system.L2_dag_x_psi, coefficient);

  // Add L2_dag_L1 component
  coefficient = - system.T;
  add_second_to_first(current_psi, system.L2_dag_L1_x_psi, coefficient);

  // Add L components, including noise terms, for other Ls (except first two)
  for (int i=2; i<system.Ls_diags_x_psi.size(); i++){
    coefficient = std::conj(system.Ls_expectations[i]) + noise[i];
    add_second_to_first(current_psi, system.Ls_diags_x_psi[i], coefficient);
  }
}


template<class T>
void update_and_normalize_psi(T& system, std::vector<comp>& noise, std::vector<comp>& current_psi){
  // Convenient function to update and normalize psi.
  update_psi(system, noise, current_psi);
  normalize(current_psi);
}


void take_euler_step(one_system& system, std::vector<comp>& noise, std::vector<comp>& current_psi){
  // Euler step for one system.
  update_structures(system, current_psi);
  update_and_normalize_psi(system, noise, current_psi);
}


void take_euler_step(two_system& system, std::vector<comp>& noise, std::vector<comp>& current_psi){
  // Euler step for two systems.
  // Unlike the one system case, we have to step alpha in time.
  update_structures(system, noise, current_psi);
  update_and_normalize_psi(system, noise, current_psi);
  step_alpha(system);
}


void take_implicit_euler_step(one_system& system, std::vector<comp>& noise, std::vector<comp>& current_psi, int& steps){
  // Implicit Euler step for one system.
  if (steps == 1){
    take_euler_step(system, noise, current_psi);
    return;
  }

  /*    If steps >= 2   */

  // temporary state next_psi
  comp_vec next_psi(system.dimension);
  std::copy(current_psi.begin(), current_psi.end(), next_psi.begin());

  // First step: update system structures and next_psi using current_psi
  update_structures(system, current_psi);
  update_and_normalize_psi(system, noise, next_psi);

  // Intermediate steps: update system structures and next_psi using next_psi
  for (int i=0; i<steps-2; i++){
    update_structures(system, current_psi);
    std::copy(current_psi.begin(), current_psi.end(), next_psi.begin());
    update_and_normalize_psi(system, noise, next_psi);
  }

  // Last step: update system structures and current_psi using next_psi
  update_structures(system, current_psi);
  update_and_normalize_psi(system, noise, current_psi);
}


void take_implicit_euler_step(two_system& system, std::vector<comp>& noise, std::vector<comp>& current_psi, int& steps){
  // Implicit Euler step for two systems.
  // This looks similar to the one system case, but there are a few small differences.
  if (steps == 1){
    take_euler_step(system, noise, current_psi);
    return;
  }

  /*    If steps >= 2   */

  // temporary state next_psi
  comp_vec next_psi(system.dimension);
  std::copy(current_psi.begin(), current_psi.end(), next_psi.begin());

  // First step: update system structures and next_psi using current_psi
  update_structures(system, noise, current_psi);
  update_and_normalize_psi(system, noise, next_psi);

  // Intermediate steps: update system structures and next_psi using next_psi
  for (int i=0; i<steps-2; i++){
    update_structures(system, noise, current_psi);
    std::copy(current_psi.begin(), current_psi.end(), next_psi.begin());
    update_and_normalize_psi(system, noise, next_psi);
  }

  // Last step: update system structures and current_psi using next_psi
  update_structures(system, noise, current_psi);
  update_and_normalize_psi(system, noise, current_psi);
  step_alpha(system);
}


template<class system_type>
void run_trajectory(system_type * system_ptr, int seed, int steps_for_noise,
                    std::vector<std::vector<comp>> * psis,
                    std::vector<std::vector<comp>> * expects,
                    int implicit_euler_steps=3){

  system_type system = *system_ptr;

  // Find dimension...
  system.dimension = system.psi0.size();

  // make generator for random numbers
  std::default_random_engine generator(seed);

  // Number of complex noise terms.
  int num_noise = system.Ls.size();

  // total steps, including initial state.
  int num_steps = int(system.duration / system.delta_t);
  int num_downsampled_steps = int(system.duration / (system.delta_t * system.downsample));

  // accepts num_type, double, or long double
  num_type mean = 0;
  num_type std_dev = sqrt(0.5);
  std::normal_distribution<num_type> distribution(0., std_dev);

  // random variables stored here. They are replenished every steps_for_noise steps.
  std::vector<std::vector<comp>> randoms(steps_for_noise, std::vector<comp>(num_noise));

  std::vector<comp> current_psi = system.psi0;

  //////// Main for loop for simulation
  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  std::copy(current_psi.begin(), current_psi.end(), (*psis)[0].begin());

  for(int i=0, j=0, k=0, l=0; i<num_steps; i++, j++, k++){
    // Get new noise terms as we go
    if (j == steps_for_noise)
      j = 0;
    if (k == system.downsample)
      k = 0;
    if (j == 0)
      get_new_randoms(randoms, steps_for_noise, num_noise, generator, distribution);
    if (system.sdeint_method == "ItoEuler")
      take_euler_step(system, randoms[j], current_psi);
    else if (system.sdeint_method == "itoImplicitEuler")
      take_implicit_euler_step(system, randoms[j], current_psi, implicit_euler_steps);
    else{
      std::cout << "sdeint_method " << system.sdeint_method << " not supported." << std::endl;
      break;
    }
    if (k == 0){
      l++; // number of psis recorded
      std::copy(current_psi.begin(), current_psi.end(), (*psis)[l].begin());
      // show_state(current_psi, system.dimension);
    }
  }

  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast< duration<double> >(t2 - t1);

  // show_state(current_psi, system.dimension);

  std::cout << "It took me " << time_span.count() << " seconds for " << num_steps << " timsteps." << std::endl;
  for (int l=0; l<(*psis).size(); l++){
    (*expects)[l] = expectations_from_ops(system.obsq, (*psis)[l]);
  }
  std::cout << "Time per step was: " << time_span.count() / num_steps * 1000000 << " micro seconds." << std::endl;
}


template<class system_type>
system_type make_system(json & j, int traj_num){
  // Generic template for making a system.
  system_type system;
  return system;
}


template<>
one_system make_system<one_system>(json& j, int traj_num){
  // Populate system from JSON file based on trajectory number.

  one_system system;

  // psi0
  system.psi0 = json_to_complex_array(j["psi0"]);
  system.dimension = system.psi0.size();

  // Other parameters
  system.duration = j["duration"];
  system.delta_t = j["delta_t"];
  system.sdeint_method = j["sdeint_method"];
  system.downsample = j["downsample"];

  // Loading various operators as diagonals
  load_diag_operator(system.H_eff, j["H_eff"], system.delta_t);
  load_diag_operator_sequence(system.Ls, j["Ls"], sqrt(system.delta_t));
  load_diag_operator_sequence(system.obsq, j["obsq"]);

  // Initialize other objects useful in the simulation.
  system.H_eff_x_psi = std::vector<comp_vec>(system.H_eff.size,
                         std::vector<comp>(system.dimension));

  system.Ls_diags_x_psi = std::vector<std::vector<comp_vec>>(system.Ls.size());
  for (int i=0; i<system.Ls.size(); i++){
    system.Ls_diags_x_psi[i] = std::vector<comp_vec>(system.Ls[i].diags.size(),
                               std::vector<comp>(system.dimension));
  }

  system.Ls_expectations = std::vector<comp>(system.Ls.size());
  return system;
}


template<>
two_system make_system<two_system>(json& j, int traj_num){
  // Populate system from JSON file based on trajectory number.

  two_system system;

  // psi0
  system.psi0 = json_to_complex_array(j["psi0"]);
  system.dimension = system.psi0.size();

  // Other parameters
  system.duration = j["duration"];
  system.delta_t = j["delta_t"];
  system.sdeint_method = j["sdeint_method"];
  system.downsample = j["downsample"];

  system.R = j["R"];
  system.T = j["T"];
  system.eps = j["eps"]; // classical transmission amplification
  system.n = j["n"]; // fictitious noise transmission amplification
  system.lambda = j["lambda"]; // Filtering parameter

  // Loading various operators as diagonals
  load_diag_operator(system.H_eff, j["H_eff"], system.delta_t);
  load_diag_operator(system.L2_dag, j["L2_dag"], sqrt(system.delta_t));
  load_diag_operator(system.L2_dag_L1, j["L2_dag_L1"], system.delta_t);
  load_diag_operator_sequence(system.Ls, j["Ls"], sqrt(system.delta_t));
  load_diag_operator_sequence(system.obsq, j["obsq"]);

  // Initialize other objects useful in the simulation.
  system.H_eff_x_psi = std::vector<comp_vec>(system.H_eff.size,
                         std::vector<comp>(system.dimension));

  system.L2_dag_x_psi = std::vector<comp_vec>(system.L2_dag.size,
                        std::vector<comp>(system.dimension));

  system.L2_dag_L1_x_psi = std::vector<comp_vec>(system.L2_dag_L1.size,
                        std::vector<comp>(system.dimension));

  system.Ls_diags_x_psi = std::vector<std::vector<comp_vec>>(system.Ls.size());
  for (int i=0; i<system.Ls.size(); i++){
    system.Ls_diags_x_psi[i] = std::vector<comp_vec>(system.Ls[i].diags.size(),
                               std::vector<comp>(system.dimension));
  }

  system.Ls_expectations = std::vector<comp>(system.Ls.size());
  return system;
}


template<class system_type>
std::vector<system_type> make_systems(json & j){
  // Make a vector of systems from JSON file.

  int ntraj = j["ntraj"];
  assert(j["seeds"].size() == ntraj);

  std::vector<system_type> systems(ntraj);
  for (int i=0; i<ntraj; i++){
    systems[i] = make_system<system_type>(j, i);
  }
  return systems;
}


template<class system_type>
void qsd(json& j, string output_file_psis, string output_file_expects, int implicit_euler_steps, int steps_for_noise = 10000){
  // Main function for simulator.
  // Populates systems from JSON file and launches individual trajectories.

  std::vector<system_type> systems = make_systems<system_type>(j);

  std::cout << "Generated systems from JSON ... " << std::endl;
  int ntraj = j["ntraj"];

  std::cout << "delta_t: " << systems[0].delta_t << std::endl;
  std::cout << "duration: " << systems[0].duration << std::endl;
  std::cout << "sdeint_method: " << j["sdeint_method"] << std::endl;

  // total steps, including initial state.
  int num_steps = int(systems[0].duration / systems[0].delta_t);
  int num_downsampled_steps = int(num_steps / systems[0].downsample);

  // Generate vectors to store output trajectory psis
  std::vector<std::vector<std::vector<comp>>> psis_lst(ntraj,
    std::vector<std::vector<comp>>(num_downsampled_steps + 1,
      std::vector<comp>(systems[0].dimension))
  );
  // Generate vectors to store output expectation values
  std::vector<std::vector<std::vector<comp>>> expects_lst(ntraj,
    std::vector<std::vector<comp>>(num_downsampled_steps + 1,
      std::vector<comp>(systems[0].obsq.size())));

  //////// Run trajectories

  // Generate threads -- one for each trajectory
  std::vector<std::thread> trajectory_threads;

  for(int i=0; i< ntraj; i++){
    system_type * system_ptr = &systems[i];
    std::vector<std::vector<comp>> * psis_ptr = &psis_lst[i];
    std::vector<std::vector<comp>> * expects_ptr = &expects_lst[i];
    int seed = j["seeds"][i];
    std::cout << "Launching trajectory with seed: " << seed << std::endl;
    // Run the various trajectories with each thread.
    trajectory_threads.push_back(std::thread(run_trajectory<system_type>, system_ptr, seed, steps_for_noise, psis_ptr, expects_ptr, implicit_euler_steps));
  }

  // join all threads
  for(int i=0; i<ntraj; i++){
    trajectory_threads[i].join();
  }

  std::cout << "converting output to JSON ... " << std::endl;
  json j_psis = complex_array_to_json(psis_lst);
  json j_expects = complex_array_to_json(expects_lst);
  string s_psis = j_psis.dump();
  string s_expects = j_expects.dump();
  std::cout << "Total size of downsampled psis data (as string): " << s_psis.length() << std::endl;
  std::cout << "Successfully converted to JSON ... Writing to file next..." << std::endl;
  write_to_file(j_psis, output_file_psis);
  std::cout << "Successfully written psis to file: " << output_file_psis << std::endl;
  write_to_file(j_expects, output_file_expects);
  std::cout << "Successfully written expects to file: " << output_file_expects << std::endl;
}


void qsd_from_json(json & j, string output_file_psis, string output_file_expects, int implicit_euler_steps){
  // Call the appropriate qsd simulator based on parameters found in the json.
  int num_systems = get_num_systems(j);

  if (num_systems == 1){
    std::cout << "running qsd for one system ..." << std::endl;

    // Generate systems -- one for each trajectory
    qsd<one_system>(j, output_file_psis, output_file_expects, implicit_euler_steps);
  }
  else if (num_systems == 2){
    std::cout << "running qsd for two systems ..." << std::endl;

    // Generate systems -- one for each trajectory
    qsd<two_system>(j, output_file_psis, output_file_expects, implicit_euler_steps);
  }
  else{
    std::cout << "qsd is not supported for " << num_systems << " systems... "  << std::endl;
    std::cout << "Exiting ... " << std::endl;
    exit(EXIT_FAILURE);
  }
}


void show_state(comp_vec current_psi, int dimension){
  // Useful function for visualizing the state in the terminal.

  // clear screen magic
  std::cout << "\033[2J\033[1;1H";

  for (int i=0; i<dimension; i++){
    std::cout << i << "  ";
    if (i < 10)
      std::cout << " ";
    int num_stars = int(norm(current_psi[i]) * 100);
    for (int j=0; j<num_stars; j++){
      std::cout << "*";
    }
    std::cout << std::endl;
  }
}


int main (int argc, char* argv[]) {
  string json_file = argv[1];
  string output_file_psis = argv[2];
  string output_file_expects = argv[3];
  int implicit_euler_steps =  2;
  json j;
  read_from_file(json_file, j);
  qsd_from_json(j, output_file_psis, output_file_expects, implicit_euler_steps);
  return 0;
}
