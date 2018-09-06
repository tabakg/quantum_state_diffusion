
/*
g++ fast_sim.cpp -o fast_sim -std=c++11
./fast_sim
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

std::vector<string> inputs_one_system = {"H_eff",
                                         "Ls",
                                         "psi0",
                                         "duration",
                                         "delta_t",
                                         "sdeint_method",
                                         "obsq",
                                         "downsample",
                                         "ntraj",
                                         "seeds"};

std::vector<string> inputs_two_systems = {"H1_eff",
                                          "H2_eff",
                                          "L1s","L2s",
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
                                          "n"};

// struct to hold operators, represented by diagonals and offsets.
typedef struct{
  std::vector<comp_vec> diags;
  std::vector<int> offsets;
  int size; // number of diagonals
} diag_op;

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

void write_to_file(json j, string output_file){
    std::ofstream o(output_file);
    o << std::setw(4) << j << std::endl;
}

void read_from_file(string & test_file_loc, json & j){
  // Read from file
  string line;
  std::ifstream myfile (test_file_loc);
  if (myfile.is_open())
  {
    while ( getline (myfile, line) )
    {
      j.merge_patch(json::parse(line));
    }
    myfile.close();
  }
  else std::cout << "Unable to open file";
}

bool matches_inputs(json & j, std::vector<string> inputs){
  // check if all inputs are fields of the json j.
  json entry;
  for(int i=0; i < std::min(inputs.size(), j.size()); i++){
    entry = inputs[i];
    if (j.find(entry) == j.end()) {
      return false;
    }
  }
  return true;
}

int get_num_systems(json & j){
  // figures out the number of systems encoded based on what the inputs are.
  // Assumes that there are no inputs matching BOTH one and two systems.
  if (matches_inputs(j, inputs_one_system))
    return 1;
  else if (matches_inputs(j, inputs_two_systems))
    return 2;
  else return -1;
}

comp_vec json_to_complex_array(json & j_arr, num_type scaling = 1.){
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

json complex_array_to_json(comp_vec vec){
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

json complex_array_to_json(std::vector<comp_vec> vec){
  json j;
  int size = vec.size();
  for(int i=0; i<size; i++){
    j.push_back(complex_array_to_json(vec[i]));
  }
  return j;
}

json complex_array_to_json(std::vector<std::vector<comp_vec>> vec){
  json j;
  int size = vec.size();
  for(int i=0; i<size; i++){
    j.push_back(complex_array_to_json(vec[i]));
  }
  return j;
}

void load_diag_operator(diag_op & op, json & diag_json, num_type scaling=1.){
  // Loads from JSON with fields "data" and "offsets" representing a sparse matrix.
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

void load_diag_operator_sequence(std::vector<diag_op> & Ls,
                                 json & diag_json,
                                 num_type scaling=1.){
  // Load each set of diagonals and offsets for each in the sequence
  int size = diag_json.size();
  Ls = std::vector<diag_op>(size);
  for (int i = 0; i < size; i++){
    load_diag_operator(Ls[i], diag_json[i], scaling=scaling);
  }
}

void get_new_randoms(std::vector<std::vector<comp>> & randoms,
                     int size1,
                     int size2,
                     std::default_random_engine & generator,
                     std::normal_distribution<num_type> & distribution){
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
  // Adds second to first
  int size = in_1.size();
  for (int i=0; i<size; i++){
    in_1[i] += in_2[i] * scalar;
  }
}

void mult_vecs(comp_vec& v1, comp_vec& v2, comp_vec& out){
  transform(v1.begin(), v1.end(), v2.begin(), out.begin(), std::multiplies<comp>() );
}

void mult_vecs_offset_upper(comp_vec& diag, comp_vec& vec, comp_vec& out, int& offset){
  /* Multiplies arrays with an offset.

  Should be equivalent to multiplying a sparse matrix with offset upper diagonal.

  IMPORTANT: This adds values to existing "out" vector; may have to zero them first.
  */
  DEBUG_MSG(std::cout << "offset " << offset << std::endl);
  DEBUG_MSG(std::cout << "diag.size() " << diag.size() << std::endl);

  transform(diag.begin()+offset, diag.end(), vec.begin()+offset, out.begin(), std::multiplies<comp>() );
  // transform(diag.begin(), diag.end()-offset, vec.begin()+offset, out.begin(), std::multiplies<comp>() );
}

void mult_vecs_offset_lower(comp_vec& diag, comp_vec& vec, comp_vec& out, int& offset){
  /* Multiplies arrays with an offset.

  Should be equivalent to multiplying a sparse matrix with offset lower diagonal.

  IMPORTANT: This adds values to existing "out" vector; may have to zero them first.
  */
  DEBUG_MSG(std::cout << "offset " << offset << std::endl);
  DEBUG_MSG(std::cout << "diag.size() " << diag.size() << std::endl);

  transform(diag.begin(), diag.end()-offset, vec.begin(), out.begin()+offset, std::multiplies<comp>() );
  // transform(diag.begin()+offset, diag.end(), vec.begin(), out.begin()+offset, std::multiplies<comp>() );
}

comp dot(comp z1, comp z2){
  // returns z1.conj() * z2
  num_type a, b, c, d;
  a = std::real(z1); b = std::imag(z1);
  c = std::real(z2); d = std::imag(z2);
  return comp(a * c + b * d, a * d - b * c);
}

comp dot_vecs(comp_vec& v1, comp_vec& v2){
  comp val(0., 0.);
  int size = v1.size();
  for (int i=0; i<size; i++){
    val += dot(v1[i], v2[i]);
  }
  return val;
}

comp norm(std::vector<comp> & vec){
  comp val = dot_vecs(vec, vec);
  return sqrt(val);
}

void normalize(std::vector<comp> & vec, comp total){
  int size = vec.size();
  for(int i=0; i<size; i++){
    vec[i] /= total;
  }
}

void normalize(std::vector<comp> & vec){
  comp total = norm(vec);
  normalize(vec, total);
}

void update_products(std::vector<int> & offsets,
                     std::vector<comp_vec> & diags,
                     comp_vec & current_psi,
                     std::vector<comp_vec> & products){
  for (int i=0; i<offsets.size(); i++){
    std::fill(products[i].begin(), products[i].end(), comp(0., 0.));
    int offset = offsets[i];
    if (offset >= 0){
      mult_vecs_offset_upper(diags[i],
                             current_psi,
                             products[i],
                             offset);
    }
    else{
      offset *= -1;
      mult_vecs_offset_lower(diags[i],
                             current_psi,
                             products[i],
                             offset);
    }
  }
}

void update_products(diag_op & op,
                     comp_vec & current_psi,
                     std::vector<comp_vec> & products){
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

// old
void update_products_sequence(std::vector<std::vector<int>> & offsets,
                              std::vector<std::vector<comp_vec>> & diags,
                              comp_vec & current_psi,
                              std::vector<std::vector<comp_vec>> & products){
  for (int i=0; i<offsets.size(); i++){
    update_products(offsets[i], diags[i], current_psi, products[i]);
  }
}

void update_products_sequence(std::vector<diag_op> & ops,
                              comp_vec & current_psi,
                              std::vector<std::vector<comp_vec>> & products){
  for (int i=0; i<ops.size(); i++){
    update_products(ops[i], current_psi, products[i]);
  }
}

comp expectation_from_vecs(std::vector<comp_vec> & product,
                            comp_vec & current_psi){
  comp expectation(0., 0.);
  for (int i=0; i<product.size(); i++){
    expectation += dot_vecs(current_psi, product[i]);
    DEBUG_MSG(std::cout << "current expectation: " << expectation << std::endl);
  }
  return expectation;
}

std::vector<comp> expectations_from_diags(std::vector<diag_op> ops, comp_vec psi){
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

void update_Ls_expectations(std::vector<std::vector<comp_vec>> & products,
                            std::vector<comp> & expectations,
                            comp_vec & current_psi){
  int size = products.size();
  for (int i=0; i<size; i++){
    DEBUG_MSG(std::cout << "old expectations[i]: " << expectations[i] << std::endl);
    expectations[i] = expectation_from_vecs(products[i], current_psi);
    DEBUG_MSG(std::cout << "new expectations[i]: " << expectations[i] << std::endl);
  }
}

void update_structures(one_system & system, std::vector<comp> & current_psi){
  // TODO: separate data for positive and negative offsets

  update_products(system.H_eff, current_psi, system.H_eff_x_psi);
  update_products_sequence(system.Ls, current_psi, system.Ls_diags_x_psi);
  update_Ls_expectations(system.Ls_diags_x_psi, system.Ls_expectations, current_psi);
}

void update_psi(one_system & system, std::vector<comp> & noise, std::vector<comp> & current_psi){

  // Add Hamiltonian component
  for (int i=0; i<system.H_eff.size; i++){
    add_second_to_first(current_psi, system.H_eff_x_psi[i]);
  }

  // Add L components, including noise terms
  comp mult_L_by;
  for (int i=0; i<system.Ls_diags_x_psi.size(); i++){
    mult_L_by = std::conj(system.Ls_expectations[i]) + noise[i];
    for (int j=0; j<system.Ls_diags_x_psi[i].size(); j++){
      add_second_to_first(current_psi, system.Ls_diags_x_psi[i][j], mult_L_by);
    }
  }
}

void update_and_normalize_psi(one_system & system, std::vector<comp> & noise, std::vector<comp> & current_psi){
  update_psi(system, noise, current_psi);
  normalize(current_psi);
}

void take_euler_step(one_system & system, std::vector<comp> & noise, std::vector<comp> & current_psi){
  update_structures(system, current_psi);
  update_and_normalize_psi(system, noise, current_psi);
}

void take_implicit_euler_step(one_system & system, std::vector<comp> & noise, std::vector<comp> & current_psi, int steps){
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
    update_structures(system, next_psi);
    std::copy(current_psi.begin(), current_psi.end(), next_psi.begin());
    update_and_normalize_psi(system, noise, next_psi);
  }

  // Last step: update system structures and current_psi using next_psi
  update_structures(system, next_psi);
  update_and_normalize_psi(system, noise, current_psi);
}


void show_state(comp_vec current_psi, int dimension){

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

void run_trajectory(one_system * system_ptr, int seed, int steps_for_noise,
                    std::vector<std::vector<comp>> * psis,
                    std::vector<std::vector<comp>> * expects,
                    int implicit_euler_steps=3){

  one_system system = *system_ptr;

  // Find dimension...
  system.dimension = system.psi0.size();

  // make generator for random numbers
  std::default_random_engine generator(seed);

  // Number of complex noise terms.
  int num_noise = system.Ls.size();

  // total steps, including initial state.
  int num_steps = int(system.duration / system.delta_t);
  int num_downsampled_steps = int(num_steps / system.downsample);

  // accepts num_type, double, or long double
  num_type mean = 0;
  num_type std_dev = sqrt(0.5);
  std::normal_distribution<num_type> distribution(0., std_dev);

  // random variables stored here. They are replenished every steps_for_noise steps.
  std::vector<std::vector<comp>> randoms(steps_for_noise, std::vector<comp>(num_noise));

  std::vector<comp> current_psi = system.psi0;

  //////// Main for loop for simulation
  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  for(int i=0, j=0, k=0, l=0; i<num_steps; i++, j++, k++){
    // Get new noise terms as we go
    if (j == steps_for_noise)
      j = 0;
    if (k == system.downsample)
      k = 0;
    if (j == 0)
      get_new_randoms(randoms, steps_for_noise, num_noise, generator, distribution);
    if (k == 0){
      std::copy(current_psi.begin(), current_psi.end(), (*psis)[l].begin());
      l++; // number of psis recorded
      // show_state(current_psi, system.dimension);
    }
    if (system.sdeint_method == "ItoEuler")
      take_euler_step(system, randoms[j], current_psi);
    else if (system.sdeint_method == "itoImplicitEuler")
      take_implicit_euler_step(system, randoms[j], current_psi, implicit_euler_steps);
    else{
      std::cout << "sdeint_method " << system.sdeint_method << " not supported." << std::endl;
      break;
    }
  }

  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast< duration<double> >(t2 - t1);

  // show_state(current_psi, system.dimension);

  std::cout << "It took me " << time_span.count() << " seconds for " << num_steps << " timsteps." << std::endl;
  for (int l=0; l<(*psis).size(); l++){
    (*expects)[l] = expectations_from_diags(system.obsq, (*psis)[l]);
  }
  std::cout << "Time per step was: " << time_span.count() / num_steps * 1000000 << " micro seconds." << std::endl;
}

one_system make_system_one(json & j, int traj_num){

  one_system system;
  //////// Define sparse matrices as diagonals and extract from JSON.

  // Other items to retrieve from JSON

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

std::vector<one_system> make_systems_one(json & j){

  int ntraj = j["ntraj"];
  assert(j["seeds"].size() == ntraj);

  std::vector<one_system> systems(ntraj);
  for (int i=0; i<ntraj; i++){
    systems[i] = make_system_one(j, i);
  }
  return systems;
}

void qsd_one_system(json & j, string output_file_psis, string output_file_expects, int implicit_euler_steps, int steps_for_noise = 10000){
  std::cout << "running qsd for one system ..." << std::endl;

  //////// Run trajectories

  // Generate systems -- one for each trajectory
  std::vector<one_system> systems = make_systems_one(j);

  std::cout << "Generated systems from JSON ... " << std::endl;
  int ntraj = j["ntraj"];

  // Generate threads -- one for each trajectory
  std::vector<std::thread> trajectory_threads;

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

  for(int i=0; i< ntraj; i++){
    one_system * system_ptr = &systems[i];
    std::vector<std::vector<comp>> * psis_ptr = &psis_lst[i];
    std::vector<std::vector<comp>> * expects_ptr = &expects_lst[i];
    int seed = j["seeds"][i];
    std::cout << "Launching trajectory with seed: " << seed << std::endl;
    // Run the various trajectories with each thread.
    trajectory_threads.push_back(std::thread(run_trajectory, system_ptr, seed, steps_for_noise, psis_ptr, expects_ptr, implicit_euler_steps));
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

void qsd_two_system(json & j, string output_file_psis, string output_file_expects, int implicit_euler_steps, int steps_for_noise = 10000){
  std::cout << "running qsd for two system ..." << std::endl;
}

void qsd_from_json(json & j, string output_file_psis, string output_file_expects, int implicit_euler_steps){
  // call the appropriate qsd simulator based on parameters found in the json.
  int num_systems = get_num_systems(j);
  if (num_systems == 1)
    qsd_one_system(j, output_file_psis, output_file_expects, implicit_euler_steps);
  else if (num_systems == 2)
    qsd_two_system(j, output_file_psis,  output_file_expects, implicit_euler_steps);
}

int main () {
  string json_file="/Users/gil/Google Drive/repos/quantum_state_diffusion/num_json_specifications/tmp_file.json";
  string output_file_psis="/Users/gil/Google Drive/repos/quantum_state_diffusion/num_json_specifications/tmp_output.json";
  string output_file_expects="/Users/gil/Google Drive/repos/quantum_state_diffusion/num_json_specifications/tmp_output_expects.json";
  int implicit_euler_steps =  2;
  json j;
  read_from_file(json_file, j);
  qsd_from_json(j, output_file_psis, output_file_expects, implicit_euler_steps);
  return 0;
}
