
/*
g++ fast_sim.cpp -o fast_sim -std=c++11
./fast_sim
*/

// basic file operations
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <thread>
#include <math.h>       /* sqrt */
#include <assert.h>     /* assert */
#include <chrono>
#include <complex>      // std::complex
#include <stdlib.h>     /* exit, EXIT_FAILURE */

#ifdef DEBUG
#define DEBUG_MSG(str) do { std::cout << str << std::endl; } while( false )
#else
#define DEBUG_MSG(str) do { } while ( false )
#endif

using json = nlohmann::json;
using namespace std::chrono;

typedef long double num_type;
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

typedef struct
{
  // psi0
  comp_vec psi0;
  int dimension;

  // Other parameters
  num_type duration;
  num_type delta_t;
  string sdeint_method;
  int downsample;
  int ntraj;

  // H_eff
  std::vector<comp_vec> H_eff_diagonals;
  std::vector<int> H_eff_offsets;
  std::vector<comp_vec> H_eff_diags_x_psi;

  // Ls
  std::vector<std::vector<comp_vec>> Ls_diagonals;
  std::vector<std::vector<int>> Ls_offsets;
  std::vector<std::vector<comp_vec>> Ls_diags_x_psi;
  std::vector<comp> Ls_expectations;

  // obsq
  std::vector<std::vector<comp_vec>> obsq_diagonals;
  std::vector<std::vector<int>> obsq_offsets;

} one_system;

void write_to_file(json j, string output_file){
    // Write to file
    // std::ofstream myfile;
    // myfile.open (test_file_loc);
    // myfile << "Writing this to a file.\n";
    // myfile.close();
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

void load_diag_operator(std::vector<comp_vec> & diagonals,
                        std::vector<int> & offsets,
                        json & diag_json,
                        num_type scaling=1.){
  // Loads from JSON with fields "data" and "offsets" representing a sparse matrix.
  json json_data = diag_json["data"];
  for (int i = 0; i < json_data.size(); i++){
    diagonals.push_back(json_to_complex_array(json_data[i], scaling=scaling));
  }
  json json_offsets = diag_json["offsets"];
  for (int i = 0; i < json_offsets.size(); i++){
    offsets.push_back(json_offsets[i]);
  }
}

void load_diag_operator_sequence(std::vector<std::vector<comp_vec>> & diagonals,
                                 std::vector<std::vector<int>> & offsets,
                                 json & diag_json,
                                 num_type scaling=1.){
  // Load each set of diagonals and offsets for each in the sequence
  std::vector<comp_vec> diags_entry;
  std::vector<int> offsets_entry;
  for (int i = 0; i < diag_json.size(); i++){
    load_diag_operator(diags_entry, offsets_entry, diag_json[i], scaling=scaling);
    diagonals.push_back(diags_entry);
    offsets.push_back(offsets_entry);
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

  IMPORTANT: This does not populate the last `offset` components of out for efficiency; they should be zero.
  */
  DEBUG_MSG(std::cout << "offset " << offset << std::endl);
  DEBUG_MSG(std::cout << "diag.size() " << diag.size() << std::endl);

  transform(diag.begin()+offset, diag.end(), vec.begin()+offset, out.begin(), std::multiplies<comp>() );
}

void mult_vecs_offset_lower(comp_vec& diag, comp_vec& vec, comp_vec& out, int& offset){
  /* Multiplies arrays with an offset.

  Should be equivalent to multiplying a sparse matrix with offset lower diagonal.

  IMPORTANT: This does not populate the first `offset` components of out for efficiency; they should be zero.
  */
  DEBUG_MSG(std::cout << "offset " << offset << std::endl);
  DEBUG_MSG(std::cout << "diag.size() " << diag.size() << std::endl);

  transform(diag.begin(), diag.end()-offset, vec.begin(), out.begin()+offset, std::multiplies<comp>() );
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

void update_products_sequence(std::vector<std::vector<int>> & offsets,
                              std::vector<std::vector<comp_vec>> & diags,
                              comp_vec & current_psi,
                              std::vector<std::vector<comp_vec>> & products){
  for (int i=0; i<offsets.size(); i++){
    update_products(offsets[i], diags[i], current_psi, products[i]);
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

  update_products(system.H_eff_offsets, system.H_eff_diagonals, current_psi, system.H_eff_diags_x_psi);
  update_products_sequence(system.Ls_offsets, system.Ls_diagonals, current_psi, system.Ls_diags_x_psi);
  update_Ls_expectations(system.Ls_diags_x_psi, system.Ls_expectations, current_psi);
}

void update_psi(one_system & system, std::vector<comp> & noise, std::vector<comp> & current_psi){

  // Add Hamiltonian component
  for (int i=0; i<system.H_eff_diags_x_psi.size(); i++){
    add_second_to_first(current_psi, system.H_eff_diags_x_psi[i]);
  }

  // Add L components, including noise terms
  comp mult_L_by;
  for (int i=0; i<system.Ls_diags_x_psi.size(); i++){
    DEBUG_MSG(std::cout << "system.Ls_expectations[i]: " << system.Ls_expectations[i] << std::endl);
    mult_L_by = std::conj(system.Ls_expectations[i]) + noise[i];
    DEBUG_MSG(std::cout << "mult_L_by: " << mult_L_by << std::endl);
    for (int j=0; j<system.Ls_diags_x_psi[i].size(); j++){
      // std::cout << norm(noise[i]) / norm(system.Ls_expectations[i]) << std::endl;
      add_second_to_first(current_psi, system.Ls_diags_x_psi[i][j], mult_L_by);
    }
  }
}

void take_euler_step(one_system & system, std::vector<comp> & noise, std::vector<comp> & current_psi){
  update_structures(system, current_psi);
  update_psi(system, noise, current_psi);
  normalize(current_psi);
}

void take_implicit_euler_step(one_system & system, std::vector<comp> & noise, std::vector<comp> & current_psi, int extra_steps = 1){

  std::vector<comp> intermediate_psi (current_psi.size());
  for (int num_step=0; num_step < extra_steps; ++num_step){
    DEBUG_MSG(std::cout << "current_psi before copying to intermediate: " << current_psi[0] << std::endl);
    std::copy(current_psi.begin(), current_psi.end(), intermediate_psi.begin());
    DEBUG_MSG(std::cout << "current_psi after copying to intermediate: " << current_psi[0] << std::endl);
    DEBUG_MSG(std::cout << "intermediate_psi after copying to intermediate: " << intermediate_psi[0] << std::endl);
    update_structures(system, intermediate_psi);
    DEBUG_MSG(std::cout << "intermediate_psi after updating system: " << intermediate_psi[0] << std::endl);
    if (num_step < extra_steps - 1){
      update_psi(system, noise, intermediate_psi);
      DEBUG_MSG(std::cout << "intermediate_psi after updating intermediate_psi (again): " << intermediate_psi[0] << std::endl);
    }
    else{
      update_psi(system, noise, current_psi);
      DEBUG_MSG(std::cout << "current_psi after updating psi: " << current_psi[0] << std::endl);
      DEBUG_MSG(std::cout << "\n" << std::endl);
      if (std::isnan(std::real(current_psi[0]))){
        std::cout << "Problem: psi has NAN component! Exiting." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }
  normalize(current_psi);
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

void run_trajectory(one_system system, int seed, int steps_for_noise, std::vector<std::vector<comp>> * psis){
  // Find dimension...
  system.dimension = system.psi0.size();

  // make generator for random numbers
  std::default_random_engine generator(seed);

  // Number of complex noise terms.
  int num_noise = system.Ls_diagonals.size();

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
      take_implicit_euler_step(system, randoms[j], current_psi, 2);
    else{
      std::cout << "sdeint_method " << system.sdeint_method << " not supported." << std::endl;
      break;
    }
  }

  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast< duration<double> >(t2 - t1);

  // show_state(current_psi, system.dimension);

  std::cout << "It took me " << time_span.count() << " seconds for " << num_steps << " timsteps." << std::endl;
  std::cout << "Time per step was: " << time_span.count() / num_steps * 1000000 << " micro seconds." << std::endl;
}

void qsd_one_system(json & j, string output_file, int steps_for_noise = 10000){
  std::cout << "running qsd for one system ..." << std::endl;

  //////// Define sparse matrices as diagonals and extract from JSON.

  one_system system;

  // Other items to retrieve from JSON

  // psi0
  system.psi0 = json_to_complex_array(j["psi0"]);
  system.dimension = system.psi0.size();

  // Other parameters
  system.duration = j["duration"];
  system.delta_t = j["delta_t"];
  system.sdeint_method = j["sdeint_method"];
  system.downsample = j["downsample"];
  system.ntraj = j["ntraj"];
  assert(j["seeds"].size() == system.ntraj);

  // Loading various operators as diagonals
  // TODO: scale operators by delta_t (and then noise terms too!)
  load_diag_operator(system.H_eff_diagonals, system.H_eff_offsets, j["H_eff"], system.delta_t);
  load_diag_operator_sequence(system.Ls_diagonals, system.Ls_offsets, j["Ls"], sqrt(system.delta_t));
  load_diag_operator_sequence(system.obsq_diagonals, system.obsq_offsets, j["obsq"]);

  // Initialize other objects useful in the simulation.
  system.H_eff_diags_x_psi = std::vector<comp_vec>(system.H_eff_diagonals.size(),
                               std::vector<comp>(system.dimension));

  system.Ls_diags_x_psi = std::vector<std::vector<comp_vec>>(system.Ls_diagonals.size());
  for (int i=0; i<system.Ls_diagonals.size(); i++){
    system.Ls_diags_x_psi[i] = std::vector<comp_vec>(system.Ls_diagonals[i].size(),
                               std::vector<comp>(system.dimension));
  }

  /* View inputs to make sure they are initialized correctly */
  // for (int i=0; i<system.Ls_diags_x_psi.size(); i++){
  //   for (int j=0; j<system.Ls_diags_x_psi[i].size(); j++){
  //     for (int k=0; k<system.Ls_diags_x_psi[i][j].size(); k++){
  //       std::cout << "system.Ls_diags_x_psi[i][j][k] = " << system.Ls_diags_x_psi[i][j][k] << std::endl;
  //     }
  //   }
  // }
  // exit(EXIT_FAILURE);

  system.Ls_expectations = std::vector<comp>(system.Ls_diagonals.size());

  //////// Run trajectories

  // Generate threads -- one for each trajectory
  std::vector<std::thread> trajectory_threads;

  // total steps, including initial state.
  int num_steps = int(system.duration / system.delta_t);
  int num_downsampled_steps = int(num_steps / system.downsample);

  // Generate vectors to store output trajectory psis
  std::vector<std::vector<std::vector<comp>>> psis_lst(system.ntraj,
    std::vector<std::vector<comp>>(num_downsampled_steps + 1,
      std::vector<comp>(system.dimension))
  );

  std::vector<std::vector<std::vector<comp>> * > psis_ptr_lst;
  for(int i=0; i< system.ntraj; i++){
    std::vector<std::vector<comp>> * psis_ptr = &psis_lst[i];
    psis_ptr_lst.push_back(psis_ptr);
  }

  // Run the various trajectories with each thread.
  for(int i=0; i<system.ntraj; i++){
    int seed = j["seeds"][i];
    std::cout << "Launching trajectory with seed: " << seed << std::endl;
    trajectory_threads.push_back(std::thread(run_trajectory, system, seed, steps_for_noise, psis_ptr_lst[i]));
  }

  // join all threads
  for(int i=0; i<system.ntraj; i++){
    trajectory_threads[i].join();
  }

  std::cout << "converting output to JSON ... " << std::endl;
  json j_psis = complex_array_to_json(psis_lst);
  string s = j_psis.dump();
  std::cout << "Total size of downsampled data (as string): " << s.length() << std::endl;
  std::cout << "Successfully converted to JSON ... Writing to file next..." << std::endl;
  write_to_file(j_psis, output_file);
  std::cout << "Successfully written to file: " << output_file << std::endl;

}

void qsd_two_system(json & j, string output_file, int steps_for_noise = 10000){
  std::cout << "running qsd for two system ..." << std::endl;
}

void qsd_from_json(json & j, string output_file){
  // call the appropriate qsd simulator based on parameters found in the json.
  int num_systems = get_num_systems(j);
  if (num_systems == 1)
    qsd_one_system(j, output_file);
  else if (num_systems == 2)
    qsd_two_system(j, output_file);
}

int main () {
  string json_file="/Users/gil/Google Drive/repos/quantum_state_diffusion/num_json_specifications/tmp_file.json";
  string output_file="/Users/gil/Google Drive/repos/quantum_state_diffusion/num_json_specifications/tmp_output.json";
  json j;
  read_from_file(json_file, j);
  qsd_from_json(j, output_file);
  return 0;
}
