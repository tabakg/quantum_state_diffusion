
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

using json = nlohmann::json;
using namespace std::chrono;

typedef float num_type;
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

// TODO: Write results to file...
// void write_to_file(string test_file_loc){
//     // Write to file
//     std::ofstream myfile;
//     myfile.open (test_file_loc);
//     myfile << "Writing this to a file.\n";
//     myfile.close();
// }

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
  transform(diag.begin(), diag.end(), vec.begin()+offset, out.begin(), std::multiplies<comp>() );
}

void mult_vecs_offset_lower(comp_vec& diag, comp_vec& vec, comp_vec& out, int& offset){
  /* Multiplies arrays with an offset.

  Should be equivalent to multiplying a sparse matrix with offset lower diagonal.

  IMPORTANT: This does not populate the first `offset` components of out for efficiency; they should be zero.
  */
  transform(diag.begin(), diag.end(), vec.begin(), out.begin()+offset, std::multiplies<comp>() );
}

comp dot(comp z1, comp z2){
  // returns z1.conj() * z2
  num_type a, b, c, d;
  a = real(z1); b = imag(z1);
  c = real(z2); d = imag(z2);
  return comp(a * c + b * d, a * d - b * c);
}

void dot_vecs(comp_vec& v1, comp_vec& v2, comp & val){
  for (int i=0; i<v1.size(); i++){
    val += dot(v1[i], v2[i]);
  }
}

comp norm(std::vector<comp> & vec){
  comp val = 0.;
  dot_vecs(vec, vec, val);
  return sqrt(val);
}

void normalize(std::vector<comp> & vec){
  comp total = norm(vec);
  for(int i=0; i<vec.size(); i++){
    vec[i] /= total;
  }
}

void normalize(std::vector<comp> & vec, comp total){
  for(int i=0; i<vec.size(); i++){
    vec[i] /= total;
  }
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
  comp expectation = 0.;
  for (int i=0; i<product.size(); i++){
    dot_vecs(current_psi, product[i], expectation);
  }
  return expectation;
}

void update_Ls_expectations(comp_vec & current_psi,
                            std::vector<std::vector<comp_vec>> & products,
                            std::vector<comp> & expectations){
  for (int i=0; i<products.size(); i++){
    expectations[i] = expectation_from_vecs(products[i], current_psi);
  }
}

void update_structures(one_system & system, std::vector<comp> & current_psi){
  // TODO: separate data for positive and negative offsets
  update_products(system.H_eff_offsets, system.H_eff_diagonals, current_psi, system.H_eff_diags_x_psi);
  update_products_sequence(system.Ls_offsets, system.Ls_diagonals, current_psi, system.Ls_diags_x_psi);
  update_Ls_expectations(current_psi, system.Ls_diags_x_psi, system.Ls_expectations);
}

void update_psi(one_system & system, std::vector<comp> & noise, std::vector<comp> & current_psi){

  // Add Hamiltonian component
  for (int i=0; i<system.H_eff_diags_x_psi.size(); i++){
    add_second_to_first(current_psi, system.H_eff_diags_x_psi[i]);
  }

  // Add L components, including noise terms
  for (int i=0; i<system.Ls_diags_x_psi.size(); i++){
    comp mult_L_by = std::conj(system.Ls_expectations[i]) + noise[i];
    for (int j=0; j<system.Ls_diags_x_psi[i].size(); j++){
      // std::cout << norm(noise[i]) / norm(system.Ls_expectations[i]) << std::endl;
      add_second_to_first(current_psi, system.Ls_diags_x_psi[i][j], mult_L_by);
    }
  }
}

void take_euler_step(one_system & system, std::vector<comp> & noise, std::vector<comp> & current_psi){
  update_structures(system, current_psi);
  update_psi(system, noise, current_psi);
  //TODO: monitor size of psi (when to normalize)
  normalize(current_psi);
}

void take_implicit_euler_step(one_system & system, std::vector<comp> & noise, std::vector<comp> & current_psi){

  std::vector<comp> intermediate_psi (current_psi.size());
  std::copy(current_psi.begin(), current_psi.end(), intermediate_psi.begin());

  // initial estimate for various structures using current_psi
  update_structures(system, current_psi);
  // update intermediate_psi using initial structure estimates
  update_psi(system, noise, intermediate_psi);
  // use intermediate_psi to update structures (estimate at next step)
  update_structures(system, intermediate_psi);
  // use the estimate from the next step to update current_psi
  update_psi(system, noise, current_psi);

  //TODO: monitor size of psi (when to normalize)
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

void run_trajectory(one_system system, int seed, int steps_for_noise){
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

  // Initialize downsampled states.
  std::vector<std::vector<comp>> psis(num_downsampled_steps, std::vector<comp>(system.dimension));

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
      std::copy(current_psi.begin(), current_psi.end(), psis[l].begin());
      l++; // number of psis recorded
      // show_state(current_psi, system.dimension);
    }
    take_implicit_euler_step(system, randoms[j], current_psi);
  }

  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast< duration<double> >(t2 - t1);

  // show_state(current_psi, system.dimension);

  std::cout << "It took me " << time_span.count() << " seconds." << std::endl;
  std::cout << "Time per step was: " << time_span.count() / num_steps * 1000000 << " micro seconds." << std::endl;
}

void qsd_one_system(json & j, int steps_for_noise = 10000){
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
  system.Ls_expectations = std::vector<comp>(system.Ls_diagonals.size());

  //////// Run trajectories

  // Generate threads -- one for each trajectory
  std::vector<std::thread> trajectory_threads;

  // Run the various trajectories with each thread.
  for(int i=0; i<system.ntraj; i++){
    int seed = j["seeds"][i];
    std::cout << "Launching trajectory with seed: " << seed << std::endl;
    trajectory_threads.push_back(std::thread(run_trajectory, system, seed, steps_for_noise));
  }

  // join all threads
  for(int i=0; i<system.ntraj; i++){
    trajectory_threads[i].join();
  }
}

void qsd_two_system(json & j){
  std::cout << "running qsd for two system ..." << std::endl;
}

void qsd_from_json(json & j){
  // call the appropriate qsd simulator based on parameters found in the json.
  int num_systems = get_num_systems(j);
  if (num_systems == 1)
    qsd_one_system(j);
  else if (num_systems == 2)
    qsd_two_system(j);
}

int main () {
  string json_file="/Users/gil/Google Drive/repos/quantum_state_diffusion/num_json_specifications/tmp_file.json";
  json j;
  read_from_file(json_file, j);
  qsd_from_json(j);
  return 0;
}
