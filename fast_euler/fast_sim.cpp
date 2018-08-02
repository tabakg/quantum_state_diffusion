
/*
g++ open_file.cpp -o open_file -std=c++11
./open_file
*/

// basic file operations
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <eigen3/Eigen/Sparse>
#include <random>
#include <thread>
#include <math.h>       /* sqrt */


// for convenience
using json = nlohmann::json;
using namespace Eigen;

typedef std::string string;
typedef std::complex<float> comp;
const std::complex<float> imag_unit(0.0,1.0);
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
  // H_eff
  std::vector<comp_vec> H_eff_diagonals;
  std::vector<comp_vec> H_eff_diags_x_psi;
  std::vector<int> H_eff_offsets;

  // Ls
  std::vector<std::vector<comp_vec>> Ls_diagonals;
  std::vector<std::vector<int>> Ls_offsets;

  // obsq
  std::vector<std::vector<comp_vec>> obsq_diagonals;
  std::vector<std::vector<int>> obsq_offsets;

  // psi0
  comp_vec psi0;

  // Other parameters
  float duration;
  float delta_t;
  string sdeint_method;
  int downsample;
  int ntraj;

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

comp_vec json_to_complex_array(json & j_arr, float scaling=1.){
  // Assumes a json serialization with fields "real" and "imag".
  // These should be arrays of the same size containing floats.
  // Output is a std::vector with complex<float> entries.
  comp_vec arr;
  int size = j_arr["real"].size();
  arr.reserve(size);
  float real_part;
  float imag_part;
  for(int n=0; n < size; n++){
    real_part = j_arr["real"][n];
    imag_part = j_arr["imag"][n];
    arr.push_back(real_part * scaling + imag_unit * imag_part * scaling);
  }
  return arr;
}

void load_diag_operator(std::vector<comp_vec> & diagonals,
                        std::vector<int> & offsets,
                        json & diag_json){
  // Loads from JSON with fields "data" and "offsets" representing a sparse matrix.
  json json_data = diag_json["data"];
  for (int i = 0; i < json_data.size(); i++){
    diagonals.push_back(json_to_complex_array(json_data[i]));
  }
  json json_offsets = diag_json["offsets"];
  for (int i = 0; i < json_offsets.size(); i++){
    offsets.push_back(json_offsets[i]);
  }
}

void load_diag_operator_sequence(std::vector<std::vector<comp_vec>> & diagonals,
                        std::vector<std::vector<int>> & offsets,
                        json & diag_json){
  // Load each set of diagonals and offsets for each in the sequence
  std::vector<comp_vec> diags_entry;
  std::vector<int> offsets_entry;
  for (int i = 0; i < diag_json.size(); i++){
    load_diag_operator(diags_entry, offsets_entry, diag_json[i]);
    diagonals.push_back(diags_entry);
    offsets.push_back(offsets_entry);
  }
}

void get_new_randoms(std::vector<std::vector<comp>> & randoms,
                     int size1,
                     int size2,
                     std::default_random_engine & generator,
                     std::normal_distribution<float> & distribution){
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

void normalize (std::vector<comp> & vec){
  float total = 0;
  int size = vec.size();
  for(int i=0; i<size; i++){
    total += std::norm(vec[i]);
  }
  for(int i=0; i<size; i++){
    vec[i] /= total;
  }
}


void take_step(one_system & system, std::vector<comp> & noise, std::vector<comp> & current_psi){

  // Update structures
  // TODO: separate data for positive and negative offsets
  for (int i=0; i<system.H_eff_offsets.size(); i++){
    int offset = system.H_eff_offsets[i];
    if (offset >= 0){
      mult_vecs_offset_upper(system.H_eff_diagonals[i],
                             current_psi,
                             system.H_eff_diags_x_psi[i],
                             offset);
    }
    else{
      offset *= -1;
      mult_vecs_offset_lower(system.H_eff_diagonals[i],
                                current_psi,
                                system.H_eff_diags_x_psi[i],
                                offset);
    }
  }

  // update psi
  for (int i=0; i<system.H_eff_diags_x_psi.size(); i++){
    add_second_to_first(current_psi, system.H_eff_diags_x_psi[i]);
  }
  // normalize psi
  normalize(current_psi);
  //TODO: monitor size of psi (when to normalize)
}

void run_trajectory(one_system system, int seed, int steps_for_noise){
  // Find dimension...
  int dim = system.psi0.size();

  // make generator for random numbers
  std::default_random_engine generator(seed);

  // Number of complex noise terms.
  int num_noise = system.Ls_diagonals.size();

  // total steps, including initial state.
  int num_steps = int(system.duration / system.delta_t);
  int num_downsampled_steps = int(num_steps / system.downsample);

  // accepts float, double, or long double
  std::normal_distribution<float> distribution(0., sqrt(system.delta_t/2.));

  // random variables stored here. They are replenished every steps_for_noise steps.
  std::vector<std::vector<comp>> randoms(steps_for_noise, std::vector<comp>(num_noise));

  // Initialize downsampled states.
  std::vector<std::vector<comp>> psis(num_downsampled_steps, std::vector<comp>(dim));

  std::vector<comp> current_psi = system.psi0;

  //////// Main for loop for simulation
  for(int i=0, j=steps_for_noise, k=system.downsample, l=0; i<num_steps; i++, j--, k--){
    // Get new noise terms as we go
    if (j == steps_for_noise){
      get_new_randoms(randoms, steps_for_noise, num_noise, generator, distribution);
    }
    else if (j == 0){
      j = steps_for_noise;
    }
    if (k == system.downsample){
      psis[l] = current_psi;
      l++; // number of psis recorded
    }
    else if (k == 0){
      k = system.downsample;
    }
    take_step(system, randoms[j], current_psi);
  }

  // Print state after simulation.
  for (int i=0; i<50; i++){
    std::cout << current_psi[i] << std::endl;
  }

}

void qsd_one_system(json & j, int steps_for_noise = 10000){
  std::cout << "running qsd for one system ..." << std::endl;

  //////// Define sparse matrices as diagonals and extract from JSON.

  one_system system;

  // Loading various operators as diagonals
  // TODO: scale operators by delta_t (and then noise terms too!)
  load_diag_operator(system.H_eff_diagonals, system.H_eff_offsets, j["H_eff"]);
  load_diag_operator_sequence(system.Ls_diagonals, system.Ls_offsets, j["Ls"]);
  load_diag_operator_sequence(system.obsq_diagonals, system.obsq_offsets, j["obsq"]);

  // Other items to retrieve from JSON

  // psi0
  system.psi0 = json_to_complex_array(j["psi0"]);

  // Other parameters
  system.duration = j["duration"];
  system.delta_t = j["delta_t"];
  system.sdeint_method = j["sdeint_method"];
  system.downsample = j["downsample"];
  system.ntraj = j["ntraj"];

  // Initialize other objects useful in the simulation.
  system.H_eff_diags_x_psi = std::vector<std::vector<comp>>(system.H_eff_diagonals.size(),
                             std::vector<comp>(system.psi0.size()));

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
