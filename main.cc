#include <dmlc/json.h>
#include <graph_runtime.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>
// reading a text file
#include <dmlc/io.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <filesystem>
#include <evtimer.h>
//ADD by River 2020.03.06 for input image

#include <string>
#include <common.h>

//End ADD by River 2020.03.06 for input image
using namespace std;
using namespace tvm::runtime;

int main(int argc, char** argv)
{ 
  std::string parameter_path="../tvm_generated_files/binary_params.bin";
  std::ifstream params(parameter_path, std::ios::binary);
  params.seekg(0, ios::end);
  int weight_length = params.tellg();
  params.seekg(0, ios::beg);
  char * buffer;
  buffer = new char[weight_length];
  for (int i =0; i < weight_length; i++)
    buffer[i] = '\0';
  params.read(buffer, weight_length);
  params.close();
  TVMByteArray params_arr;
  params_arr.data = buffer;
  params_arr.size = weight_length;

  std::string input_path = "../tvm_generated_files/input.bin";
  std::ifstream input_data(input_path, std::ios::binary);
  input_data.seekg(0, ios::end);
  int input_length = input_data.tellg();
  input_data.seekg(0, ios::beg);
  std::cout << "Input Length:" << input_length << "\n";
  float *input_buffer;
  input_buffer = new float[input_length/4];
  input_data.read((char *)input_buffer,input_length);
  input_data.close();


  std::string graph_path = "../tvm_generated_files/graph.json";
  std::ifstream graph_json(graph_path, std::ios::binary);
  graph_json.seekg(0, ios::end);
  int graph_length = graph_json.tellg();
  graph_json.seekg(0, ios::beg);
  char *graph;
  graph = new char[graph_length];
  graph_json.read(graph,graph_length);
  graph_json.close();
  
  tvm::runtime::Module mod_hostc_wrapper = (*tvm::runtime::Registry::Get("runtime.module.hostc_static"))("hostcmodule");
  tvm::runtime::Module mod_opencl = (*tvm::runtime::Registry::Get("runtime.module.loadfile_cl"))("../tvm_generated_files/device_code.cl");
  mod_hostc_wrapper.Import(mod_opencl);
  auto pGraphRuntime = tvm::runtime::Registry::Get("tvm.graph_runtime.create");
  tvm::runtime::Module run_mod = (*pGraphRuntime)(graph, mod_hostc_wrapper, 1, 0);
  auto set_input = run_mod.GetFunction("set_input", false);
  auto load_params = run_mod.GetFunction("load_params", false);
  auto run = run_mod.GetFunction("run", false);
  auto get_input_tensor = run_mod.GetFunction("get_input", false);
  auto get_num_output = run_mod.GetFunction("get_num_outputs");
  auto get_output_tensor = run_mod.GetFunction("get_output");
  auto get_debug_node = run_mod.GetFunction("debug_node");
  DLTensor *input_tensor= get_input_tensor(0);
  DLTensor *input = set_input_tensor(input_tensor,input_buffer);

  set_input(0,input);
  load_params(params_arr);
  run();
  int num_outputs = get_num_output();
  for(int i=0;i<num_outputs;i++)
  {
    std::cout << "----------------\n";
    printf("output%d:\n",i+1);
    DLTensor *output_tensor = get_output_tensor(i);
    print_node(output_tensor);
  }
  return 0;

}
