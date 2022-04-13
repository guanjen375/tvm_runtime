/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file opencl_module.cc
 */
#include <dmlc/memory_io.h>
#include <tvm/runtime/registry.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <ev_opencl_common.h>
#include <kdisp.h>
#include <evthreads.h>
#define cpu 1 // EV61
extern std::unordered_map<std::string, kernelfunc *> funcmap;
namespace tvm {
namespace runtime {

class EV_OpenCLWrappedFunc {
 public:
  // initialize the OpenCL function.
  void Init(std::string func_name,std::vector<size_t> arg_size,const std::vector<std::string>& thread_axis_tags)  
  {
    func_name_ = func_name;
    arg_size_ = arg_size;
    thread_axis_cfg_.Init(arg_size.size(), thread_axis_tags);  
  }
  // invoke the function with void arguments
 void operator()(TVMArgs args,
                  TVMRetValue* rv,
                  void** void_args) const {
    ThreadWorkLoad wl = thread_axis_cfg_.Extract(args);
    int num_n=wl.work_size[0];
    if(wl.work_size[3]!=64)
	    std::cout << "Local Size Error\n";
    int num_cpu=cpu; // number of cpus
    evInitKernelDispatcher();
    int idx=0;
    unsigned int num_args = arg_size_.size() + 2 ; // one for work_group size, one for number of cpus
    int arg_kind[num_args];
    int grid[]= {(int)num_cpu};
    int threads[]={1};
    for(int i=0;i<num_args;i++)
    {
      if(args.type_codes[i]==0)
        arg_kind[i]=2; // Argument is 32-bit integer
      else
        arg_kind[i]=0; // Argument is global pointer
    }
    arg_kind[num_args-2]=2;
    arg_kind[num_args-1]=2;
    auto it = funcmap.find(func_name_);
    if(it==funcmap.end())
        std::cout << "Function not found\n";
    kernel_specification kernel_info={func_name_.c_str(),num_args,it->second,arg_kind,(64,1,1)};
    kernelhndl ev_kernel;
    ev_kernel=evCreateKernel(&kernel_info);
    evSetKernelArg(ev_kernel,num_args-2,&num_n);
    evSetKernelArg(ev_kernel,num_args-1,&num_cpu);
    for(int i=0;i<num_args-2;i++)
    {
        idx=args.type_codes[i];
        if(idx==0)
            evSetKernelArg(ev_kernel,i,(void *)&(args.values[i].v_int64)); //Argument is v_int64
        else
            evSetKernelArg(ev_kernel,i,(void *)args.values[i].v_handle); // Argument is v_handle
    }
    evEnqueueNDRangeKernel(ev_kernel,1,NULL, grid, threads);
    evReleaseKernel(ev_kernel);
  }

 private:
  // The name of the function.
  std::string func_name_;
  // convert code for void argument
  std::vector<size_t> arg_size_;
  // thread axis config
  ThreadAxisConfig thread_axis_cfg_;
};

PackedFunc EV_OpenCLModuleNode::GetFunction(
    const std::string& name,
    const ObjectPtr<Object>& sptr_to_self) {
  auto it = fmap_.find(name);
  const FunctionInfo& info = it->second;
  EV_OpenCLWrappedFunc f;
  std::vector<size_t> arg_size(info.arg_types.size());
  for (size_t i = 0; i < info.arg_types.size(); ++i) {
    DLDataType t = info.arg_types[i];
    CHECK_EQ(t.lanes, 1U);
    if (t.code == kTVMOpaqueHandle) {
      // specially store pointer type size in OpenCL driver
      arg_size[i] = sizeof(void*);
    } else {
      uint32_t bits = t.bits;
      CHECK_EQ(bits % 8, 0U);
      arg_size[i] = bits / 8;
    }
  }
  // initialize the wrapped func.
  f.Init(name, arg_size, info.thread_axis_tags);
  return PackFuncVoidAddr(f, info.arg_types);
}

Module EV_OpenCLModuleLoad(std::string name)
{
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string meta_file = GetMetaFilePath(name);
  LoadMetaDataFromFile(meta_file, &fmap);
  auto n = make_object<EV_OpenCLModuleNode>(name,fmap);
  return Module(n);
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_cl")
.set_body_typed(EV_OpenCLModuleLoad);


}  // namespace runtime
}  // namespace tvm
