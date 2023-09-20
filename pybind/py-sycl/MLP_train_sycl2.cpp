
// #include <sycl/sycl.hpp>
#include <CL/sycl.hpp>
#include <iostream>
#include <limits>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace std;
using namespace sycl;

// MLP metadata.
constexpr int BS = 512;
constexpr int L1 = 8;
constexpr int L2 = 128;
constexpr int L3 = 128;
constexpr int L4 = 4;


constexpr float GAMMA = 0.1;
constexpr float LEARNRATE = 1.0;

// int VerifyResult(float (*c_back)[P]);
// struct params {
//     std::vector<std::vector<float>> hiddenWeights;
//     std::vector<std::vector<float>> hiddenBiases;
//     std::vector<std::vector<float>> outputWeights;
//     std::vector<std::vector<float>> outputBiases;
// };


// data packs output by the train_itr kernel. They are only used to realize
// pass-by-reference functionalities when using pybind.
struct params_out {
    params_out() {
        hiddenWeights1_d.resize(L1);
        for (size_t i=0; i<L1; i++)hiddenWeights1_d[i].resize(L2);
        hiddenBiases1_d.resize(L2);
        hiddenWeights2_d.resize(L2);
        for (size_t i=0; i<L2; i++)hiddenWeights2_d[i].resize(L3); 
        hiddenBiases2_d.resize(L3);       
        
        outputWeights_d.resize(L3);
        for (size_t i=0; i<L3; i++)outputWeights_d[i].resize(L4);
        outputBiases_d.resize(4);  
    }
    std::vector<std::vector<float>> hiddenWeights1_d;
    std::vector<float> hiddenBiases1_d;
    std::vector<std::vector<float>> hiddenWeights2_d;
    std::vector<float> hiddenBiases2_d;
    std::vector<std::vector<float>> outputWeights_d;
    std::vector<float> outputBiases_d;
    void print_params(){
      cout<<"--- Printing paramemeters ---\n";
      cout<<"hiddenBiases1: ";
      for (size_t i=0; i<L2; i++){
        cout << hiddenBiases1_d[i] <<" ";
      }
      cout<<"hiddenBiases2: ";
      for (size_t i=0; i<L3; i++){
        cout << hiddenBiases2_d[i] <<" ";
      }
      cout<<"\noutputBiases: ";
      for (size_t i=0; i<L4; i++){
        cout << outputBiases_d[i] <<" ";
      }
      cout<<"\nhiddenWeights1: \n";
      for (size_t i=0; i<L1; i++){
        for (size_t j=0; j<L2; j++){
          cout << hiddenWeights1_d[i][j] <<" ";
        }
        cout << "\n";
      }
      cout<<"\nhiddenWeights2: \n";
      for (size_t i=0; i<L2; i++){
        for (size_t j=0; j<L3; j++){
          cout << hiddenWeights2_d[i][j] <<" ";
        }
        cout << "\n";
      }
      cout<<"\noutputWeights: \n";
      for (size_t i=0; i<L3; i++){
        for (size_t j=0; j<L4; j++){
          cout << outputWeights_d[i][j] <<" ";
        }
        cout << "\n";
      }
      cout<<"--- End ---\n";
    }
};


class DQNTrainer {     
  public:  
    // --- weights and biases ---
    float(*hiddenWeights1)[L2];
    float * hiddenBiases1;
    float(*hiddenWeights2)[L3];
    float * hiddenBiases2;
    float(*outputWeights)[L4];
    float * outputBiases;
    float(*hiddenWeightsTarget1)[L2];
    float * hiddenBiasesTarget1;
    float(*hiddenWeightsTarget2)[L3];
    float * hiddenBiasesTarget2;
    float(*outputWeightsTarget)[L4];
    float * outputBiasesTarget;
    // --- end weights and biases ---
    
    // --- kernel queue ---
    queue q;
    // --- end kernel queue ---

    // --- intermiediate results used in each training itr ---
    float(*in_arr)[L1];
    float(*out1_arr)[L2];
    float(*out2_arr)[L3];
    float(*outQ_arr)[L4];

    float(*in_arr_n)[L1];
    float(*out1_arr_snt)[L2];
    float(*out2_arr_snt)[L3];
    float(*outQtarg_arr)[L4];

    float *outputBiasGradients; 
    float(*error_signals)[L4];
    float(*outputWeightsGradients)[L4];
    float(*hidden_error_signals2)[L3];
    float *hiddenBiasGradients2; 
    float(*hiddenWeightsGradients2)[L3];
    float(*hidden_error_signals1)[L2];
    float *hiddenBiasGradients1; 
    float(*hiddenWeightsGradients1)[L2];
    // float targ_max_Q=new float[BS];
    // --- end intermiediate results used in each training itr ---

    // Args of the constructor will be passed from external python code (pytorch xavier init)
    DQNTrainer(std::vector<std::vector<float>> hw1, std::vector<float> hb1, std::vector<std::vector<float>> hw2, std::vector<float> hb2,
    std::vector<std::vector<float>> ow, std::vector<float> ob) {  
      // WB.hiddenWeights = hw;
      hiddenWeights1 = new float[L1][L2];
      hiddenWeightsTarget1 = new float[L1][L2];
      for (size_t i = 0; i < L1; i++) {
        for (size_t j = 0; j < L2; j++){
          hiddenWeights1[i][j] = hw1[i][j];
        }  
      }
      // WB.hiddenBiases = hb;
      hiddenBiases1 = new float[L2];
      hiddenBiasesTarget1 = new float[L2];
      for (size_t i = 0; i < L2; i++) hiddenBiases1[i]=hb1[i];

      hiddenWeights2 = new float[L2][L3];
      hiddenWeightsTarget2 = new float[L2][L3];
      for (size_t i = 0; i < L2; i++) {
        for (size_t j = 0; j < L3; j++){
          hiddenWeights2[i][j] = hw2[i][j];
        }  
      }
      // WB.hiddenBiases = hb;
      hiddenBiases1 = new float[L3];
      hiddenBiasesTarget1 = new float[L3];
      for (size_t i = 0; i < L3; i++) hiddenBiases1[i]=hb2[i];

      // WB.outputWeights = ow;
      outputWeights = new float[L3][L4];
      outputWeightsTarget = new float[L3][L4];
      for (size_t i = 0; i < L3; i++) {
        for (size_t j = 0; j < L4; j++){
          outputWeights[i][j] = ow[i][j];
        }  
      }
      // WB.outputBiases = ob;
      outputBiases = new float[L4];
      outputBiasesTarget = new float[L4];
      for (size_t i = 0; i < L4; i++) outputBiases[i]=ob[i];

      target_sync();

      q = queue(sycl::default_selector{} );
      // q = queue(sycl::gpu_selector{} );
      cout << "Trainer constrtuction done. Using device: " << q.get_device().get_info<info::device::name>() << "\n";
    
      in_arr = new float[BS][L1];
      out1_arr = new float[BS][L2];
      out2_arr = new float[BS][L3];
      outQ_arr = new float[BS][L4];
      in_arr_n = new float[BS][L1];
      out1_arr_snt = new float[BS][L2];
      out2_arr_snt = new float[BS][L3];
      outQtarg_arr = new float[BS][L4];
      outputBiasGradients = new float[L4]; //L3
      error_signals = new float[BS][L4];
      outputWeightsGradients = new float[L3][L4];
      hidden_error_signals2 = new float[BS][L3];
      hiddenWeightsGradients2 = new float[L2][L3];
      hiddenBiasGradients2 = new float[L3];
      hidden_error_signals1 = new float[BS][L2];
      hiddenWeightsGradients1 = new float[L1][L2];
      hiddenBiasGradients1 = new float[L2];
      cout << "Memory Allocation: done\n";
    }


    ~DQNTrainer() {
      del_2darr<L2>(hiddenWeights1);
      del_2darr<L2>(hiddenWeightsTarget1);
      del_arr(hiddenBiases1);
      del_arr(hiddenBiasesTarget1);
      del_2darr<L3>(hiddenWeights2);
      del_2darr<L3>(hiddenWeightsTarget2);
      del_arr(hiddenBiases2);
      del_arr(hiddenBiasesTarget2);
      del_2darr<L4>(outputWeights);
      del_2darr<L4>(outputWeightsTarget);
      del_arr(outputBiases);
      del_arr(outputBiasesTarget);
      del_2darr<L1>(in_arr);
      del_2darr<L2>(out1_arr);
      del_2darr<L3>(out2_arr);
      del_2darr<L4>(outQ_arr);
      del_2darr<L1>(in_arr_n);
      del_2darr<L2>(out1_arr_snt);
      del_2darr<L3>(out2_arr_snt);
      del_2darr<L4>(outQtarg_arr);
      del_arr(outputBiasGradients);
      del_2darr<L4>(error_signals);
      del_2darr<L4>(outputWeightsGradients);
      del_2darr<L3>(hidden_error_signals2);
      del_2darr<L3>(hiddenWeightsGradients2);
      del_arr(hiddenBiasGradients2);
      del_2darr<L2>(hidden_error_signals1);
      del_2darr<L2>(hiddenWeightsGradients1);
      del_arr(hiddenBiasGradients1);
      cout<<"Trainer destructed, memory freed\n";
    }


    void target_sync(){
      for (size_t i = 0; i < L1; i++) {
        for (size_t j = 0; j < L2; j++){
          hiddenWeightsTarget1[i][j] = hiddenWeights1[i][j];
        }  
      }
      for (size_t i = 0; i < L2; i++) hiddenBiasesTarget1[i]=hiddenBiases1[i];

      for (size_t i = 0; i < L2; i++) {
        for (size_t j = 0; j < L3; j++){
          hiddenWeightsTarget2[i][j] = hiddenWeights2[i][j];
        }  
      }
      for (size_t i = 0; i < L3; i++) hiddenBiasesTarget2[i]=hiddenBiases2[i];

      for (size_t i = 0; i < L3; i++) {
        for (size_t j = 0; j < L4; j++){
          outputWeightsTarget[i][j] = outputWeights[i][j];
        }  
      }
      for (size_t i = 0; i < L4; i++) outputBiasesTarget[i]=outputBiases[i];
    }

    // return new prriorities as ouptut
    std::vector<float> train_itr(std::vector<std::vector<float>> s_in, std::vector<int> actions, std::vector<std::vector<float>> snext_in, std::vector<float> rewards,
                   std::vector<int> dones, bool targ_sync){
      // float(*in_arr)[L1] = new float[BS][L1];
      for (size_t i=0; i<BS; i++){
        for (size_t j=0; j<L1; j++){
          in_arr[i][j]=s_in[i][j];
        }
      }
      // float(*in_arr_n)[L1] = new float[BS][L1];
      for (size_t i=0; i<BS; i++){
        for (size_t j=0; j<L1; j++){
          in_arr_n[i][j]=snext_in[i][j];
        }
      }
      // == FW: policy net
      MMFW<BS, L1,L2>(in_arr, hiddenWeights1, hiddenBiases1, out1_arr);
      // VerifyResult(out1_arr);//success - Aug 15
      q.wait();
      MMFW<BS, L2,L3>(out1_arr, hiddenWeights2, hiddenBiases2, out2_arr);
      MMFW<BS, L3,L4>(out2_arr, outputWeights, outputBiases, outQ_arr);
      q.wait();
      
      // == FW: taget net
      MMFW<BS, L1,L2>(in_arr_n, hiddenWeightsTarget1, hiddenBiasesTarget1, out1_arr_snt);
      q.wait();
      MMFW<BS, L2,L3>(out1_arr_snt, hiddenWeightsTarget2, hiddenBiasesTarget2, out2_arr_snt);
      q.wait();
      MMFW<BS, L3,L4>(out2_arr_snt, outputWeightsTarget, outputBiasesTarget, outQtarg_arr);
      q.wait();
      float targ_max_Q[BS];
      argmax_a_Q(outQtarg_arr,targ_max_Q);

      // == Loss
      std::vector<float> out_prs;
      out_prs.resize(BS);
      for (size_t j=0;j<L4;j++){
        outputBiasGradients[j]=0;
        for (size_t i=0;i<BS;i++){
          float targi=rewards[i] + (1-dones[i])*GAMMA*targ_max_Q[i];
          error_signals[i][j]=(targi-outQ_arr[i][j])* (outQ_arr[i][j] > 0 ? 1 : 0);
          outputBiasGradients[j]+= error_signals[i][j];
          out_prs[i]=error_signals[i][actions[i]];
        }
        outputBiasGradients[j]/=BS;
        // == WU - output bias 
        outputBiases[j] -= outputBiasGradients[j];
      }

      // == output Layer grad aggregation
      MM_WA<L3, BS, L4>(out2_arr, error_signals, outputWeightsGradients);

      //== output Layer BW
      MMBW<BS, L4,L3>(error_signals, outputWeights, hidden_error_signals2);
      q.wait();
      // == WU - hidden bias 2
      for (size_t i=0; i<L3; i++){
        hiddenBiasGradients2[i]=0;
        for (size_t j=0; j<BS; j++){
          hiddenBiasGradients2[i]+=hidden_error_signals2[j][i];
        }
        hiddenBiasGradients2[i]/=BS;
        hiddenBiases2[i] -= hiddenBiasGradients2[i];
      }
      // == hidden Layer grad aggregation
      // BS*L2 x BS*L3 = L2*L3
      MM_WA<L2, BS, L3>(out1_arr, hidden_error_signals2, hiddenWeightsGradients2);

      //== BW
      MMBW<BS, L3,L2>(hidden_error_signals2, hiddenWeights2, hidden_error_signals1);
      q.wait();
      // == WU - hidden bias 2
      for (size_t i=0; i<L2; i++){
        hiddenBiasGradients1[i]=0;
        for (size_t j=0; j<BS; j++){
          hiddenBiasGradients1[i]+=hidden_error_signals1[j][i];
        }
        hiddenBiasGradients1[i]/=BS;
        hiddenBiases1[i] -= hiddenBiasGradients1[i];
      }
      // == hidden Layer grad aggregation
      // BS*L1 x BS*L2 = L1*L2
      MM_WA<L1, BS, L2>(in_arr, hidden_error_signals1, hiddenWeightsGradients1);

      // == WU-weights
      M_WU<L1,L2>(hiddenWeights1, hiddenWeightsGradients1);
      M_WU<L2,L3>(hiddenWeights2, hiddenWeightsGradients2);
      M_WU<L3,L4>(outputWeights, outputWeightsGradients);

      if (targ_sync){
        target_sync();
      }

      // cout<<"train_itr done\n";
      return out_prs;

    }

    params_out updated_params(){
      params_out param_pack;
      for (size_t i=0; i<L2; i++){
        param_pack.hiddenBiases1_d[i] = hiddenBiases1[i];
      }
      for (size_t i=0; i<L3; i++){
        param_pack.hiddenBiases2_d[i] = hiddenBiases2[i];
      }
      for (size_t i=0; i<L4; i++){
        param_pack.outputBiases_d[i] = outputBiases[i];
      }
      
      for (size_t i=0; i<L1; i++){
        for (size_t j=0; j<L2; j++){
          param_pack.hiddenWeights1_d[i][j] = hiddenWeights1[i][j];
        }
      }
      for (size_t i=0; i<L2; i++){
        for (size_t j=0; j<L3; j++){
          param_pack.hiddenWeights2_d[i][j] = hiddenWeights2[i][j];
        }
      }
      for (size_t i=0; i<L3; i++){
        for (size_t j=0; j<L4; j++){
          param_pack.outputWeights_d[i][j] = outputWeights[i][j];
        }
      }
      return param_pack;
    }

    

    // In FW: bsize is batch size, LL is earlier-layer dim, NL is later-layer dim. bsize*LL x LL*NL = bsize*NL
    template<int bsize, int LL, int NL>
    void MMFW(float (*Mx_in)[LL], float (*weights)[NL], float *biases, float (*Mx_out)[NL]){
      // Intialize output buffer
      for (int i = 0; i < bsize; i++)
        for (int j = 0; j < NL; j++) Mx_out[i][j] = 0.0f;

      try {

        buffer<float, 2> in_buf(reinterpret_cast<float *>(Mx_in), range(bsize, LL));
        buffer<float, 2> w_buf(reinterpret_cast<float *>(weights), range(LL, NL));
        buffer<float> b_buf(biases, range(NL));
        // buffer out_buf(reinterpret_cast<float **>(c_back), range(M, P));
        buffer<float, 2> out_buf(reinterpret_cast<float *>(Mx_out), range(bsize, NL));

        // cout << "Problem size: c(" << bsize << "," << NL << ") = a(" << bsize << "," << LL
        //     << ") * b(" << LL << "," << NL << ")\n";

        // Submit command group to queue to multiply matrices: c = a * b
        q.submit([&](auto &h) {
          // Read from a and b, write to c
          accessor a(in_buf, h, read_only);
          accessor w(w_buf, h, read_only);
          accessor b(b_buf, h, read_only);
          accessor c(out_buf, h, write_only);

          int width_a = in_buf.get_range()[1];

          // Execute kernel.
          h.parallel_for(range(bsize, NL), [=](auto index) {
            // Get global position in Y direction.
            int row = index[0];
            // Get global position in X direction.
            int col = index[1];
            float sum = 0.0f;
            // Accumulate
            for (int i = 0; i < width_a; i++) {
              sum += a[row][i] * w[i][col];
            }
            sum+=b[row]; //add bias

            c[index]=std::max(0.0f, sum); //apply relu
          });
        });
      } catch (sycl::exception const &e) {
        cout << "FW: An exception is caught while multiplying matrices.\n";
        terminate();
      }
    }

    // In BW: bsize is batch size, LL is later-layer dim, NL is earlier-layer dim. bsize*LL x LL*NL = bsize*NL
    template<int bsize, int LL, int NL> //LL=L3, NL=L2
    void MMBW(float (*Err_in)[LL], float (*weights)[LL], float (*Mx_out)[NL]){
      // Intialize output buffer
      for (int i = 0; i < bsize; i++)
        for (int j = 0; j < NL; j++) Mx_out[i][j] = 0.0f;

      // Initialize the device queue with the default selector. The device queue is
      // used to enqueue kernels. It encapsulates all states needed for execution.
      try {

        buffer<float, 2> in_buf(reinterpret_cast<float *>(Err_in), range(bsize, LL));
        buffer<float, 2> w_buf(reinterpret_cast<float *>(weights), range(NL,LL));
        buffer<float, 2> out_buf(reinterpret_cast<float *>(Mx_out), range(bsize, NL));

        // Submit command group to queue to multiply matrices: c = a * b
        q.submit([&](auto &h) {
          // Read from a and b, write to c
          accessor a(in_buf, h, read_only);
          accessor w(w_buf, h, read_only);
          accessor c(out_buf, h, write_only);

          int width_a = in_buf.get_range()[1]; //L3

          // Execute kernel.
          h.parallel_for(range(bsize, NL), [=](auto index) {
            // Get global position in Y direction.
            int row = index[0]; //0...BS
            // Get global position in X direction.
            int col = index[1]; //0...L2
            float sum = 0.0f;
            // Accumulate
            for (int i = 0; i < width_a; i++) { //0...L3
              sum += a[row][i] * w[col][i];
            }
            c[index]=sum > 0 ? 1 : 0; //apply relu der
          });

        });
      } catch (sycl::exception const &e) {
        cout << "BW: An exception is caught while multiplying matrices.\n";
        terminate();
      }
    }

    // In WA: bsize is batch size, LL is earlier-layer dim, NL is later-layer dim. bsize*LL x LL*NL = bsize*NL
    template<int LL, int bsize, int NL> 
    void MM_WA(float (*Act_in)[LL], float (*Err_in)[NL], float (*Mx_out)[NL]){
      // Intialize output buffer
      for (int i = 0; i < LL; i++)
        for (int j = 0; j < NL; j++) Mx_out[i][j] = 0.0f;

      try {

        buffer<float, 2> in1_buf(reinterpret_cast<float *>(Act_in), range(bsize, LL));
        buffer<float, 2> in2_buf(reinterpret_cast<float *>(Err_in), range(bsize, NL));
        buffer<float, 2> out_buf(reinterpret_cast<float *>(Mx_out), range(LL, NL));

        // Submit command group to queue to multiply matrices: c = a * b
        q.submit([&](auto &h) {
          accessor a(in1_buf, h, read_only);
          accessor b(in2_buf, h, read_only);
          accessor c(out_buf, h, write_only);
          // Execute kernel.
          h.parallel_for(range(LL, NL), [=](auto index) {
            int row = index[0]; //0...LL
            int col = index[1]; //0...NL
            float sum = 0.0f;
            for (int i = 0; i < bsize; i++) { //0...BS
              sum += a[i][row] * b[i][col];
            }
            c[index]=sum/bsize; //average all gradients
          });
        });
      } catch (sycl::exception const &e) {
        cout << "WA: An exception is caught while multiplying matrices.\n";
        terminate();
      }
    }
    template<int LL, int NL> 
    void M_WU(float (*weights)[NL], float (*wg)[NL]){
      try {
        buffer<float, 2> wg_buf(reinterpret_cast<float *>(wg), range(LL, NL));
        buffer<float, 2> w_buf(reinterpret_cast<float *>(weights), range(LL, NL));
        // Submit command group to queue to multiply matrices: c = a * b
        q.submit([&](auto &h) {
          accessor a(wg_buf, h, read_only);
          accessor w(w_buf, h, read_write);
          // Execute kernel.
          h.parallel_for(range(LL, NL), [=](auto index) {
            w[index] -= a[index];
          });
        });
      } catch (sycl::exception const &e) {
        cout << "WA: An exception is caught while multiplying matrices.\n";
        terminate();
      }
    }

    // input: BS*L3; output: BS
    void argmax_a_Q(float (*in)[L4],float *out){
      for (size_t i=0;i<BS;i++){
        float max_Q=-9999;
        for (size_t j=0;j<L4;j++){
          if(in[i][j]>max_Q){
            max_Q=in[i][j];
          }
        }
        out[i]=max_Q;
      }
    }

    void del_arr(float *arr){
      delete[] arr;
    }

    template<int L>
    void del_2darr(float (*arr)[L]){
      delete[] arr;
    }

};



bool ValueSame(float a, float b) {
  return fabs(a - b) < numeric_limits<float>::epsilon();
}


int main() {
  cout<<"start"<<std::endl;
  std::vector<std::vector<float>> hw1;
  hw1.resize(L1);
  for (size_t i = 0; i < L1; i++){
    hw1[i].resize(L2);
    for (ssize_t j=0; j<L2; j++){
      hw1[i][j] = (i + 1.0f)/2;
    }
  }
  std::vector<float> hb1;
  hb1.resize(L2);
  for (size_t i = 0; i < L2; i++){hb1[i]=0.1;}

  std::vector<std::vector<float>> hw2;
  hw2.resize(L2);
  for (size_t i = 0; i < L2; i++){
    hw2[i].resize(L3);
    for (ssize_t j=0; j<L3; j++){
      hw2[i][j] = (i + 1.0f)/2;
    }
  }
  std::vector<float> hb2;
  hb2.resize(L3);
  for (size_t i = 0; i < L3; i++){hb2[i]=0.1;}


  std::vector<std::vector<float>> ow;
  ow.resize(L3);
  for (size_t i = 0; i < L3; i++){
    ow[i].resize(L4);
    for (ssize_t j=0; j<L4; j++){
      ow[i][j] = (i + 1.0f);
    }
  }
  std::vector<float> ob;
  ob.resize(L4);
  for (size_t i = 0; i < L4; i++){ob[i]=0.2;}

  DQNTrainer trainer(hw1, hb1, hw2, hb2, ow, ob);

  std::vector<std::vector<float>> inputs;
  inputs.resize(BS);
  for (size_t i = 0; i < BS; i++){
    inputs[i].resize(L1);
    for (size_t j=0; j<L1; j++){
      inputs[i][j] = 0.1f;
    }
  }
  std::vector<std::vector<float>> snt_inputs;
  snt_inputs.resize(BS);
  for (size_t i = 0; i < BS; i++){
    snt_inputs[i].resize(L1);
    for (size_t j=0; j<L1; j++){
      snt_inputs[i][j] = 0.5f;
    }
  }

  std::vector<float> rs;
  rs.resize(BS);
  std::vector<int> ds;
  ds.resize(BS);
  std::vector<int> as; //actions can be 0 or 1
  as.resize(BS);
  for (size_t i = 0; i < BS; i++){
    rs[i]=1.5f;
    ds[i]=0;
    if(i==0)as[i]=0;
    else{
      if(as[i-1]==0)as[i]=1;
      else as[i]=0;
    }
  }

  std::vector<float> new_prs;


  new_prs = trainer.train_itr(inputs,as,snt_inputs,rs,ds,false);
  cout<<"new_prs: [";
  for (size_t i = 0; i < BS; i++){
    cout<<new_prs[i]<<' ';
  }
  cout<<"]\n";
  trainer.train_itr(inputs,as,snt_inputs,rs,ds,true);
  params_out dpack = trainer.updated_params();
  dpack.print_params();
  return 0;
  // =============== END MAIN =================
}

// // int VerifyResult(float (*c_back)[P]) {
// int VerifyResult(float (*c_back)[P]) {
//   // Check that the results are correct by comparing with host computing.
//   int i, j, k;

//   // 2D arrays on host side.
//   float(*a_host)[N] = new float[M][N];
//   float(*b_host)[P] = new float[N][P];
//   float(*c_host)[P] = new float[M][P];

//   // Each element of matrix a is 1.
//   for (i = 0; i < M; i++)
//     for (j = 0; j < N; j++) a_host[i][j] = 1.0f;

//   // Each column of b_host is the sequence 1,2,...,N
//   for (i = 0; i < N; i++)
//     for (j = 0; j < P; j++) b_host[i][j] = i + 1.0f;

//   // c_host is initialized to zero.
//   for (i = 0; i < M; i++)
//     for (j = 0; j < P; j++) c_host[i][j] = 0.0f;

//   for (i = 0; i < M; i++) {
//     for (k = 0; k < N; k++) {
//       // Each element of the product is just the sum 1+2+...+n
//       for (j = 0; j < P; j++) {
//         c_host[i][j] += a_host[i][k] * b_host[k][j];
//       }
//     }
//   }

//   bool mismatch_found = false;

//   // Compare host side results with the result buffer from device side: print
//   // mismatched data 5 times only.
//   int print_count = 0;

//   for (i = 0; i < M; i++) {
//     for (j = 0; j < P; j++) {
//       if (!ValueSame(c_back[i][j], c_host[i][j])) {
//         cout << "Fail - The result is incorrect for element: [" << i << ", "
//              << j << "], expected: " << c_host[i][j]
//              << ", but found: " << c_back[i][j] << "\n";
//         mismatch_found = true;
//         print_count++;
//         if (print_count == 5) break;
//       }
//     }

//     if (print_count == 5) break;
//   }

//   delete[] a_host;
//   delete[] b_host;
//   delete[] c_host;

//   if (!mismatch_found) {
//     cout << "Success - The results are correct!\n";
//     return 0;
//   } else {
//     cout << "Fail - The results mismatch!\n";
//     return -1;
//   }
// }