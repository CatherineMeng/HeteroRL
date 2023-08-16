
#include <sycl/sycl.hpp>
#include <iostream>
#include <limits>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace std;
using namespace sycl;

// MLP metadata.
constexpr int BS = 16;
constexpr int L1 = 4;
constexpr int L2 = 64;
constexpr int L3 = 2;

constexpr int M = BS;
constexpr int N = L1;
constexpr int P = L2;

constexpr float GAMMA = 0.1;
constexpr float LEARNRATE = 1.0;

int VerifyResult(float (*c_back)[P]);
// struct params {
//     std::vector<std::vector<float>> hiddenWeights;
//     std::vector<std::vector<float>> hiddenBiases;
//     std::vector<std::vector<float>> outputWeights;
//     std::vector<std::vector<float>> outputBiases;
// };

class DQNTrainer {     
  public:  
    // --- weights and biases ---
    float(*hiddenWeights)[L2];
    float * hiddenBiases;
    float(*outputWeights)[L3];
    float * outputBiases;
    float(*hiddenWeightsTarget)[L2];
    float * hiddenBiasesTarget;
    float(*outputWeightsTarget)[L3];
    float * outputBiasesTarget;
    // --- end weights and biases ---
    
    // --- kernel queue ---
    queue q;
    // --- end kernel queue ---

    // --- intermiediate results used in each training itr ---
    float(*in_arr)[L1];
    float(*out1_arr)[L2];
    float(*outQ_arr)[L3];

    float(*in_arr_n)[L1];
    float(*out1_arr_snt)[L2];
    float(*outQtarg_arr)[L3];

    float *outputBiasGradients; 
    float(*error_signals)[L3];
    float(*outputWeightsGradients)[L3];
    float(*hidden_error_signals)[L2];
    float *hiddenBiasGradients; 
    float(*hiddenWeightsGradients)[L2];

    // float targ_max_Q=new float[BS];
    // --- end intermiediate results used in each training itr ---

    // Args of the constructor will be passed from external python code (pytorch xavier init)
    DQNTrainer(std::vector<std::vector<float>> hw, std::vector<float> hb,
    std::vector<std::vector<float>> ow, std::vector<float> ob) {  
      // WB.hiddenWeights = hw;
      hiddenWeights = new float[L1][L2];
      hiddenWeightsTarget = new float[L1][L2];
      for (size_t i = 0; i < L1; i++) {
        for (size_t j = 0; j < L2; j++){
          hiddenWeights[i][j] = hw[i][j];
        }  
      }
      // WB.hiddenBiases = hb;
      hiddenBiases = new float[L2];
      hiddenBiasesTarget = new float[L2];
      for (size_t i = 0; i < L2; i++) hiddenBiases[i]=hb[i];

      // WB.outputWeights = ow;
      outputWeights = new float[L2][L3];
      outputWeightsTarget = new float[L2][L3];
      for (size_t i = 0; i < L2; i++) {
        for (size_t j = 0; j < L3; j++){
          outputWeights[i][j] = ow[i][j];
        }  
      }
      // WB.outputBiases = ob;
      outputBiases = new float[L3];
      outputBiasesTarget = new float[L3];
      for (size_t i = 0; i < L3; i++) outputBiases[i]=ob[i];

      target_sync();

      // q = queue(sycl::default_selector{} );
      q = queue(sycl::gpu_selector{} );
      cout << "Trainer constrtuction done. Using device: " << q.get_device().get_info<info::device::name>() << "\n";
    
      in_arr = new float[BS][L1];
      out1_arr = new float[BS][L2];
      outQ_arr = new float[BS][L3];
      in_arr_n = new float[BS][L1];
      out1_arr_snt = new float[BS][L2];
      outQtarg_arr = new float[BS][L3];
      outputBiasGradients = new float[L3]; //L3
      error_signals = new float[BS][L3];
      outputWeightsGradients = new float[L2][L3];
      hidden_error_signals = new float[BS][L2];
      hiddenWeightsGradients = new float[L1][L2];
      hiddenBiasGradients = new float[L2];
      cout << "Memory Allocation: done\n";
    }


    ~DQNTrainer() {
      del_2darr<L2>(hiddenWeights);
      del_2darr<L2>(hiddenWeightsTarget);
      del_arr(hiddenBiases);
      del_arr(hiddenBiasesTarget);
      del_2darr<L3>(outputWeights);
      del_2darr<L3>(outputWeightsTarget);
      del_arr(outputBiases);
      del_arr(outputBiasesTarget);
      del_2darr<L1>(in_arr);
      del_2darr<L2>(out1_arr);
      del_2darr<L3>(outQ_arr);
      del_2darr<L1>(in_arr_n);
      del_2darr<L2>(out1_arr_snt);
      del_2darr<L3>(outQtarg_arr);
      del_arr(outputBiasGradients);
      del_2darr<L3>(error_signals);
      del_2darr<L3>(outputWeightsGradients);
      del_2darr<L2>(hidden_error_signals);
      del_2darr<L2>(hiddenWeightsGradients);
      del_arr(hiddenBiasGradients);
      cout<<"Trainer destructed, memory freed\n";
    }


    void target_sync(){
      for (size_t i = 0; i < L1; i++) {
        for (size_t j = 0; j < L2; j++){
          hiddenWeightsTarget[i][j] = hiddenWeights[i][j];
        }  
      }
      for (size_t i = 0; i < L2; i++) hiddenBiasesTarget[i]=hiddenBiases[i];
      for (size_t i = 0; i < L2; i++) {
        for (size_t j = 0; j < L3; j++){
          outputWeightsTarget[i][j] = outputWeights[i][j];
        }  
      }
      for (size_t i = 0; i < L3; i++) outputBiasesTarget[i]=outputBiases[i];
    }

    void train_itr(std::vector<std::vector<float>> s_in, std::vector<std::vector<float>> snext_in, std::vector<float> rewards,
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
      MMFW<BS, L1,L2>(in_arr, hiddenWeights, hiddenBiases, out1_arr);
      // VerifyResult(out1_arr);//success - Aug 15
      q.wait();
      MMFW<BS, L2,L3>(out1_arr, outputWeights, outputBiases, outQ_arr);
      q.wait();
      
      // == FW: taget net
      MMFW<BS, L1,L2>(in_arr_n, hiddenWeightsTarget, hiddenBiasesTarget, out1_arr_snt);
      q.wait();
      MMFW<BS, L2,L3>(out1_arr_snt, outputWeightsTarget, outputBiasesTarget, outQtarg_arr);
      q.wait();
      float targ_max_Q[BS];
      argmax_a_Q(outQtarg_arr,targ_max_Q);

      // == Loss
      for (size_t j=0;j<L3;j++){
        outputBiasGradients[j]=0;
          for (size_t i=0;i<BS;i++){
          float targi=rewards[i] + (1-dones[i])*GAMMA*targ_max_Q[i];
          error_signals[i][j]=(targi-outQ_arr[i][j])* (outQ_arr[i][j] > 0 ? 1 : 0);
          outputBiasGradients[j]+= error_signals[i][j];
        }
        outputBiasGradients[j]/=BS;
        // == WU - output bias 
        outputBiases[j] -= outputBiasGradients[j];
      }

      // == output Layer grad aggregation
      MM_WA<L2, BS, L3>(out1_arr, error_signals, outputWeightsGradients);

      //== output Layer BW
      MMBW<BS, L3,L2>(error_signals, outputWeights, hidden_error_signals);

      q.wait();

      // == WU - hidden bias
      for (size_t i=0; i<L2; i++){
        hiddenBiasGradients[i]=0;
        for (size_t j=0; j<BS; j++){
          hiddenBiasGradients[i]+=hidden_error_signals[j][i];
        }
        hiddenBiasGradients[i]/=BS;
        hiddenBiases[i] -= hiddenBiasGradients[i];
      }

      // == hidden Layer grad aggregation
      // BS*L1 x BS*L2 = L1*L2
      MM_WA<L1, BS, L2>(in_arr, hidden_error_signals, hiddenWeightsGradients);
      // del_2darr<L1>(in_arr);

      // == WU-weights
      M_WU<L1,L2>(hiddenWeights, hiddenWeightsGradients);
      M_WU<L2,L3>(outputWeights, outputWeightsGradients);

      if (targ_sync){
        target_sync();
      }

      cout<<"train_itr done\n";

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
    void argmax_a_Q(float (*in)[L3],float *out){
      for (size_t i=0;i<BS;i++){
        float max_Q=-9999;
        for (size_t j=0;j<L3;j++){
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
  std::vector<std::vector<float>> hw;
  hw.resize(L1);
  for (size_t i = 0; i < L1; i++){
    hw[i].resize(L2);
    for (ssize_t j=0; j<L2; j++){
      hw[i][j] = i + 1.0f;
    }
  }
  std::vector<float> hb;
  hb.resize(L2);
  for (size_t i = 0; i < L2; i++){hb[i]=0;}

  std::vector<std::vector<float>> ow;
  ow.resize(L2);
  for (size_t i = 0; i < L2; i++){
    for (ssize_t j=0; j<L3; j++){
      ow[i].resize(L3);
      ow[i][j] = i + 1.0f;
    }
  }
  std::vector<float> ob;
  ob.resize(L3);
  for (size_t i = 0; i < L3; i++){ob[i]=0;}

  DQNTrainer trainer(hw, hb,ow, ob);

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
  for (size_t i = 0; i < BS; i++){
    rs[i]=1.5f;
    ds[i]=0;
  }

  trainer.train_itr(inputs,snt_inputs,rs,ds,false);
  trainer.train_itr(inputs,snt_inputs,rs,ds,true);
  return 0;
  // =============== END MAIN =================
}

// int VerifyResult(float (*c_back)[P]) {
int VerifyResult(float (*c_back)[P]) {
  // Check that the results are correct by comparing with host computing.
  int i, j, k;

  // 2D arrays on host side.
  float(*a_host)[N] = new float[M][N];
  float(*b_host)[P] = new float[N][P];
  float(*c_host)[P] = new float[M][P];

  // Each element of matrix a is 1.
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) a_host[i][j] = 1.0f;

  // Each column of b_host is the sequence 1,2,...,N
  for (i = 0; i < N; i++)
    for (j = 0; j < P; j++) b_host[i][j] = i + 1.0f;

  // c_host is initialized to zero.
  for (i = 0; i < M; i++)
    for (j = 0; j < P; j++) c_host[i][j] = 0.0f;

  for (i = 0; i < M; i++) {
    for (k = 0; k < N; k++) {
      // Each element of the product is just the sum 1+2+...+n
      for (j = 0; j < P; j++) {
        c_host[i][j] += a_host[i][k] * b_host[k][j];
      }
    }
  }

  bool mismatch_found = false;

  // Compare host side results with the result buffer from device side: print
  // mismatched data 5 times only.
  int print_count = 0;

  for (i = 0; i < M; i++) {
    for (j = 0; j < P; j++) {
      if (!ValueSame(c_back[i][j], c_host[i][j])) {
        cout << "Fail - The result is incorrect for element: [" << i << ", "
             << j << "], expected: " << c_host[i][j]
             << ", but found: " << c_back[i][j] << "\n";
        mismatch_found = true;
        print_count++;
        if (print_count == 5) break;
      }
    }

    if (print_count == 5) break;
  }

  delete[] a_host;
  delete[] b_host;
  delete[] c_host;

  if (!mismatch_found) {
    cout << "Success - The results are correct!\n";
    return 0;
  } else {
    cout << "Fail - The results mismatch!\n";
    return -1;
  }
}