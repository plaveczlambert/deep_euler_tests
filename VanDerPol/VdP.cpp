/*Deep Euler implementation of Van der Pol equation*/

#include <iostream>
#include <fstream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <string>
#include <chrono>

#include <torch/script.h>

using namespace std;

const int nn_inputs = 3;
const int nn_outputs = 2;
c10::TensorOptions global_tensor_op;

//Modify these to load the correct model
string file_name = "../simulations/vdp_dem_test.txt";
string model_file = "../training/traced_model_vdp.pt"; 
string scaler_file = "../training/scaler_vdp.psca";
string output_log = "../simulations/output_compare.txt";

typedef double value_type;
typedef vector<value_type> state_type;

struct std_scaler {
	torch::Tensor mean;
	torch::Tensor scale;

	torch::Tensor operator()(torch::Tensor tensor) {
		return (tensor - mean) / scale;
	}
	torch::Tensor inverse_transform(torch::Tensor tensor) {
		return tensor * scale + mean;
	}
	void parse(istream& is, int numel) {
		mean = torch::ones({ 1, numel });
		is.get();
		double temp = 0.0;
		for (int i = 0; i < numel; i++) {
			is >> temp;
			mean[0][i] = temp;
		}
		is.get();
		is.get();
		scale = torch::ones({ 1, numel });
		is.get();
		for (int i = 0; i < numel; i++) {
			is >> temp;
			scale[0][i] = temp;
		}
		is.get();
		is.get();
	}
};

struct norm_scaler {
	torch::Tensor data_min;
	torch::Tensor data_max;
	double min = 0;
	double max = 0;

	torch::Tensor operator()(torch::Tensor tensor) {
		torch::Tensor X_std = (tensor - data_min) / (data_max - data_min);
		return X_std * (max - min) + min;
	}
	torch::Tensor inverse_transform(torch::Tensor tensor) {
		torch::Tensor Y_std = (tensor - min) / (max - min);
		return Y_std * (data_max - data_min) + data_min;
	}
	void parse(istream& is) {
		data_min = torch::ones({ 1,nn_outputs });
		is.get();
		double temp = 0.0;
		for (int i = 0; i < nn_outputs; i++) {
			is >> temp;
			data_min[0][i] = temp;
		}
		is.get();
		is.get();
		data_max = torch::ones({ 1, nn_outputs });
		is.get();
		for (int i = 0; i < nn_outputs; i++) {
			is >> temp;
			data_max[0][i] = temp;
		}
		is.get(); //']'
		is.get(); //'\n'
		is >> min;
		is >> max;
	}
};

//ode function of Van der Pol equation
class VdP {
	std::ofstream outputs_out;
	double mu = 1.5;
public:
	torch::jit::script::Module model; //the neural network
	torch::Tensor inputs; //reused tensor of inputs
	std_scaler in_transf;
	std_scaler out_transf;

	VdP(std::array<double, nn_inputs> inital_values) {
		outputs_out.open(output_log);
		outputs_out.precision(17);
		outputs_out.flags(ios::scientific);

		//metamodel initializations
		inputs = torch::ones({ 1, nn_inputs }, global_tensor_op);
		
		for (int i = 0; i < nn_inputs; i++) inputs[0][i] = inital_values[i];
		
		try {
			model = torch::jit::load(model_file);
			std::vector<torch::jit::IValue> inp;
			inp.push_back(torch::ones({ 1, nn_inputs }, global_tensor_op));
			cout << "777" << endl;
			std::cout << inp << endl;
			cout << "7" << endl;
			// Execute the model and turn its output into a tensor.
			at::Tensor output = model.forward(inp).toTensor().detach();
			cout << "77" << endl;
			std::cout << output << endl;
		}
		catch (const c10::Error& e) {
			std::cerr << "Error loading the model: " << e.what() << endl;
			exit(-1);
		}
		cout << "43" << endl;
		ifstream in(scaler_file);
		if (!in) {
			std::cerr << "Error loading the scalers." << endl;
			exit(-1);
		}
		out_transf.parse(in, nn_outputs);
		in_transf.parse(in, nn_inputs);
		in.close();
	}

	/*Rewrites the errors array with the predicted local truncation errors*/
	double* local_error(double t, double t_next, const double* x, double* errors) {

		//updating inputs
		inputs[0][0] = t_next-t;

		for (int i = 0; i < nn_inputs - 1; i++) {
			inputs[0][i + 1] = x[i];
		}
		outputs_out << t << " ";
		//log inputs
		for (int i = 0; i < nn_inputs; i++) {
			outputs_out << inputs[0][i].item<double>() << " ";
		}

		//scaling
		torch::Tensor scaled = in_transf(inputs);
		std::vector<torch::jit::IValue> inps;
		inps.push_back(scaled);
		//evaluating
		torch::Tensor loc_trun_err = model.forward(inps).toTensor().detach();
		loc_trun_err = 	out_transf.inverse_transform(loc_trun_err);

		//log outputs
		for (int i = 0; i < nn_outputs; i++) {
			errors[i] = loc_trun_err[0][i].item<double>();
			outputs_out << errors[i] << " ";
		}
		outputs_out << endl;
		return errors;
	}

	/*ODE function. In the pointer x the values are rewritten with the computed slopes*/
	void operator()(double t, double* x) {
		double dxdt = -mu * (x[1] * x[1] - 1) * x[0] - x[1];
		x[1] = x[0];
		x[0] = dxdt;
	}
};

class ODESolver
{
public:

	ODESolver(int order) :order(order) {};

	bool setInitialCondition(double* conds, double at) {
		begin_t = at;
		init_conds = (double*)malloc(sizeof(double) * order);
		for (int u = 0; u < order; u++) {
			init_conds[u] = conds[u];
		}
		return true;
	}
	void setTimeStep(double dt) {
		delta_t = dt;
	}
	bool setStepNumber(int steps) {
		max_l = steps;
		return true;
	}
	
	void solve(VdP& vdp, ostream& os) {

		double* vector = (double*)malloc(sizeof(double) * order);
		for (int u = 0; u < order; u++) {
			vector[u] = init_conds[u];
		}

		//preparations
		double* k = (double*)malloc(sizeof(double) * order);
		double* local_error = (double*)malloc(sizeof(double) * order);
		double t = begin_t;
		int l = 0;
		
		os << t;
		for (int i = 0; i < order; i++) {
			os << " " << vector[i];
		}
		os << endl;

		#pragma warning(disable:6011)
		while (l < max_l) {

			for (int j = 0; j < order; j++) {
				k[j] = vector[j];
			}
			vdp.local_error(t, t + delta_t, vector, local_error);
			vdp(t, k);
			for (int j = 0; j < order; j++) {
				vector[j] = vector[j] + delta_t * k[j] +delta_t * delta_t * local_error[j];
				//To change to Euler Method uncomment the following, comment out the previous
				//vector[j] = vector[j] + delta_t * k[j];																 
			}
			l++;
			t += delta_t;
			os << t;
			for (int i = 0; i < order; i++) {
				os << " " << vector[i];
			}
			os << endl;
		}
		free(k);
		free(vector);
	}

	~ODESolver(){
		free(init_conds);
	}
private:
	int type = 0;
	int order = 1;
	double* init_conds;
	double begin_t = 0;
	double delta_t = 0.1;
	int max_l = 10;
	double distance(double a, double b) {
		if (a < b)return b - a;
		else return a - b;
	}
};




int main() {
	global_tensor_op = torch::TensorOptions().dtype(torch::kFloat64);
	cout << "Van der Pol with metamodel started\n" << setprecision(17) << endl;

	double* x = new double[nn_outputs]{ 4.0, 3.0 };

	ofstream ofs(file_name);
	if(!ofs.is_open())exit(-1);
	ofs.precision(17);
	ofs.flags(ios::scientific);
	cout << "Writing file: " << file_name << endl;

	//initial conditions
	std::array<value_type, nn_inputs> initial_inputs = { 1e-5, x[0], x[1]};
	
	double t_start = 0.0;
	VdP bubi(initial_inputs);
	
	ODESolver solver(nn_outputs);
	solver.setInitialCondition(x, 0.0);
	solver.setTimeStep(0.1);
	solver.setStepNumber(500);

	cout << "Solving..." << endl;
	auto t1 = chrono::high_resolution_clock::now();
	solver.solve(bubi, ofs);
	auto t2 = chrono::high_resolution_clock::now();
	cout << "Time (ms):" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << endl;
	
	ofs.flush();
	ofs.close();

	cout << "Ready"<< endl;
	return 0;
}