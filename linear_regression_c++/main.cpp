#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <sstream>
#include <random>


using namespace std;

//gonna use while returning parameters
struct Parameters {
    vector<double> W;
    double b;
};
struct Gradients {
    double dW;
    double db;
};



vector<string> get_data() {
    ifstream file("tvmarketing.csv"); 
    vector<string> result;
    string line;
    while (getline(file, line)) {
        result.push_back(line); 
    }
    file.close();
    return result;
}

// for getting spesific column for usage 
vector<double> get_column(const vector<string>& data,int get_index){
    vector<double> res_column;
    for(int i=1; i<data.size();i++){
        //i=1 bsc of header of dataset
        stringstream ss(data[i]);
        string item;
        int index =0;
        while(getline(ss,item,',')){
            if (index==get_index){
                res_column.push_back(stod(item));
                break;
            }
            index++;
        }

    }
    return res_column;


}

//taking mean of a vector of doubles
double mean(const vector<double>& data) {
    double sum = 0.0;
    for (double num : data) {
        sum += num;
    }
    return sum / data.size();
}

double std_dev(const vector<double>&data,double mean_value){
    double sum =0.0;
    for(double x: data)sum+= (x-mean_value)*(x-mean_value);
    return sqrt(sum/data.size());
}

vector<double> normalize(const vector<double>&column,double mean_value,double std_dev){
    vector<double> result;
    for(double x : column){
        x= (x-mean_value)/std_dev;
        result.push_back(x);
    }
    return result;
}

Parameters initialize_parameters(int n_x = 1) {
    Parameters params;
    params.W.resize(n_x);

    // random generator
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dis(0.0, 0.01);

    for(int i = 0; i < n_x; i++) {
        params.W[i] = dis(gen);
    }

    params.b = 0.0;

    return params;
}
vector<double> forward_propagation(const vector<double>&normalized_sales,const Parameters&params){

    vector<double> W = params.W, Y_Hat;                 
    double b = params.b;
    
    for (double x: normalized_sales){
        double item= W[0]*x+b;
        Y_Hat.push_back(item);
    }
    return Y_Hat;
}
double compute_cost(const vector<double>& Y_Hat, const vector<double>& Y) {
    int m= Y_Hat.size();
    double cost=0.0;
    for(int i=0;i<m;i++){
        double diff=Y_Hat[i]-Y[i];
        cost+=(diff*diff);
    }
    return cost/(2*m);


}
Gradients backward_propagation(const vector<double>& Y_Hat,const vector<double>& X,const vector<double>& Y){

    vector<double> dZ(Y.size());
    
    for (size_t i = 0; i < Y.size(); ++i) {
        dZ[i] = Y_Hat[i] - Y[i];
    }
    double dW = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        dW += dZ[i] * X[i];
    }
    dW = dW / X.size();

    double db=0.0;
    for(double val:dZ) db+=val;
    db/=Y.size();

    Gradients grad;
    grad.dW = dW;
    grad.db = db;
    return grad;

}

Parameters update_parameters(const Parameters& params, const Gradients& grads, double learning_rate = 0.01){
    Parameters updated_params;

    // for now we have just one weight
    updated_params.W.resize(params.W.size());
    for (size_t i = 0; i < params.W.size(); ++i) {
        updated_params.W[i] = params.W[i] - learning_rate * grads.dW;
    }

    updated_params.b = params.b - learning_rate * grads.db;

    return updated_params;
}
Parameters train(const vector<double>& X, const vector<double>& Y, int num_iterations, double learning_rate) {
    Parameters params = initialize_parameters();

    // loop
    for (int i = 0; i < num_iterations; ++i) {
        vector<double> Y_Hat = forward_propagation(X, params);
        double cost = compute_cost(Y_Hat, Y);
        Gradients grads = backward_propagation(Y_Hat, X, Y);
        params = update_parameters(params, grads, learning_rate);

        if (i % 100 == 0) {
            cout << "iteration " << i << " cost = " << cost << endl;
        }
    }

    return params;
}
double predict(double raw_tv_budget,const Parameters& params, double mean_tv, double std_tv,double mean_sales, double std_sales) {
    double normalized_tv = (raw_tv_budget - mean_tv) / std_tv;
    double normalized_prediction = (params.W[0] * normalized_tv) + params.b;
    double real_sales_prediction = (normalized_prediction * std_sales) + mean_sales;

    return real_sales_prediction;
}



int main() {
    vector<string> data = get_data();
    vector<double> tv= get_column(data,0), sales= get_column(data,1);
    cout << "\ntotal lines: " << data.size() << endl;
    //get data    
    double mean_value_tv=mean(tv), mean_value_sales=mean(sales);

    double std_dev_tv=std_dev(tv,mean_value_tv),std_dev_sales=std_dev(sales,mean_value_sales);  

    vector<double> normalized_X=normalize(tv,mean_value_tv,std_dev_tv), normalized_Y=normalize(sales,mean_value_sales,std_dev_sales);
    
    Parameters trained_params = train(normalized_X, normalized_Y, 1000, 0.01);
    Parameters params;
    vector<double> W = trained_params.W;                 
    double b = trained_params.b;


    cout << "\n Final parameters" << endl;
    cout << "Weight: " << W[0] << endl;
    cout << "Bias:   " << b << endl;


    double test_budget = 2312; // example TV budget
    double predicted_sales = predict(test_budget, trained_params, mean_value_tv, std_dev_tv, mean_value_sales, std_dev_sales);

    cout << "\nPrediction Test" << endl;
    cout << "TV Budget: " << test_budget << endl;
    cout << "Predicted Sales: " << predicted_sales << endl;    
}
