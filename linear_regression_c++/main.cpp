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




int main() {
    vector<string> data = get_data();
    vector<double> tv= get_column(data,0), sales= get_column(data,1);
    //get data and converted    
    double mean_value_tv=mean(tv), mean_value_sales=mean(sales);

    double std_dev_tv=std_dev(tv,mean_value_tv),std_dev_sales=std_dev(sales,mean_value_sales);  

    vector<double> normalized_tv=normalize(tv,mean_value_tv,std_dev_tv), normalized_sales=normalize(sales,mean_value_sales,std_dev_sales);
    Parameters params = initialize_parameters(); 
    vector<double> W = params.W;                 
    double b = params.b;                         

    

    cout << "\ntotal lines: " << data.size() << endl;
}
