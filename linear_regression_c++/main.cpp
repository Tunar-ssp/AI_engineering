#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <sstream>


using namespace std;

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

// double normalize(const vector<double>&data,double mean_value,double)



int main() {
    vector<string> data = get_data();
    vector<double> tv= get_column(data,0), sales= get_column(data,1);
    double mean_value_tv=mean(tv), mean_value_sales=mean(sales);
    double std_dev_tv=std_dev(tv,mean_value_tv),std_dev_sales=std_dev(sales,mean_value_sales);


    cout << "\ntotal lines: " << data.size() << endl;
}
