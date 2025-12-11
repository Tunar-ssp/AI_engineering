#include <iostream>
#include <string>
#include <vector>
#include <fstream>

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

int main() {
    vector<string> data = get_data();

    cout << "\ntotal lines: " << data.size() << endl;
}
