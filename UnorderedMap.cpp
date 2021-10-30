#include <iostream>
#include <unordered_map>
#include <vector>
#include <stdlib.h>

using namespace std;

void uniqueNum(vector<int>& lst){
    unordered_map<int, int> lib;
    for(auto num : lst){
        lib[num]++;
    }
    for(auto it : lib){
        cout << it.first << " ";
    }
    cout << endl;
}

int main(){
    vector<int> lst;
    int rd = 0;
    srand(time(NULL));
    for(int i = 0; i < 10; i++){
        rd = rand() % 10;
        cout << rd << " ";
        lst.push_back(rd);
    }
    cout << endl;
    uniqueNum(lst);
}