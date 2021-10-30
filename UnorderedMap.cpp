#include <iostream>
#include <unordered_map>
#include <vector>
#include <stdlib.h>

using namespace std;

/* 
map vs unordered_map 

map: increasing order
unordered_map: no order

map: self balance BST like red-black tree
unordered_map: hash table

map: 
    log(n) -- search
    log(n) + Rebalance -- insert and delete
unordered_map:
    average: O(1)
    worst: O(n)
*/

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