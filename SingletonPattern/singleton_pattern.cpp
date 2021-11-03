#include <iostream>
#include <string>
using namespace std;

class StringSingleton {
    private:
        string myString;

    public:
    // const? why?
    // reference? why?
        string getString() const{
            return myString;
        }
        void setString(const string& newStr){
            myString = newStr;
        }

        // can't forget static
        static StringSingleton& Instance(){
            // same thing: can't forget static
            // and the rule of unique_ptr
            static unique_ptr<StringSingleton> instance (new StringSingleton);
            return *instance;
        }
    
    private:
        StringSingleton(){}
        // disable copy constructor and copy assignment
        // don't forget const
        StringSingleton(const StringSingleton& old);
        const StringSingleton& operator= (const StringSingleton& old);
};

int main(){
    StringSingleton::Instance().setString("hello");
    cout << StringSingleton::Instance().getString() << endl;
}