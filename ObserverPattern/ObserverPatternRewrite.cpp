#include <iostream>
#include <vector>
using namespace std;

class Observer{
    public:
        virtual void update() = 0;
};

class Subject{
    private:
        vector<Observer*> obs;
    public:
        void attach(Observer* ob){
            obs.push_back(ob);
        }
        void detach(Observer* ob){
            obs.erase(remove(obs.begin(), obs.end(), ob));
        }
        void notify(){
            for(auto* ob : obs){
                ob->update();
            }
        }
};

class WindSpeed : public Subject{
    private:
        int ws;
    public:
        void setSpeed(int s){
            ws = s;
            notify();
        }
        int getSpeed(){
            return ws;
        }      
};

class ControlPanel : public Observer{
    private:
        WindSpeed& ws;
    public:
        ControlPanel(WindSpeed& ws) : ws(ws){
            ws.attach(this);
        }
        ~ControlPanel(){
            ws.detach(this);
        }
        void update() override{
            cout << "Control Panel: " << ws.getSpeed() << endl;
        }
};

class Kernel : public Observer{
    private:
        WindSpeed& ws;
    public:
        Kernel(WindSpeed& ws) : ws(ws){
            ws.attach(this);
        }
        ~Kernel(){
            ws.detach(this);
        }
        void update() override{
            cout << "Kernel: " << ws.getSpeed() << endl;
        }
};

int main(){
    WindSpeed ws;
    ws.setSpeed(10);
    ControlPanel cp(ws);
    Kernel k(ws);
    ws.setSpeed(5);
    ws.setSpeed(1);
}