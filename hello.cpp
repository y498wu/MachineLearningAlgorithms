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
        void attach(Observer& ob){
            obs.push_back(&ob);
        }
        void detach(Observer& ob){
            obs.erase(remove(obs.begin(), obs.end(), &ob));
        }
        void notify(){
            for(auto& ob : obs){
                ob->update();
            }
        }
};

class WindSpeed : public Subject{
    private:
        int speed;
    public:
        int getSpeed(){
            return speed;
        }
        void setSpeed(int newSpeed){
            speed = newSpeed;
            notify();
        }
};

class ControlPanel : public Observer{
    private:
        WindSpeed& ws;
    public:
        ControlPanel(WindSpeed& newWs) : ws (newWs){
            ws.attach(*this);
        }
        ~ControlPanel(){
            ws.detach(*this);
        }
        void update() override{
            cout << "Control panel: " << ws.getSpeed() << endl;
        }
};

class Kernel : public Observer{
    private:
        WindSpeed& ws;
    public:
        Kernel(WindSpeed& newWs) : ws(newWs){
            ws.attach(*this);
        }
        ~Kernel(){
            ws.detach(*this);
        }
        void update() override{
            cout << "Kernel: " << ws.getSpeed() << endl;
        }
};

int main(){
    WindSpeed ws;
    ws.setSpeed(3);
    ControlPanel cp(ws);
    Kernel kn(ws);
    ws.setSpeed(5);
    ws.setSpeed(1);
}

