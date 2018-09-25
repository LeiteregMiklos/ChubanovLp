#include <iostream>
#include <Eigen/Dense>
#include <math.h>
#include <vector>
#include <string>
#include <random>
#include <cstdio>
#include <ctime>
#include <list>

class LPsolver
{
    public:
        int n,m,status;
        double delta,logPa;
        long double ynorm;
        long double logynorm;
        Eigen::VectorXd x,y,ATy;
        Eigen::MatrixXd A,ATA; //let the columns of A be normalized
        Eigen::MatrixXd Aori;
        std::vector<long double> Anorms; // let this hold the real norms of A
        std::vector<long double> logAnorms;
        bool big;

        std::list<Eigen::VectorXd> dirs;

    LPsolver(Eigen::MatrixXd A)
    {
        status=0;
        this->A=A;
        Aori=A;
        m=A.rows(); n=A.cols();
        Anorms.resize(n);
        logAnorms.resize(n);
        for(int i=0;i<n;i++){Anorms[i]=A.col(i).norm();}
        for(int i=0;i<n;i++){logAnorms[i]=log(A.col(i).norm());}
        x.resize(n);
        x.setOnes();
        logPa=findPa();
        delta=finddelta();
        y=A*x;
        ynorm=y.norm();
        logynorm=log(y.norm());
        y.normalize();
        for(int i=0;i<n;i++){(this->A).col(i).normalize();}
        ATA=(this->A).transpose()*(this->A);
        ATy.resize(n);
        //for(int i=0;i<n;i++){ATy(i)=A.col(i).dot(y);}
        ATy=(this->A).transpose()*y;
        big=false;
    }

    void solve()
    {
        int t=0;
        int iter=0;

        while(logynorm>=log(delta) && logPa<=0)
        {
            iter++;
            if(!big){
                for(int i=0;i<n;i++)
                {
                    if (Anorms[i]>pow(10,150))
                    {
                        big=true;
                    }
                }
            }
            double mi=0;
            int k=0;
            for(int i=0;i<n;i++)
            {
                if(ATy(i)<mi)
                {
                    mi=ATy(i);
                    k=i;
                }
            }
            if(mi==0)
            {
                status=-1;

                Eigen::VectorXd Ty=y;
                for(auto v : dirs){Ty=(Eigen::MatrixXd::Identity(m,m)+v*v.transpose())*Ty; Ty.normalize();}
                double mi2=100;
                for(int i=0;i<m;i++)
                {
                    if(((Aori.transpose()*Ty).transpose())(i)<mi2){mi2=((Aori.transpose()*Ty).transpose())(i);}
                }
                std::cout << "dualis megoldas pontossaga: " << mi2 << std::endl;
                std::cout << "status: -1 " << "iter: " << iter << " rescale: " << t << " big: " << big << std::endl;
                return;
            }

            if(mi<-1.0/(11*m))
            {
                if(!big){
                    x=x-(mi*(ynorm/Anorms[k]))*Eigen::VectorXd::Unit(n,k);
                } else {
                    x=x-(mi*exp(logynorm-logAnorms[k]))*Eigen::VectorXd::Unit(n,k);
                }

                Eigen::VectorXd y_=y-mi*A.col(k);
                double len=y_.norm();
                ynorm*=len;
                logynorm+=log(len);
                y=y_.normalized();
                ATy=(ATy-mi*ATA.col(k))/len;
            }
            else
            {
                t++;
                A=(A+y*ATy.transpose());
                ynorm*=2;
                logynorm+=log(2);
                std::vector<double> norms(n);
                for(int i=0;i<n;i++){norms[i]=A.col(i).norm();}
                for(int i=0;i<n;i++){Anorms[i]*=norms[i];}
                for(int i=0;i<n;i++){logAnorms[i]+=log(norms[i]);}

                ATA=(ATA+3*ATy*ATy.transpose());
                for(int i=0;i<n;i++)
                {
                    for(int j=0;j<n;j++)
                    {
                        ATA(i,j)=ATA(i,j)/(norms[i]*norms[j]);
                    }
                }
                for(int i=0;i<n;i++){A.col(i).normalize();}

                ATy=2*ATy;
                for(int i=0;i<n;i++)
                {
                    ATy(i)=ATy(i)/norms[i];
                }
                //pontosítás
                if(t%(m*n)==0){
                    ATA=A.transpose()*A;
                    ATy=A.transpose()*y;
                    if(!big){
                        y.setZero();
                        for(int i=0;i<n;i++){y+=A.col(i)*Anorms[i]*x(i);}
                        ynorm=y.norm();
                        logynorm=log(ynorm);
                        y.normalize();
                    }
                }
                logPa+=log(3.0/2.0) ;

                dirs.push_front(y);
                //suggestion: keep Ty and check if A.transpose()*Ty > -e where e is tiny
            }
        }
        if(logynorm<log(delta))
        {
            status=1;
            for(int i=0;i<n;i++)
            {
                if(!big)
                {
                    A.col(i)*=Anorms[i];
                } else {
                    A.col(i)*=exp(logAnorms[i]);
                }
            }
            y*=ynorm;
            x=x-A.transpose()*(A*A.transpose()).inverse()*y;
            std::cout << "megoldas pontossaga: " << (Aori*x).norm() << std::endl;
            for(int i=0;i<n;i++)
            {
                if(x(i) < 0){std::cout << "negativ koordinata!!";}
            }
        }
        else
        {
            status=-2;
            //std::cout << y.transpose()*A <<std::endl;
            Eigen::VectorXd Ty=y;
            for(auto v : dirs){Ty=(Eigen::MatrixXd::Identity(m,m)+v*v.transpose())*Ty; Ty.normalize();}
            double mi2=100;
            for(int i=0;i<m;i++)
            {
                if(((Aori.transpose()*Ty).transpose())(i)<mi2){mi2=((Aori.transpose()*Ty).transpose())(i);}
            }
            std::cout << "dualis megoldas pontossaga: " << mi2 << std::endl;
        }
        std::cout << "status: " << status << " iter: " << iter << " rescale: " << t << " big: " << big << std::endl;
    }

    double findPa()
    {
        std::vector<double> l(n);
        for(int i=0;i<n;i++){l[i]=A.col(i).norm();}
        std::sort(l.begin(), l.end(), std::greater<int>());
        double logPa=log(pow(m,1.5));
        for(int i=0;i<m;i++)
        {
            logPa+=l[i];
        }
        logPa+=l[0];
        return -m*logPa;
    }

    double finddelta()
    {
        Eigen::MatrixXd invAAT=(A*A.transpose()).inverse();
        double mi=0;
        for(int i=0;i<n;i++)
        {
            if((invAAT*A.col(i)).norm()>mi)
            {
                mi=(invAAT*A.col(i)).norm();
            }
        }
        return 1/mi;
    }
};

void runner(int n, int m)
{
    for(int k=1;k<10;k++)
    {
        std::cout << k << std::endl << std::endl ;
        std::default_random_engine generator(k);
        std::uniform_int_distribution<int> distribution(-100,100);
        distribution(generator);
        Eigen::MatrixXd A(m,n);
        for(int i=0;i<m;i++)
        {
            for(int j=0;j<n;j++)
            {
                A(i,j)=distribution(generator);
            }
        }
        if((A*A.transpose()).determinant()==0){std::cout << "dependent rows" << std::endl;}
        std::clock_t start;
        start = std::clock();
        LPsolver L(A);
        L.solve();
        std::cout << ( std::clock() - start ) / (double) CLOCKS_PER_SEC << std::endl << std::endl;
    }
}

void runner2(int num)
{
    for(int k=1;k<=num;k++)
    {
        FILE* fp;
        std::stringstream ss;
        ss << k;
        std::string fname = "lp" + ss.str() + ".txt";
        fp = fopen(fname.c_str(),"r");
        int n,m;
        fscanf(fp,"%d %d",&m,&n);
        std::cout << k << std::endl << std::endl ;
        Eigen::MatrixXd A(m,n);
        for(int i=0;i<m;i++)
        {
            for(int j=0;j<n;j++)
            {
                int in;
                fscanf(fp,"%d",&in);
                A(i,j)=in;
            }
        }
        fclose(fp);
        if((A*A.transpose()).determinant()==0){std::cout << "dependent rows" << std::endl;}
        std::clock_t start;
        start = std::clock();
        LPsolver L(A);
        L.solve();
        std::cout << ( std::clock() - start ) / (double) CLOCKS_PER_SEC << std::endl << std::endl;
    }
}

int main()
{
    runner(10,5);
}

