#include <iostream>
#include <Eigen/Dense>
#include <math.h>
#include <vector>
#include <string>
#include <random>
#include <cstdio>
#include <ctime>
#include <stdlib.h>
#include <sstream>

class LPsolver
{
    public:
        int n,m,t,T,status;
        Eigen::VectorXd x,xlast,y,ylast,y_,ATy;
        Eigen::MatrixXd A,Pa; //let the columns of A be normalized
        Eigen::MatrixXd Aori;
        bool step;
        double eps;

    LPsolver(Eigen::MatrixXd A)
    {
        status=0;
        this->A=A;
        Aori=A;
        m=A.rows(); n=A.cols();
        y.setConstant(n,1.0/n);
        //for(i=0;i<n;i++){y(i)=1/n;}
        y_=y;
        T=findT();
        Pa=Eigen::MatrixXd::Identity(n,n)-A.transpose()*(A*A.transpose()).inverse()*A;
        x=Pa*y;
        eps=pow(10.0,-10);
    }

    int BP()
    {
        step=false;
        while(maxy() < 2.0*eTx())
        {
            //std::cout << x.norm() << std::endl;
            int k=0;
            for(int i=0;i<n;i++)
            {
                if(x(i)<x(k)){k=i;}
            }
            if(x(k)>0){return 0;}
            double alpha=(Pa.col(k).transpose()).dot(Pa.col(k)-x)/((Pa.col(k)-x).squaredNorm());
            ylast=y; xlast=x;
            y=alpha*y+(1-alpha)*Eigen::VectorXd::Unit(n,k);
            x=alpha*x+(1-alpha)*Pa.col(k);
            step=true;

            if(x.norm()<eps){return 1;}
        }
        return 2;
    }
    int solve()
    {
        t=0;
        int scale=0;
        std::vector<double> divs1(n),divs2(n);
        for(int i=0;i<n;i++)
        {
            divs1[i]=1.0;
            divs2[i]=1.0;
        }
        int st;
        while(t<T)
        {
            t++;
            st=BP();
            if(st==0){
                Eigen::VectorXd xout=x;
                for(int i=0;i<n;i++)
                {
                    xout[i]*=divs1[i];
                }
                //return xout;
                std::cout << "primal" << std::endl;
                std::cout << (A*x).norm() << " scale " << scale << std::endl;
                return 1;
            }
            if(st==1)
            {
                Eigen::VectorXd zout=(A*A.transpose()).inverse()*A*y;
                //return yout;
                std::cout << "dual" << std::endl;
                std::cout << scale << std::endl;
                std::cout << zout.transpose()*Aori << std::endl;
                return -1;
            }
            if(st==2)
            {
                int k=0;
                for(int i=0;i<n;i++)
                {
                    if(y(i)>y(k)){k=i;}
                }
                A.col(k)=A.col(k)/2.0;
                Pa=Eigen::MatrixXd::Identity(n,n)-A.transpose()*(A*A.transpose()).inverse()*A;
                divs1[k]=divs1[k]/2.0;
                if(step)
                {
                    y_=ylast;
                    for(int i=0;i<n;i++){divs2[i]=1;}
                    //std::cout << "lep: " << x.norm() << std::endl;
                } else {
                    //std::cout << "nemlep: " << x.norm() << std::endl;
                }
                divs2[k]=divs2[k]/2;
                for(int i=0;i<n;i++){y(i)=y_(i)*divs2[i];}
                Eigen::VectorXd e;
                e.setOnes(n);
                y=y/e.dot(y);
                x=Pa*y;
                scale++;
            }
        }
        std::cout << scale << std::endl;
        return -2;
    }

    double findT()
    {
        double Lmin=1;
        for(int i=0;i<n;i++){Lmin*=A.col(i).norm();}

        return n*log2(Lmin);
    }

    double eTx()
    {
        double d=0;
        for(int i=0;i<n;i++){d+=std::max(0.0,x(i));}
        return d;
    }
    double maxy()
    {
        double ma=0;
        for(int i=0;i<n;i++)
        {
            if(y(i)>ma){ma=y(i);}
        }
        return ma;
    }

};

void solvable(Eigen::MatrixXd &A)
{
    Eigen::VectorXd x(A.cols());
    for(int i=0;i<A.cols();i++){x(i)=1.0/(i+1);}
    Eigen::VectorXd y=A*x;
    A.conservativeResize(A.rows(), A.cols()+1);
    A.col(A.cols()-1) = -y;
}

void runner()
{

    FILE *fp2;
    fp2=fopen("times.txt","w");
    FILE* fp;
    fp = fopen("out.txt","r");
    int num;
    fscanf(fp,"%d",&num);
    for(int k=1;k<=num;k++)
    {
        int n,m;
        fscanf(fp,"%d %d",&m,&n);
        fprintf(fp2,"%d\n",k);
        fprintf(fp2,"%d %d\n",m,n);
        std::cout << k << std::endl;
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
        if((A*A.transpose()).determinant()==0){std::cout << "dependent rows" << std::endl;}
        std::clock_t start;
        start = std::clock();
        solvable(A);
        LPsolver L(A);
        int stat=L.solve();
        std::cout << ( std::clock() - start ) / (double) CLOCKS_PER_SEC << std::endl << std::endl;
        fprintf(fp2,"%d ",stat);
        fprintf(fp2,"%f\n",( std::clock() - start ) / (double) CLOCKS_PER_SEC);
    }
    fclose(fp);
    fclose(fp2);
}

void writer(int m, int n, int mi, int ma, int num)
{
    FILE* fp;
    fp = fopen("out.txt","w");
    fprintf(fp,"%d\n",num);
    for(int k=1;k<=num;k++)
    {
        std::default_random_engine generator(k);
        std::uniform_int_distribution<int> distribution(mi,ma);
        distribution(generator);
        fprintf(fp,"%d %d\n",m,n);
        for(int i=0;i<m;i++)
        {
            for(int j=0;j<n;j++)
            {
                fprintf(fp,"%d ",distribution(generator));
            }
            fprintf(fp,"\n");
        }
    }
    fclose(fp);
}

int main()
{

    writer(5,10,-100,100,10);
    runner();
}

