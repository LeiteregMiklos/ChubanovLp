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

#include <thread>

class LPsolver
{
    public:
        int t,T,state,mode;
        Eigen::VectorXd x,xlast,y,ylast,y_,ATy,b,oricol,divs,numdivs,xout;
        Eigen::MatrixXd A,Pa; //let the columns of A be normalized
        Eigen::MatrixXd Aori;
        bool step;
        bool finished,verbose;
        double eps;//inaccuracy

    LPsolver(Eigen::MatrixXd A,Eigen::VectorXd b,int mode) //mode 1:modify b to guarantee full supp solution
    {
        this->mode=mode;
        finished=false; verbose=false;
        Aori=A;
        if(mode==1)
        {
            eps=findEps(A);
            Eigen::VectorXd v,e;
            e.setConstant(A.cols(),1.0);
            v=A*e;
            b=b+v*eps;
        }
        A.conservativeResize(A.rows(), A.cols()+1);
        A.col(A.cols()-1) = -b;
        this->A=A;
        removeDependentRows();
        A=this->A;
        this->b=b;

        //initialize the algortihm
        y.setConstant(A.cols(),1.0/A.cols());
        y_=y;
        t=findt();
        //Pa=Eigen::MatrixXd::Identity(A.cols(),A.cols())-A.transpose()*(A*A.transpose()).inverse()*A;
        Pa=findPa();
        x=Pa*y;
        //inaccuracy=pow(10.0,-10);
        oricol.resize(A.cols());
        for(int i=0;i<A.cols();i++)
        {
            oricol(i)=i;
        }
        divs.resize(A.cols());
        numdivs.resize(A.cols());
        for(int i=0;i<A.cols();i++)
        {
            divs[i]=1.0;
            numdivs[i]=0;
        }
    }

    int BP() //-1: infinite loop 0:rescale 1:solution 2:dual solution
    {
        step=false;
        while(maxy() < 2.0*eTx())
        {
            //std::cout << x.norm() << std::endl;
            int k=0;
            for(int i=0;i<A.cols();i++)
            {
                if(x(i)<x(k)){k=i;}
            }
            if(x(k)>0){return 1;}
            //std::cout <<  std::endl << x(k) << std::endl;
            double alpha=(Pa.col(k).transpose()).dot(Pa.col(k)-x)/((Pa.col(k)-x).squaredNorm());
            ylast=y; xlast=x;
            alpha=std::min(alpha,1.0); //unnecessary
            y=alpha*y+(1-alpha)*Eigen::VectorXd::Unit(A.cols(),k);
            x=alpha*x+(1-alpha)*Pa.col(k);
            step=true;

            if (alpha>=1)
            {
                //x=Pa*y ;
                /*std::cout << std::endl << std::endl << (Pa*x-x).norm() << std::endl; //should be zero
                std::cout << Pa.col(k).dot(x) << std::endl;
                std::cout << x(k) << std::endl;
                std::cout << (Pa.col(k)-x).dot(x) << std::endl;
                std::cout << (A*x).norm() << std::endl;*/
                return -1;
            }

            //if(x.norm()<inaccuracy){return 2;}
        }
        return 0;
    }
    int solve(bool &done) //-1=infinite loop 0:x_n=0 1:primal 2:dual 3:infis but no dual 4:did no finish
    {
        int scale=0;

        int st;
        while(!done && A.cols()>0)
        {
            st=BP();
            if(st==-1){
                if(verbose){std::cout << "inf loop" << std::endl; }
                state=-1;
                return -1;
            }
            if(st==1){
                xout.setZero(Aori.cols());
                if(oricol(A.cols()-1)==Aori.cols())
                {
                    for(int i=0;i<A.cols()-1;i++)
                    {
                        xout[oricol(i)]=(x(i)*divs[i])/(x(A.cols()-1)*divs[A.cols()-1]);
                    }
                    //return xout;
                    if(verbose){std::cout << "primal" << std::endl;
                    std::cout << "||A*x-b||: "<< (Aori*xout-b).norm() << " scale " << scale << std::endl;} //".." << (A*x).norm() <<
                    //Eigen::MatrixXd B=findFeasibleBase(Aori,xout);
                    //Eigen::FullPivLU<Eigen::MatrixXd> dec(B);
                    //std::cout << dec.solve(b) << std::endl;
                    int k=0;
                    for(int i=0;i<A.cols();i++)
                    {
                        if(x(i)<x(k)){k=i;}
                    }
                    if(x(k)<0){ std::cout << "x_min= " << x(k) << std::endl; }
                    state=1;
                    done=true;
                    finished=true;
                    return 1;
                } else {if(verbose){std::cout << "infeas2" << std::endl;}}
                done=true;
                finished=true;
                state=0;
                return 0;
            }
            /*if(st==2)
            {
                Eigen::VectorXd zout=(A*A.transpose()).inverse()*A*y;
                //return yout;
                std::cout << "dual" << std::endl;
                std::cout << scale << std::endl;
                std::cout << zout.transpose()*Aori << std::endl;
                return 2;
            }*/
            if(st==0)
            {
                int k=0;
                for(int i=0;i<A.cols();i++)
                {
                    if(y(i)>y(k)){k=i;}
                }
                A.col(k)=A.col(k)/2.0;
                //Pa=Eigen::MatrixXd::Identity(A.cols(),A.cols())-A.transpose()*(A*A.transpose()).inverse()*A; //optimize
                Pa=findPa();
                divs[k]=divs[k]/2.0;
                numdivs[k]++;
                if(step)
                {
                    y_=ylast;
                    //std::cout << "lep: " << x.norm() << std::endl;
                } else {
                    //std::cout << "nemlep: " << x.norm() << std::endl;
                }
                y_(k)=y_(k)/2;
                Eigen::VectorXd e; e.setOnes(A.cols());
                y=y_/e.dot(y_);
                x=Pa*y;
                scale++;

                if(numdivs[k]>t && mode!=1)
                {
                    int n=A.cols();
                    A.block(0,k,A.rows(),A.cols()-1-k) = A.rightCols(A.cols()-1-k);
                    A.conservativeResize(A.rows(),A.cols()-1);
                    removeDependentRows();
                    Pa=findPa();
                    y.segment(k,n-k-1)=y.tail(n-k-1);
                    y.conservativeResize(n-1);

                    Eigen::VectorXd e; e.setOnes(A.cols());
                    y=y/(e.dot(y));
                    x=Pa*y;
                    y_=y;

                    oricol.segment(k,n-k-1)=oricol.tail(n-k-1);
                    oricol.conservativeResize(n-1);
                    //std::cout << oricol.transpose() << std::endl;

                    divs.segment(k,n-k-1)=oricol.tail(n-k-1);
                    divs.conservativeResize(n-1);

                    numdivs.segment(k,n-k-1)=oricol.tail(n-k-1);
                    numdivs.conservativeResize(n-1);
                }
            }
        }
        if(done!=true){
            if(verbose){std::cout << "infeas" << std::endl << "scale: " << scale << std::endl;}
            done=true;
            finished=true;
            state=3;
            return 3;
        }
        state=4;
        return 4;
    }
    Eigen::MatrixXd findPa()
    {
        Eigen::MatrixXd P=Eigen::MatrixXd::Identity(A.cols(),A.cols());
        Eigen::FullPivLU<Eigen::MatrixXd> dec(A*A.transpose());
        for(int i=0;i<A.cols();i++)
        {
            P.col(i)=P.col(i)-A.transpose()*dec.solve(A*Eigen::VectorXd::Unit(A.cols(),i));
        }
        return P;
    }
    void removeDependentRows()
    {
        Eigen::MatrixXd At=A.transpose();
        Eigen::MatrixXd Im=At.fullPivLu().image(At);

        if (Im.cols()<At.cols())
        {
            A=Im.transpose();
        }
    }
    Eigen::MatrixXd findFeasibleBase(Eigen::MatrixXd A, Eigen::VectorXd b) //assuming A*x=b //not working
    {
        Eigen::FullPivLU<Eigen::MatrixXd> dec(A);
        Eigen::MatrixXd l = Eigen::MatrixXd::Identity(A.rows(),A.rows());
        l.block(0,0,A.rows(),A.rows()).triangularView<Eigen::StrictlyLower>() =
        dec.matrixLU().block(0,0,A.rows(),A.rows()).triangularView<Eigen::StrictlyLower>();
        std::cout << l.inverse()*A << std::endl;
        return A;
    }
    double findt()
    {
        std::vector<double> l(A.cols());
        for(int i=0;i<A.cols();i++){l[i]=A.col(i).norm();}
        std::sort(l.begin(), l.end(), std::greater<int>());
        double detB=1;
        for(int i=0;i<A.rows();i++)
        {
            detB*=l[i];
        }
        return log2(detB);
    }
    double findEps(const Eigen::MatrixXd& A)
    {
        std::vector<double> l(A.cols());
        for(int i=0;i<A.cols();i++){l[i]=A.col(i).norm();}
        std::sort(l.begin(), l.end(), std::greater<int>());
        double detB=1;
        for(int i=0;i<A.rows();i++)
        {
            detB*=l[i];
        }
        Eigen::VectorXd v,e;
        e.setConstant(A.cols(),1.0);
        v=A*e;
        return 1.0/(detB*v.lpNorm<1>());
    }
    double eTx() //sum of positive cordinates of x
    {
        double d=0;
        for(int i=0;i<A.cols();i++){d+=std::max(0.0,x(i));}
        return d;
    }
    double maxy() //max coordinate of y
    {
        double ma=0;
        for(int i=0;i<A.cols();i++)
        {
            if(y(i)>ma){ma=y(i);}
        }
        return ma;
    }
};

void optiMatrix(Eigen::MatrixXd &A, Eigen::VectorXd &b, Eigen::VectorXd c, double val)
{
    Eigen::MatrixXd newA;
    Eigen::VectorXd zeros;
    zeros.setConstant(A.rows(),0.0);
    newA.resize(A.rows()+1,A.cols()+1);
    newA << A , zeros,
            c.transpose() , -1;
    A=newA;
    b.conservativeResize(b.size()+1);
    b(b.size()-1)=val;
}

void dual(const Eigen::MatrixXd &A, const Eigen::VectorXd &b,Eigen::MatrixXd &dA ,Eigen::VectorXd &db);

bool solvable(const Eigen::MatrixXd &A,const Eigen::VectorXd &b,Eigen::VectorXd &x)
{
    Eigen::MatrixXd dA;
    Eigen::VectorXd db;

    dual(A,b,dA,db);

    LPsolver L(A,b,1);
    LPsolver L2(dA,db,1);
    bool done=false;
    std::thread first (&LPsolver::solve,&L,std::ref(done));
    L2.solve(done);
    first.join();
    if((L.finished && L.state==1) || (L2.finished && L2.state!=1)){x=L.xout; return true;} else {x=L2.x; return false;}
}

double optimize(const Eigen::MatrixXd &A,const Eigen::VectorXd &b,const Eigen::VectorXd& c)
{
    Eigen::VectorXd x;
    if(!solvable(A,b,x))
    {
        std::cout << "not solvable" << std::endl;
        return -100000;
    }
    double val1=c.transpose()*x;
    double val2=1000;
    Eigen::MatrixXd A0=A;
    Eigen::VectorXd b0=b;

    optiMatrix(A0,b0,c,val1);
    /*bool bsolv=solvable(A0,b0,x);
    while (!bsolv)
    {
        val1=val1*10;
        b0(b0.size()-1)=val1;
        bsolv=solvable(A0,b0,x);
    }*/
    b0(b0.size()-1)=val2;
    bool bsolv=solvable(A0,b0,x);
    while(bsolv)
    {
        val2=val2*10;
        b0(b0.size()-1)=val2;
        bsolv=solvable(A0,b0,x);
        if(val2>10000000){ std::cout << "likely infinite" << std::endl;
                return 10000000;}
    }
    while(val2-val1>0.01)
    {
        b0(b0.size()-1)=(val1+val2)/2.0;
        if(solvable(A0,b0,x)) { val1=(val1+val2)/2.0;} else {val2=(val1+val2)/2.0;}
    }
    x.conservativeResize(x.size()-1);
    std::cout << "||Ax-b||: " << (A*x-b).norm() << std::endl;
    std::cout << c.transpose()*x << std::endl;
    return val1;

}

void makesolvable(Eigen::MatrixXd &A)
{
    Eigen::VectorXd x(A.cols());
    for(int i=0;i<A.cols();i++){x(i)=1.0/(i+1);}
    Eigen::VectorXd y=A*x;
    A.conservativeResize(A.rows(), A.cols()+1);
    A.col(A.cols()-1) = -y;
}

void dual(const Eigen::MatrixXd &A, const Eigen::VectorXd &b,Eigen::MatrixXd &dA ,Eigen::VectorXd &db)
{
    dA.resize(A.cols()+1,A.rows()*2+A.cols()+1);
    Eigen::MatrixXd I=Eigen::MatrixXd::Identity(A.cols(),A.cols());
    Eigen::VectorXd zeros,zeros2;
    zeros.setConstant(A.cols(),0.0);
    zeros2.setConstant(A.cols(),0.0);
    dA << A.transpose(), -A.transpose(), -I*1000, zeros,
          b.transpose(), -b.transpose(), zeros2.transpose(), 1*1000;

    db.setConstant(A.cols()+1,0);
    db(A.cols())=-1;
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
        Eigen::MatrixXd A(m,n-1);
        Eigen::VectorXd b(m);
        for(int i=0;i<m;i++)
        {
            for(int j=0;j<n-1;j++)
            {
                int in;
                fscanf(fp,"%d",&in);
                A(i,j)=in;
            }
            int in;
            fscanf(fp,"%d",&in);
            b(i)=in;
        }
        /*A.conservativeResize(A.rows()+1,A.cols());
        A.row(A.rows()-1).setZero();
        b.setZero();
        b.conservativeResize(b.size()+1);
        b(b.size()-1)=1;
        std::cout << A << std::endl << b << std::endl;*/

        //optimize(A,b,c);
        if(false){
            Eigen::MatrixXd dA;
            Eigen::VectorXd db;

            dual(A,b,dA,db);

            LPsolver L(A,b,0);
            LPsolver L2(dA,db,0);
            bool done=false;
            std::thread first (&LPsolver::solve,&L,std::ref(done));
            L2.solve(done);
            first.join();
            if(L.finished){std::cout << "primal" << std::endl;} else {std::cout << "dual" << std::endl;}
        }
        if(true){
            Eigen::VectorXd c;
            c.setConstant(A.cols(),-1.0);
            std::cout << "first" << std::endl;
            std::clock_t start;
            std::cout << "opt: " << optimize(A,b,c) << std::endl;
            fprintf(fp2,"%f\n",( std::clock() - start ) / (double) CLOCKS_PER_SEC);
        }
        if(false){
            Eigen::MatrixXd dA;
            Eigen::VectorXd db;

            dual(A,b,dA,db);

            LPsolver L(A,b,1);
            LPsolver L2(dA,db,1);

            bool done=false;
            std::cout << "first" << std::endl;
            std::clock_t start;
            start = std::clock();
            L.solve(done);
            std::cout << "time: " << ( std::clock() - start ) / (double) CLOCKS_PER_SEC << std::endl << std::endl;
            //fprintf(fp2,"%d ",stat);

            std::cout << "second" << std::endl;
done=false;
            start = std::clock();
            L2.solve(done);
            std::cout << "time: " << ( std::clock() - start ) / (double) CLOCKS_PER_SEC << std::endl << std::endl;
        }




        //std::cout << ( std::clock() - start ) / (double) CLOCKS_PER_SEC << std::endl << std::endl;
        //fprintf(fp2,"%d ",stat);
        //fprintf(fp2,"%f\n",( std::clock() - start ) / (double) CLOCKS_PER_SEC);
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

void writer2(int m, int n, int mi, int ma, int num) //non max support solution
{
    FILE* fp;
    fp = fopen("out2.txt","w");
    fprintf(fp,"%d\n",num);
    for(int k=1;k<=num;k++)
    {
        std::default_random_engine generator(k);
        std::uniform_int_distribution<int> distribution(mi,ma);
        distribution(generator);
        Eigen::MatrixXd A(m,n);
        fprintf(fp,"%d %d\n",m,n);
        for(int i=0;i<m;i++)
        {
            for(int j=0;j<n;j++)
            {
                A(i,j)=distribution(generator);
            }
        }
        for(int j=0;j<n;j++)
        {
            A(j%m,j)=0;
        }
        Eigen::VectorXd e; e.setConstant(m,1);
        for(int j=0;j<n;j++)
        {
            A(j%m,j)=-e.dot(A.col(j));
        }
        A.col(n-1)=A.col(n-1);//+e;
        for(int i=0;i<m;i++)
        {
            for(int j=0;j<n;j++)
            {
                int k=A(i,j); //dont know why this is necessary
                fprintf(fp,"%d ",k); //A(i,j) here wrote zeros
            }
            fprintf(fp,"\n");
        }
    }
    fclose(fp);
}

int main()
{

    writer(15,30,-100,100,10);
    runner();
}


