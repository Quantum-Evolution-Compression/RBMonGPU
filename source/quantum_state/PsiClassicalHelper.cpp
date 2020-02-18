#include "quantum_state/PsiClassicalHelper.hpp"
#include "quantum_state/PsiDeepMin.hpp"

#include <fstream>
#include <cstdlib>
#include <complex>
#include <vector>
#include <Eigen/Core> // Matrix and Array classes, basic linear algebra (including triangular and selfadjoint products), array manipulation
#include <iostream>

using namespace std;

namespace Peter {

int L=10;
int H=1;
int const numberOfVarParameters=4;
int PerturbationTheoryOrder = 1; 

#define cdouble std::complex<double>
cdouble I(0.0,1.0);
vector< vector< vector<int> > > S;     // i,j -- coordinates of the plaquette, only k=2 is used
int const numberOfVarParametrsMax=300000; // before compression Stripe3Order (12321), Square 3Order (12160) // debug:
int indexVP[numberOfVarParametrsMax][2] = {0}; // initialize with zeros


Eigen::VectorXcd varW(numberOfVarParameters); // variational parameters

rbm_on_gpu::PsiDeepMin* psi_neural = nullptr;

void load_neural_network(string directory, int index) {
    if(psi_neural) {
        delete psi_neural;
    }
    psi_neural = new rbm_on_gpu::PsiDeepMin(directory + "/psi_" + to_string(index-1) + "_compressed.txt");
}


/*
std::complex<double> Classical_wavefunction(std::vector<int> &S, std::vector<std::complex<double> > &varW, int numberOfVarParameters)
    {
    std::complex<double> varW0 = varW[numberOfVarParameters];
    std::complex<double> Heff_plaquetteComplex = 0;

    int omega;
    for (int j=0; j<L; j++)
        {
        omega =   1*(S[j]*S[(j+1)%L]+1)/2 +        2*(S[(j+1)%L]*S[(j+2)%L]+1)/2 +  4*(S[(j+2)%L]*S[(j+3)%L]+1)/2 +   8*(S[(j+3)%L]*S[(j+4)%L]+1)/2
			   + 16*(S[(j+4)%L]*S[(j+5)%L]+1)/2 + 32*(S[(j+5)%L]*S[(j+6)%L]+1)/2 + 64*(S[(j+6)%L]*S[(j+7)%L]+1)/2 ;
		Heff_plaquetteComplex += (-I)*varW[omega];
        }

    return exp(varW0+Heff_plaquetteComplex);
    }
*/

void loadVP(std::string directory, int index, std::string ReIm) // two calls are necessary: LoadVP("Re",..,..); LoadVP("Im",..,..);
    {
    std::string filenamePos = directory + "/a_VP_" + to_string(index) + "_" + ReIm + ".csv";
	std::ifstream filePos;
	filePos.open (filenamePos.c_str());
    
    if (filePos.is_open()==true)  cout << filenamePos << " was successfully loaded by loadVP()" << endl;
    if (filePos.is_open()==false) cout << filenamePos << " was NOT loaded by loadVP()" << endl;
    
	std::string temp;

    for (int i=0; i<numberOfVarParameters; i++)
        {
        getline (filePos, temp);
        if (ReIm.find("Re") != std::string::npos) varW(i) +=   atof(temp.c_str());
        if (ReIm.find("Im") != std::string::npos) varW(i) += I*atof(temp.c_str());
        }
    getline (filePos, temp);
    if (ReIm.find("Re") != std::string::npos) varW(numberOfVarParameters) +=   atof(temp.c_str()); // "dumb" variational parameter for normalization
    if (ReIm.find("Im") != std::string::npos) varW(numberOfVarParameters) += I*atof(temp.c_str());


    filePos.close();
	}



void Compress_Load(std::string directory, int index)
    {

    string filenamePos = directory + "/a_indexVP_" + to_string(index) + ".csv";
	ifstream filePos;
	filePos.open (filenamePos.c_str());

    if (filePos.is_open()==true)  cout << filenamePos << " was successfully loaded by Compress_Load()" << endl;
    if (filePos.is_open()==false) cout << filenamePos << " was NOT loaded by Compress_Load()" << endl;
    
    int indexCompressed=1;
    string temp;
    for (int i=0; i<numberOfVarParametrsMax; i++)
        {
        getline (filePos, temp);
        indexVP[i][0] = atoi(temp.c_str());;
        if (indexVP[i][0]!=0) indexCompressed++;
        }

    cout << "Number of decompressed variational parameters: " << indexCompressed << endl;
    filePos.close();
    }

cdouble psi_0_local(int i, int j, int fl)
    {
    cdouble psi_0_local_temp=1.0;

    vector<int> spins(L);
    for (int j=0; j<L; j++) spins[j] = S[0][j][2];
    psi_0_local_temp *= exp(psi_neural->log_psi_s(spins));
    return psi_0_local_temp;
    }

int FindOmega(int i, int j) //
	{
    int Omega = (S[i][(j+1)%L][2]+S[i][(j-1+L)%L][2])/2;  // = -1,0,1
    return Omega;
    }

void FlipPlaquette(int i, int j)
	{
    S[i][j][2] *= -1;
    }

int if_Flippable(int i, int j)
	{
    return S[i][j][2];
	}

cdouble Heff_plaquetteComplex(int i, int j, Eigen::VectorXcd& varW) // doesn't take into account the factor of 2
	{
	//for (int x=0; x<L; x++) S[0][x][2] = S_1D[x];

	int CompressionMode=0;

	int fl = if_Flippable(i, j); // flippability, +1,0,-1


    int Omega = FindOmega (i, j);

	string VariationalModeString="Ising";

    if (VariationalModeString=="Ising")
        {
        cdouble Heff_plaquetteComplex=0.0;
        int n1OrderVP, n2OrderVP, nPlaqSymClassTotal;
        int omega = fl*Omega;

        int j_ind=0; // j-index for non-uniform states
        int j_ind_max=1; // for compression
        if (0)//(TildeRun==1 || InitialState!=0)
            {
            j_ind=abs(j-L/2);
            j_ind_max = L/2+1;
            }

        // 0th order. 0-th parameter for "misses"
        double Es = -S[0][j][2]*(S[0][(j+1)%L][2]+S[0][(j-1+L)%L][2])/2;
        Heff_plaquetteComplex += (-I)*Es*varW(indexVP[1][0]);
        if (CompressionMode==1) for (int j_indt=0; j_indt<j_ind_max; j_indt++) indexVP[1][1]+=1;
        if (PerturbationTheoryOrder==0) return Heff_plaquetteComplex;

        cdouble psi_0_local_ij= psi_0_local(i,j, fl);

        // 1-order: flip one (this) spin
            {
            FlipPlaquette(i,j); // flip
            cdouble psi_0_local_ij_flip = psi_0_local(i,j, -fl);
            FlipPlaquette(i,j); // flip back
            Heff_plaquetteComplex += (-I)*psi_0_local_ij_flip/psi_0_local_ij*varW(indexVP[2+(1+omega)][0]); // 9 first-ordetr VP; +1 -- for all other variational parameters
            if (CompressionMode==1) for (int j_indt=0; j_indt<j_ind_max; j_indt++)    indexVP[2+(1+omega)][1]+=1;
            // total 1-ord: 2+3 = 5 (0-th is dumb)

            if (PerturbationTheoryOrder==1) return Heff_plaquetteComplex;
            }

        // 2-nd order
            {
            n1OrderVP = 5;                                                                                                                                      //      A
            vector<int> i1List, j1List; // neighb plaquette #1. total 12 possibilities, divided in symmetry classes depending on the symmetry of the state      //    8 9 B
            int nPlaqTotal = 2; // number of spins                                                                                                                                //  7 6 x 0 1    A=10, B=11
            i1List = {i        , i        };      //    5 3 2
            j1List = {(j+1+L)%L, (j-1+L)%L};      //      4

            vector<int> SymClassList; // symmetry classes for all 2 spins
            //int nPlaqSymClassTotal; // total number of symmetry classes

            cdouble Psi;

            // symmetry classes
            // 1. with L-R symmetry
            if (1)
                {
                SymClassList = {0,0}; // both spins are in the same symmetry class
                nPlaqSymClassTotal = 1;
                }

            int i1,j1,fl1;
            for (int nPlaq=0; nPlaq < nPlaqTotal; nPlaq+=1) // loop over all 12 plaquettes
                {
                i1  = i1List[nPlaq];
                j1  = j1List[nPlaq];

                // find the symmetry class of this plaquette
                int nPlaqSymClass = SymClassList[nPlaq];

                // 1. main contribution <V(t')V(t'')>
                Psi = 1.0;
                Psi /= psi_0_local_ij;
                FlipPlaquette(i, j); // flip plaquette
                Psi *= psi_0_local(i,j, -fl);

                fl1 = if_Flippable(i1, j1);
                if (fl1!=0)
                    {
                    int omega1 = fl1*FindOmega(i1,j1);

                    // find psi-factors
                    Psi /= psi_0_local(i1,j1,fl1);
                    FlipPlaquette(i1,j1); // flip plaquette-1
                    Psi *= psi_0_local(i1,j1,-fl1);
                    FlipPlaquette(i1,j1); // flip plaquette-1 back

                    // omega, omega1
                    Heff_plaquetteComplex += (-I)*Psi*varW(indexVP[n1OrderVP + 3*2*(1+omega) + 2*(1+omega1) + 0][0]);
                    if (CompressionMode==1) for (int j_indt=0; j_indt<j_ind_max; j_indt++)  indexVP[n1OrderVP + 3*2*(1+omega) + 2*(1+omega1) + 0][1]+=1;
                    }
                FlipPlaquette(i, j);   // flip plaquette back

                // 2. contribution <V(t')><V(t'')>
                Psi = 1.0;
                Psi /= psi_0_local_ij;
                FlipPlaquette(i, j); // flip plaquette
                Psi *= psi_0_local(i,j, -fl);
                FlipPlaquette(i, j);   // flip plaquette back

                fl1 = if_Flippable(i1, j1);
                if (fl1!=0)
                    {
                    int omega1 = fl1*FindOmega(i1,j1);

                    // find psi-factors
                    Psi /= psi_0_local(i1,j1,fl1);
                    FlipPlaquette(i1,j1); // flip plaquette-1
                    Psi *= psi_0_local(i1,j1,-fl1);
                    FlipPlaquette(i1,j1); // flip plaquette-1 back

                    // omega, omega1
                    Heff_plaquetteComplex += (-I)*Psi*varW(indexVP[n1OrderVP + 3*2*(1+omega) + 2*(1+omega1) + 1][0]);
                    if (CompressionMode==1)  for (int j_indt=0; j_indt<j_ind_max; j_indt++) indexVP[n1OrderVP + 3*2*(1+omega) + 2*(1+omega1) + 1][1]+=1;
                    // total 2-ord: 3*3*2=18
                    }
                }

            if (PerturbationTheoryOrder==2) return Heff_plaquetteComplex;
            }

        // 3-rd order, partial
        n2OrderVP = n1OrderVP+3*3*2;
        vector<vector<int>> Path; // Path[nPath][coord], where coord = i1,j1,i2,j2
        //cout << "omega="<< omega << endl;
        for (int dir=1; dir>=-1; dir-=2)
            {
            Path.clear();

            Path.push_back({i, (j+1*dir+  L)%L, i, (j+2*dir+2*L)%L}); // 0-R1-R2 (0 is default)
            Path.push_back({i, (j+1*dir+  L)%L, i, (j          )%L}); // 0-R1-0
            Path.push_back({i, (j+1*dir+  L)%L, i, (j-1*dir+  L)%L}); // 0-R1-L1
            Path.push_back({i, (j+2*dir+2*L)%L, i, (j+1*dir+  L)%L}); // 0-R2-R1

            int nPathTotal = Path.size();

            int nPathDiff = nPathTotal; // distinguish all the paths // uncommented out 11.12.2019 for tests // DEBUG
            int nTerms=5; // number of different terms (contributions)

            cdouble Psi;

            for (int nPath=0; nPath < nPathTotal; nPath+=1)
                {

                // 0. main contribution <V(t')V(t'')V(t''')>
                Psi = 1.0;
                Psi /= psi_0_local_ij;
                FlipPlaquette(i, j); // flip plaquette
                Psi *= psi_0_local(i,j, -fl);
                // FlipPlaquette(i, j); // flip plaquette back

                int i1 = Path[nPath][0];
                int j1 = Path[nPath][1];
                int fl1 = if_Flippable(i1, j1);
                if (fl1!=0)
                    {
                    int omega1 = fl1*FindOmega(i1,j1);

                    Psi /= psi_0_local(i1,j1,fl1);
                    FlipPlaquette(i1,j1); // flip plaquette-1
                    Psi *= psi_0_local(i1,j1,-fl1);
                    //FlipPlaquette(i1,j1); // flip plaquette-1

                    int i2  = Path[nPath][2];
                    int j2  = Path[nPath][3];

                    int fl2 = if_Flippable(i2, j2);
                    if (fl2!=0)
                        {
                        int omega2 = fl2*FindOmega(i2,j2);

                        Psi /= psi_0_local(i2,j2,fl2);
                        FlipPlaquette(i2,j2); // flip plaquette-2
                        Psi *= psi_0_local(i2,j2,-fl2);
                        FlipPlaquette(i2,j2); // flip plaquette-2 back

                        // omega, omega1, omega2
                        Heff_plaquetteComplex +=  (-I)*Psi*varW(indexVP[n2OrderVP + 3*3*nTerms*nPathDiff*(1+omega) + 3*nTerms*nPathDiff*(1+omega1) + nTerms*nPathDiff*(1+omega2) + nTerms*(nPath%nPathDiff) + 0][0]);
                        if (CompressionMode==1)  for (int j_indt=0; j_indt<j_ind_max; j_indt++) indexVP[n2OrderVP + 3*3*nTerms*nPathDiff*(1+omega) + 3*nTerms*nPathDiff*(1+omega1) + nTerms*nPathDiff*(1+omega2) + nTerms*(nPath%nPathDiff) + 0][1]+=1;
                        //if (CompressionMode==1)  for (int j_indt=0; j_indt<j_ind_max; j_indt++)  indexVP[n2OrderVP + 9*9*9*nTerms*nPathDiff*j_indt + 9*9*nTerms*nPathDiff*(4+omega) + 9*nTerms*nPathDiff*(4+omega1) + nTerms*nPathDiff*(4+omega2) + nTerms*(nPath%nPathDiff) + 0][1]+=1;
                        }

                    FlipPlaquette(i1, j1);   // flip plaquette-1 back
                    }
                FlipPlaquette(i, j);   // flip plaquette back
                //--

                // 1. contribution <V(t')><V(t'')V(t''')>
                Psi = 1.0;
                Psi /= psi_0_local_ij;
                FlipPlaquette(i, j); // flip plaquette
                Psi *= psi_0_local(i,j, -fl);
                FlipPlaquette(i, j); // flip plaquette back

                i1 = Path[nPath][0];
                j1 = Path[nPath][1];
                fl1 = if_Flippable(i1, j1);
                if (fl1!=0)
                    {
                    int omega1 = fl1*FindOmega(i1,j1);

                    Psi /= psi_0_local(i1,j1,fl1);
                    FlipPlaquette(i1,j1); // flip plaquette-1
                    Psi *= psi_0_local(i1,j1,-fl1);
                    //FlipPlaquette(i1,j1); // flip plaquette-1

                    int i2  = Path[nPath][2];
                    int j2  = Path[nPath][3];

                    int fl2 = if_Flippable(i2, j2);
                    if (fl2!=0)
                        {
                        int omega2 = fl2*FindOmega(i2,j2);

                        Psi /= psi_0_local(i2,j2,fl2);
                        FlipPlaquette(i2,j2); // flip plaquette-2
                        Psi *= psi_0_local(i2,j2,-fl2);
                        FlipPlaquette(i2,j2); // flip plaquette-2 back

                        // omega, omega1, omega2
                        Heff_plaquetteComplex +=  (-I)*Psi*varW(indexVP[n2OrderVP + 3*3*nTerms*nPathDiff*(1+omega) + 3*nTerms*nPathDiff*(1+omega1) + nTerms*nPathDiff*(1+omega2) + nTerms*(nPath%nPathDiff) + 1][0]);
                        if (CompressionMode==1)  for (int j_indt=0; j_indt<j_ind_max; j_indt++) indexVP[n2OrderVP + 3*3*nTerms*nPathDiff*(1+omega) + 3*nTerms*nPathDiff*(1+omega1) + nTerms*nPathDiff*(1+omega2) + nTerms*(nPath%nPathDiff) + 1][1]+=1;
                        }

                    FlipPlaquette(i1, j1);   // flip plaquette-1 back
                    }
                //FlipPlaquette(i, j);   // flip plaquette back
                //--

                // 2. contribution <V(t'')><V(t')V(t''')>: omega1,omega,omega2

                i1 = Path[nPath][0];
                j1 = Path[nPath][1];
                fl1 = if_Flippable(i1, j1);
                if (fl1!=0)
                    {
                    int omega1 = fl1*FindOmega(i1,j1);

                    Psi = 1.0;
                    Psi /= psi_0_local(i1,j1,fl1);
                    FlipPlaquette(i1,j1); // flip plaquette-1
                    Psi *= psi_0_local(i1,j1,-fl1);
                    FlipPlaquette(i1,j1); // flip plaquette-1 back


                    Psi /= psi_0_local_ij;
                    FlipPlaquette(i, j); // flip plaquette
                    Psi *= psi_0_local(i,j, -fl);
                    //FlipPlaquette(i, j); // flip plaquette back

                    int i2  = Path[nPath][2];
                    int j2  = Path[nPath][3];

                    int fl2 = if_Flippable(i2, j2);
                    if (fl2!=0)
                        {
                        int omega2 = fl2*FindOmega(i2,j2);

                        Psi /= psi_0_local(i2,j2,fl2);
                        FlipPlaquette(i2,j2); // flip plaquette-2
                        Psi *= psi_0_local(i2,j2,-fl2);
                        FlipPlaquette(i2,j2); // flip plaquette-2 back

                        // omega, omega1, omega2
                        Heff_plaquetteComplex +=  (-I)*Psi*varW(indexVP[n2OrderVP + 3*3*nTerms*nPathDiff*(1+omega) + 3*nTerms*nPathDiff*(1+omega1) + nTerms*nPathDiff*(1+omega2) + nTerms*(nPath%nPathDiff) + 2][0]);
                        if (CompressionMode==1)  for (int j_indt=0; j_indt<j_ind_max; j_indt++) indexVP[n2OrderVP + 3*3*nTerms*nPathDiff*(1+omega) + 3*nTerms*nPathDiff*(1+omega1) + nTerms*nPathDiff*(1+omega2) + nTerms*(nPath%nPathDiff) + 2][1]+=1;
                        }

                    FlipPlaquette(i, j);   // flip plaquette back
                    }

                //--

                // 3. contribution <V(t''')><V(t')V(t'')>: omega2,omega,omega1
                int i2 = Path[nPath][2];
                int j2 = Path[nPath][3];
                int fl2 = if_Flippable(i2,j2);
                if (fl2!=0)
                    {
                    int omega2 = fl2*FindOmega(i2,j2);

                    Psi = 1.0;
                    Psi /= psi_0_local(i2,j2,fl2);
                    FlipPlaquette(i2,j2); // flip plaquette-2
                    Psi *= psi_0_local(i2,j2,-fl2);
                    FlipPlaquette(i2,j2); // flip plaquette-2 back

                    Psi /= psi_0_local_ij;
                    FlipPlaquette(i, j); // flip plaquette
                    Psi *= psi_0_local(i,j, -fl);
                    //FlipPlaquette(i, j); // flip plaquette back

                    i1  = Path[nPath][0];
                    j1  = Path[nPath][1];
                    fl1 = if_Flippable(i1,j1);
                    if (fl1!=0)
                        {
                        int omega1 = fl1*FindOmega(i1,j1);

                        Psi /= psi_0_local(i1,j1,fl1);
                        FlipPlaquette(i1,j1); // flip plaquette-1
                        Psi *= psi_0_local(i1,j1,-fl1);
                        FlipPlaquette(i1,j1); // flip plaquette-1 back

                        // omega, omega1, omega2
                        Heff_plaquetteComplex +=  (-I)*Psi*varW(indexVP[n2OrderVP + 3*3*nTerms*nPathDiff*(1+omega) + 3*nTerms*nPathDiff*(1+omega1) + nTerms*nPathDiff*(1+omega2) + nTerms*(nPath%nPathDiff) + 3][0]);
                        if (CompressionMode==1)  for (int j_indt=0; j_indt<j_ind_max; j_indt++) indexVP[n2OrderVP + 3*3*nTerms*nPathDiff*(1+omega) + 3*nTerms*nPathDiff*(1+omega1) + nTerms*nPathDiff*(1+omega2) + nTerms*(nPath%nPathDiff) + 3][1]+=1;
                        }

                    //FlipPlaquette(i1, j1);   // flip plaquette-1 back
                    FlipPlaquette(i, j);   // flip plaquette back
                    }
                //--

                // 4. contribution <V(t')><V(t'')><V(t''')>

                Psi = 1.0;
                Psi /= psi_0_local_ij;
                FlipPlaquette(i, j); // flip plaquette
                Psi *= psi_0_local(i,j, -fl);
                FlipPlaquette(i, j); // flip plaquette back

                i1 = Path[nPath][0];
                j1 = Path[nPath][1];
                fl1 = if_Flippable(i1, j1);
                if (fl1!=0)
                    {
                    int omega1 = fl1*FindOmega(i1,j1);

                    Psi /= psi_0_local(i1,j1,fl1);
                    FlipPlaquette(i1,j1); // flip plaquette-1
                    Psi *= psi_0_local(i1,j1,-fl1);
                    FlipPlaquette(i1,j1); // flip plaquette-1

                    int i2  = Path[nPath][2];
                    int j2  = Path[nPath][3];

                    int fl2 = if_Flippable(i2, j2);
                    if (fl2!=0)
                        {
                        int omega2 = fl2*FindOmega(i2,j2);

                        Psi /= psi_0_local(i2,j2,fl2);
                        FlipPlaquette(i2,j2); // flip plaquette-2
                        Psi *= psi_0_local(i2,j2,-fl2);
                        FlipPlaquette(i2,j2); // flip plaquette-2 back

                        // omega, omega1, omega2
                        Heff_plaquetteComplex +=  (-I)*Psi*varW(indexVP[n2OrderVP + 3*3*nTerms*nPathDiff*(1+omega) + 3*nTerms*nPathDiff*(1+omega1) + nTerms*nPathDiff*(1+omega2) + nTerms*(nPath%nPathDiff) + 4][0]);
                        if (CompressionMode==1)  for (int j_indt=0; j_indt<j_ind_max; j_indt++) indexVP[n2OrderVP + 3*3*nTerms*nPathDiff*(1+omega) + 3*nTerms*nPathDiff*(1+omega1) + nTerms*nPathDiff*(1+omega2) + nTerms*(nPath%nPathDiff) + 4][1]+=1;
                        }

                    //FlipPlaquette(i1, j1);   // flip plaquette-1 back
                    }
                //FlipPlaquette(i, j);   // flip plaquette back
                //--

                }


            }

        return Heff_plaquetteComplex;
        }
	}


cdouble findHeffComplex(vector<int> &spins) //
	{

    cdouble tempHeff = 0;

	for (int j=0; j<L; j++)  S[0][j][2] = spins[j];
	tempHeff += psi_neural->log_psi_s(spins);


    int i,j;
	for (i=0; i<H; i++)
		{
        for (j=0; j<L; j++)
			{
			tempHeff += Heff_plaquetteComplex(i,j, varW);
			}
		}



	cdouble varW0 = varW(numberOfVarParameters);

	return varW0+tempHeff;
	}


}
