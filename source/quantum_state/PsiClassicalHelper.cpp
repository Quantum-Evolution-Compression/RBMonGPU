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
#define cdouble std::complex<double>
cdouble I(0.0,1.0);

// _______________________________________________________________
// parameters read from file

int L;
int H;
int enable_full_table;
int PerturbationTheoryOrder;
// _______________________________________________________________

int numberOfVarParameters;

double time_current;
double time_epoch;


// _______________________________________________________________
// vectors

vector<vector<vector<int>>> S;     // i,j -- coordinates of the plaquette, only k=2 is used
int const numberOfVarParametrsMax=300000; // before compression Stripe3Order (12321), Square 3Order (12160) // debug:
int indexVP[numberOfVarParametrsMax][2] = {0}; // initialize with zeros

Eigen::VectorXcd varW; // (numberOfVarParameters+1); // variational parameters; last one is the "dumb" VP for normalization and the global phase

//_______________________________________________________________
// neural network

rbm_on_gpu::PsiDeepMin* psi_neural = nullptr;

//_______________________________________________________________

void LoadParameters(string directory)
	{
	std::string filenameParam = directory + "/a_parameters.csv";
	std::ifstream fileParam;
	fileParam.open (filenameParam.c_str());

	if (fileParam.is_open()==true)  cout << filenameParam << " was successfully loaded by loadVP()" << endl;
    if (fileParam.is_open()==false) cout << filenameParam << " was NOT loaded by loadVP()" << endl;
	
	int Nparams = 16;
	vector<string> dataVector(Nparams);

	int i = 0;
	string temp;
	
	while (i < Nparams)
		{
		getline (fileParam, temp, ',');
		getline (fileParam, dataVector[i]);
    	//cout << dataVector[i] << endl;
	 	i++;
		}
    
    cout << "Trying to read " << Nparams << " parameters from a_parameters.csv" << endl;
    
	L       = atoi(dataVector[0].c_str()); 
	H       = atoi(dataVector[1].c_str());
	enable_full_table= atoi(dataVector[2].c_str());
	//extra1= atoi(dataVector[3].c_str());
	//extra2= atoi(dataVector[4].c_str());
	//t_start = atof(dataVector[5].c_str());
	//t_stop  = atof(dataVector[6].c_str());
	//dt      = atof(dataVector[7].c_str());
	//Nsteps  = atoi(dataVector[8].c_str());
	//Ntherm  = atoi(dataVector[9].c_str());
    //Nmeasurements    = atoi(dataVector[10].c_str());
	//J       = atof(dataVector[11].c_str());
	//DirectEnumerationMode  = atoi(dataVector[12].c_str());
	string VariationalModeString  = dataVector[13].c_str();
	PerturbationTheoryOrder= atoi(dataVector[14].c_str());

	if (enable_full_table) psi_neural->enable_full_table();
	
    if (VariationalModeString.size()!=0)
		{
		int stringLength = VariationalModeString.size();
		if (VariationalModeString[stringLength-1]=='\r') VariationalModeString.erase(stringLength-1);
		}
		
	if (VariationalModeString=="Ising")
		{
		cout << "VariationalModeString=Ising" << endl;
		if (PerturbationTheoryOrder==1) numberOfVarParameters=4;
		if (PerturbationTheoryOrder==2) numberOfVarParameters=18;
		}     
	else
		{
		cout << "Unknown VariationalModeString:" << VariationalModeString << endl;
		return;
		}
		

	S = vector<vector<vector<int>>> (H, vector<vector<int>>(L, vector<int>(5)));  
    
	varW = Eigen::VectorXcd::Zero(numberOfVarParameters+1);
	}
	
//_______________________________________________________________

void load_neural_network(string directory, int index) 
	{
    if(psi_neural) 
        delete psi_neural;
    
    psi_neural = new rbm_on_gpu::PsiDeepMin(directory + "/psi_" + to_string(index-1) + "_compressed.txt");
	//psi_neural->enable_full_table();
	}

//_______________________________________________________________

void loadVP(std::string directory, int index, std::string ReIm) // two calls are necessary: LoadVP("Re",..,..); LoadVP("Im",..,..);
    {
    std::string filenamePos = directory + "/a_VP_" + to_string(index-1) + "_" + ReIm + ".csv";
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

	getline (filePos, temp);
	if (ReIm.find("Re") != std::string::npos) time_current = atof(temp.c_str()); // reading the current time
    if (ReIm.find("Im") != std::string::npos) time_epoch   = atof(temp.c_str()); // reading the duration of the epoch time

	cout << "varW(0)=" << varW(0) << endl;
    cout << "varW(1)=" << varW(1) << endl;
	if (ReIm.find("Re") != std::string::npos) cout << "time_current=" <<  time_current << endl;
	if (ReIm.find("Im") != std::string::npos) cout << "time_epoch="   <<  time_epoch   << endl;

    filePos.close();
	}

//_______________________________________________________________

void Compress_Load(std::string directory, int index)
    {

    string filenamePos = directory + "/a_indexVP_" + to_string(index-1) + ".csv";
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
	cout << "indexVP[0][0],indexVP[1][0],indexVP[2][0]=" << indexVP[0][0] << ", " << indexVP[1][0] << ", " << indexVP[2][0] << endl;
    filePos.close();
    }

//_______________________________________________________________

cdouble psi_0_local(int i, int j, int fl) // in interaction representation
    {
    cdouble psi_0_local_temp=1.0;

    vector<int> spins(L);
	double Es_total=0;
    for (int j=0; j<L; j++)
		{
		spins[j] =   S[0][j][2]; // This is super inefficient. psi_0_local is called very frequently. [Peter, 20.02.2020]
		Es_total += -S[0][j][2]*(S[0][(j+1)%L][2]+S[0][(j-1+L)%L][2])/2;
		}

	psi_0_local_temp *= exp(psi_neural->log_psi_s(spins));
    psi_0_local_temp *= exp((-I)*(time_current-time_epoch)*Es_total); // interaction->Schrodinger representation additional rotation
    return psi_0_local_temp;
    }

//_______________________________________________________________

int FindOmega(int i, int j) //
	{
    int Omega = (S[i][(j+1)%L][2]+S[i][(j-1+L)%L][2])/2;  // = -1,0,1
    return Omega;
    }

//_______________________________________________________________

void FlipPlaquette(int i, int j)
	{
    S[i][j][2] *= -1;
    }

//_______________________________________________________________

int if_Flippable(int i, int j)
	{
    return S[i][j][2];
	}

//_______________________________________________________________

cdouble Heff_plaquetteComplex(int i, int j, Eigen::VectorXcd& varW) // doesn't take into account the factor of 2
																	// returns Shroedinger representation
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
	return 0; // should never happen
	}

//_______________________________________________________________

cdouble findHeffComplex(vector<int> &spins) // returns log(wavefunction) in the interaction representation
	{

    cdouble tempHeff = 0;
	double Es_total = 0;
	for (int j=0; j<L; j++)
		{
		S[0][j][2] = spins[j];
		Es_total += -spins[j]*(spins[(j+1)%L]+spins[(j-1+L)%L])/2;
		}
	tempHeff += psi_neural->log_psi_s(spins); // already interaction representation

    int i,j;
	for (i=0; i<H; i++)
		{
        for (j=0; j<L; j++)
			{
			tempHeff += Heff_plaquetteComplex(i,j, varW); // contrubution of the last epoch in Heisenberg representation
			}
		}
	tempHeff += (+I)*Es_total*time_epoch;  // "rotation" to obtain the interaction representation from Schroedinger; cancels extra phase-contribution from the last epoch 


	cdouble varW0 = varW(numberOfVarParameters);

	return varW0+tempHeff;
    // return tempHeff;
	}


}
