#include <string>
#include <complex>
#include <vector>


namespace Peter {

void LoadParameters(std::string directory);
void load_neural_network(std::string directory, int index);
void loadVP(std::string directory, int index, std::string ReIm);
void Compress_Load(std::string directory, int index);
std::complex<double> findHeffComplex(std::vector<int> &spins);

}
