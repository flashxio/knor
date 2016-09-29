#include <El.hpp>

using namespace El;

int main( int argc, char* argv[] )
{
    Environment env( argc, argv );

    try
    {
        const std::string data_fn =
            El::Input<std::string>("-f","datafile (TSV)","");

        DistMatrix<double> A;
        DistMatrix<double> B;
        El::Read(A, data_fn, El::ASCII);
        El::Transpose(A, B);

        std::string::size_type idx = data_fn.rfind('.');
        std::string outfn = data_fn.substr(0, idx) + std::string("_cw");

        El::Write(B, outfn, El::ASCII);
    }
    catch( std::exception& e ) { ReportException(e); }

    return 0;
}
