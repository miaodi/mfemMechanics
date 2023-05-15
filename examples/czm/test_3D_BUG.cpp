
#include "Plugin.h"
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class GeneralResidualMonitor : public IterativeSolverMonitor
{
public:
    GeneralResidualMonitor( const std::string& prefix_, int print_lvl ) : prefix( prefix_ )
    {
        print_level = print_lvl;
    }

    virtual void MonitorResidual( int it, double norm, const Vector& r, bool final );

private:
    const std::string prefix;
    int print_level;
    mutable double norm0;
};

void ReferenceConfiguration( const Vector& x, Vector& y )
{
    // Set the reference, stress free, configuration
    y = x;
}

void GeneralResidualMonitor::MonitorResidual( int it, double norm, const Vector& r, bool final )
{
    if ( print_level == 1 || ( print_level == 3 && ( final || it == 0 ) ) )
    {
        mfem::out << std::setprecision( 16 ) << prefix << " iteration " << setw( 2 ) << it << " : ||r|| = " << norm;
        if ( it > 0 )
        {
            mfem::out << ",  ||r||/||r_0|| = " << norm / norm0;
        }
        else
        {
            norm0 = norm;
        }
        mfem::out << '\n';
    }
}

int main( int argc, char* argv[] )
{
    // 1. Parse command-line options.
    const char* mesh_file = "../../data/3D_DCB_BUG.msh";
    int order = 1;
    bool static_cond = false;
    bool visualization = 1;
    int refineLvl = 0;
    int localRefineLvl = 0;

    OptionsParser args( argc, argv );
    args.AddOption( &mesh_file, "-m", "--mesh", "Mesh file to use." );
    args.AddOption( &order, "-o", "--order", "Finite element order (polynomial degree)." );
    args.AddOption( &static_cond, "-sc", "--static-condensation", "-no-sc", "--no-static-condensation",
                    "Enable static condensation." );
    args.AddOption( &visualization, "-vis", "--visualization", "-no-vis", "--no-visualization",
                    "Enable or disable GLVis visualization." );
    args.AddOption( &refineLvl, "-r", "--refine-level", "Finite element refine level." );
    args.AddOption( &localRefineLvl, "-lr", "--local-refine-level", "Finite element local refine level." );
    args.Parse();
    if ( !args.Good() )
    {
        args.PrintUsage( cout );
        return 1;
    }
    args.PrintOptions( cout );
    cout << static_cond << endl;

    // 2. Read the mesh from the given mesh file. We can handle triangular,
    //    quadrilateral, tetrahedral or hexahedral elements with the same code.
    Mesh* mesh = new Mesh( mesh_file, 1, 1 );
    int dim = mesh->Dimension();

    // 3. Select the order of the finite element discretization space. For NURBS
    //    meshes, we increase the order by degree elevation.
    if ( mesh->NURBSext )
    {
        mesh->DegreeElevate( order, order );
    }

    //  4. Mesh refinement
    for ( int k = 0; k < refineLvl; k++ )
        mesh->UniformRefinement();

    for ( int k = 0; k < localRefineLvl; k++ )
    {
        int ne = mesh->GetNE();
        auto eles = mesh->GetElementsArray();
        Array<Refinement> refinements;
        for ( int i = 0; i < ne; i++ )
        {
            double* node{ nullptr };
            for ( int j = 0; j < eles[i]->GetNVertices(); j++ )
            {
                const int vi = eles[i]->GetVertices()[j];
                node = mesh->GetVertex( vi );
                if ( std::abs( node[1] ) < 1e-10 && node[0] + 1e-10 > 0 )
                {
                    refinements.Append( i );
                    break;
                }
            }
        }
        mesh->GeneralRefinement( refinements );
    }
    ofstream file;
    file.open( "refined.vtk" );
    mesh->PrintVTK( file );
    file.close();

    // 5. Define a finite element space on the mesh. Here we use vector finite
    //    elements, i.e. dim copies of a scalar finite element space. The vector
    //    dimension is specified by the last argument of the FiniteElementSpace
    //    constructor. For NURBS meshes, we use the (degree elevated) NURBS space
    //    associated with the mesh nodes.
    FiniteElementCollection* fec;
    FiniteElementSpace* fespace;
    if ( mesh->NURBSext )
    {
        fec = NULL;
        fespace = mesh->GetNodes()->FESpace();
    }
    else
    {
        fec = new DG_FECollection( order, dim, mfem::BasisType::GaussLobatto );
        fespace = new FiniteElementSpace( mesh, fec, dim, mfem::Ordering::byVDIM );
    }
    cout << "Number of vertices: " << fespace->GetNV() << endl;
    cout << "Number of finite element unknowns: " << fespace->GetTrueVSize() << endl << "Assembling: " << endl;
    Array<int> ess_tdof_list, ess_bdr( mesh->bdr_attributes.Max() );
    ess_bdr = 0;
    ess_bdr[13] = 1;

    fespace->GetEssentialTrueDofs( ess_bdr, ess_tdof_list );

    printf( "Mesh is %i dimensional.\n", dim );
    printf( "Number of mesh attributes: %i\n", mesh->attributes.Size() );
    printf( "Number of boundary attributes: %i\n", mesh->bdr_attributes.Size() );
    printf( "Number of elements: %i\n", fespace->GetNE() );
    printf( "Max of boundary attributes: %i\n", mesh->bdr_attributes.Max() );
    const mfem::FiniteElement* fe;
    mfem::ElementTransformation* tr;
    mfem::IntegrationPoint ip;
    Array<int> vdofs;
    mfem::Vector pt, shape;
    shape.SetSize( 8 );

    mfem::IntegrationRules GLIntRules( 0, mfem::Quadrature1D::GaussLobatto );
    auto intrule = GLIntRules.Get( mfem::Geometry::SQUARE, 2 * 1 - 3 );
    for ( int i = 0; i < intrule.GetNPoints(); i++ )
    {
        auto pt = intrule.IntPoint( i );
        std::cout << pt.x << " " << pt.y << " " << pt.z << std::endl;
    }
    // vdofs.Print();
    // std::cout<<std::endl;
    // fespace->GetElementVDofs(1, vdofs);
    // vdofs.Print();
    // std::cout<<std::endl;
    // mfem::DenseMatrix mat;
    // mat = static_cast<mfem::IsoparametricTransformation*>(fespace->GetElementTransformation(0))->GetPointMat();
    // mat.Print(std::cout, 20);
    // mat = static_cast<mfem::IsoparametricTransformation*>(fespace->GetElementTransformation(1))->GetPointMat();
    // mat.Print(std::cout, 20);

    // 8. Define the solution vector x as a finite element grid function
    //    corresponding to fespace. Initialize x with initial guess of zero,
    //    which satisfies the boundary conditions.
    Vector Nu( mesh->attributes.Max() );
    Nu = .0;
    PWConstCoefficient nu_func( Nu );

    Vector E( mesh->attributes.Max() );
    E = 1E8;
    PWConstCoefficient E_func( E );

    IsotropicElasticMaterial iem( E_func, nu_func );

    plugin::Memorize mm( mesh );

    auto intg = new plugin::NonlinearElasticityIntegrator( iem, mm );
    intg->setNonlinear( false );

    NonlinearForm* nlf = new NonlinearForm( fespace );
    nlf->AddDomainIntegrator( intg );
    nlf->SetEssentialTrueDofs( ess_tdof_list );

    GeneralResidualMonitor newton_monitor( "Newton", 1 );
    GeneralResidualMonitor j_monitor( "GMRES", 3 );

    // Set up the Jacobian solver
    auto j_gmres = new UMFPackSolver();

    auto newton_solver = new plugin::Crisfield();

    // Set the newton solve parameters
    newton_solver->iterative_mode = true;
    newton_solver->SetSolver( *j_gmres );
    newton_solver->SetOperator( *nlf );
    newton_solver->SetPrintLevel( -1 );
    newton_solver->SetMonitor( newton_monitor );
    newton_solver->SetRelTol( 1e-7 );
    newton_solver->SetAbsTol( 1e-13 );
    newton_solver->SetMaxIter( 7 );
    newton_solver->SetPrintLevel( 0 );
    newton_solver->SetDelta( 1e-3 );
    newton_solver->SetMaxDelta( 1e-1 );
    newton_solver->SetMinDelta( 1e-14 );
    newton_solver->SetMaxStep( 2 );
    newton_solver->SetPhi( 10. );
    Vector zero;

    GridFunction u( fespace );
    u = 0.;
    VectorArrayCoefficient d( dim );
    for ( int i = 0; i < dim; i++ )
    {
        d.Set( i, new ConstantCoefficient( 0.0 ) );
    }

    Vector topDisp( mesh->bdr_attributes.Max() );
    topDisp = .0;
    topDisp( 11 ) = -1e-1;
    topDisp( 12 ) = 1e-1;
    d.Set( 1, new PWConstCoefficient( topDisp ) );

    Vector activeBCX( mesh->bdr_attributes.Max() );
    activeBCX = 0.0;
    activeBCX( 10 ) = 1e15;
    Vector activeBCY( mesh->bdr_attributes.Max() );
    activeBCY = 0.0;
    activeBCY( 10 ) = 1e15;
    activeBCY( 11 ) = 1e15;
    activeBCY( 12 ) = 1e15;

    VectorArrayCoefficient hevi( dim );
    hevi.Set( 0, new PWConstCoefficient( activeBCX ) );
    hevi.Set( 1, new PWConstCoefficient( activeBCY ) );
    hevi.Set( 2, new PWConstCoefficient( activeBCX ) );
    nlf->AddInteriorFaceIntegrator( new plugin::NonlinearInternalPenaltyIntegrator( 1e16 ) );
    nlf->AddInteriorFaceIntegrator( new plugin::CZMIntegrator( mm, 5.7E8, 5.7E8, 1E-3, 5E-3 ) );

    auto penalty = new plugin::NonlinearDirichletPenaltyIntegrator( d, hevi );
    penalty->SetIntRule( &GLIntRules.Get( mfem::Geometry::SQUARE, -1 ) );
    nlf->AddBdrFaceIntegrator( penalty );

    // nlf->AddBdrFaceIntegrator( new plugin::NonlinearVectorBoundaryLFIntegrator( f ) );
    // 15. Save data in the ParaView format
    ParaViewDataCollection paraview_dc( "bar_3D_bug", mesh );
    paraview_dc.SetPrefixPath( "ParaView" );
    paraview_dc.SetLevelsOfDetail( order );
    paraview_dc.SetCycle( 0 );
    paraview_dc.SetDataFormat( VTKFormat::BINARY );
    paraview_dc.SetHighOrderOutput( true );
    paraview_dc.SetTime( 0.0 ); // set the time
    paraview_dc.RegisterField( "Displace", &u );
    newton_solver->SetDataCollection( &paraview_dc );
    paraview_dc.Save();

    newton_solver->Mult( zero, u );

    return 0;
}