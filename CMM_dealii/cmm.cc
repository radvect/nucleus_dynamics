#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <sstream>

#include <deal.II/base/tensor_function.h>

using namespace dealii;
template <int dim>
class CellMot
{
public:
  CellMot (const std::string &prm_file);
  void run ();
private:
  void print_mesh_info (const Triangulation<dim> &tria,
                     const std::string        &filename);
  void grid_1 ();
  void make_grid_and_dofs ();
  inline void get_params (); 
  void on_membrane ();
  inline void move_mesh ();
  void assemble_system_parts ();
  void assemble_system ();
  void assemble_system_a_m ();
  void makenewkm ();
  void solve (BlockVector<double> &return_vector);
  void solve2 (BlockVector<double> &return_vector);
  void solve3 (BlockVector<double> &return_vector);
  void solve4 (BlockVector<double> &return_vector);
  void solve5 (BlockVector<double> &return_vector);
  void solve6 (BlockVector<double> &return_vector);
  void output_results () const;
  void output_results_v () const;
  void output_results_f () const;
  void output_results_u () const;
  void l2_diff ();
  Triangulation<dim>   triangulation;
  FESystem<dim>        fe;
  DoFHandler<dim>      dof_handler;
  dealii::AffineConstraints<double> constraints;
  BlockSparsityPattern      	sparsity_pattern;
  BlockSparseMatrix<double> 	system_matrix,
				A_matrix,
				B_matrix;

  BlockVector<double> solution;
  BlockVector<double> old_solution;
  BlockVector<double> system_rhs;
  BlockVector<double> old_sol;

  SparsityPattern      sparsity_pattern_a;
  SparseMatrix<double> old_mass;   
  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> laplace_matrix;

  Vector<double>       displacement;
  Vector<double>       old_displacement;
  double	pr;	
  double       	youngs;
  double       	nu;
  double  c11;
  double  c12;
  double  c33;
  double  d11;
  double  d12;
  double  d33;

  int 	 	refine;
  double        time;
  double        time_step;
  unsigned int  timestep_number;
  double x;
  double vol1;
ParameterHandler parameters;

};
template <int dim>
void CellMot<dim>::print_mesh_info(const Triangulation<dim> &tria,
                     const std::string        &filename)
{
  std::cout << "Mesh info:" << std::endl
            << " dimension: " << dim << std::endl
            << " no. of cells: " << tria.n_active_cells() << std::endl;
  std::ofstream out (filename.c_str());
  GridOut grid_out;
  grid_out.write_eps (tria, out);
  std::cout << " written to " << filename
            << std::endl
            << std::endl;
}
template <int dim>
void CellMot<dim>::grid_1 ()
{
//  Triangulation<2> triangulation;
  GridIn<3> gridin;
  gridin.attach_triangulation(triangulation);
  std::ifstream f("sphere4.msh");
  gridin.read_msh(f);
//  print_mesh_info(triangulation, "grid-1.eps");
}




// CLASS IMPLIMENTATION AND DECLARING PARAMETERS

template <int dim>
CellMot<dim>::CellMot (const std::string &prm_file)
  :
fe(FE_Q<dim>(1), dim),
dof_handler (triangulation)
{
parameters.declare_entry ("timestep", "0.001",
                          Patterns::Double (0, 1),
                          "the size of time step");
parameters.declare_entry ("youngs", "1.5",
                          Patterns::Double (0, 10),
                          "Youngs Modulus of actin network");
parameters.declare_entry ("nu", "0.3",
                          Patterns::Double (0, 10),
                          "Poisson ratio of actin network");

parameters.declare_entry ("pr", "0.26",
                          Patterns::Double (-2, 10),
                          "pressure coefficient");
parameters.declare_entry ("refine", "4",
                          Patterns::Integer (0, 10),
                          "grid refinement");
parameters.declare_entry ("tol", "0.0001",
                          Patterns::Double (0, 1),
                          "tolerance");


parameters.parse_input (prm_file);
}

template <int dim>
Point<dim> ellipse (const Point<dim> &p)
{
  Point<dim> q = p;

  q[2]/=1.5 ;
  return q;
}

// MAKE GRID AND DOFS AND SPARSITY PATTERNS
// Initialise system matrices

template <int dim>
void CellMot<dim>::make_grid_and_dofs ()
{
//  GridGenerator::hyper_ball (triangulation);
  grid_1();
  //GridGenerator::hyper_shell (triangulation,Point<dim>(0,0),0.5,1.0,96,false);
  //GridGenerator::hyper_ball (triangulation);
  static const SphericalManifold<dim> boundary(Point<dim>(1.0, 1.0, 1.0));
  triangulation.set_all_manifold_ids(0);
  triangulation.set_manifold(0, boundary);
  refine = parameters.get_integer ("refine");
  triangulation.refine_global (refine);
  //GridTools::transform (&ellipse<dim>, triangulation); 
  
  dof_handler.distribute_dofs (fe);
  
  std::cout << "Number of active cells: "
	    << triangulation.n_active_cells()
	    << std::endl
	    << std::endl;
  constraints.clear ();
  DoFTools::make_hanging_node_constraints (dof_handler,
					   constraints);
  constraints.close();
  
  
  // initialising sparsity patterns in the FBE
  
  //CompressedSparsityPattern b_sparsity(dof_handler.n_dofs());
  DynamicSparsityPattern b_sparsity(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
				  b_sparsity,
				  constraints,
				  true);
  sparsity_pattern.reinit (dim,dim);
  for (unsigned int i=0; i<dim; ++i)
    for (unsigned int j=0; j<dim; ++j)
      sparsity_pattern.block(i,j).copy_from(b_sparsity);
  sparsity_pattern.collect_sizes();
  system_matrix.reinit (sparsity_pattern);
  
  A_matrix.reinit(sparsity_pattern);
  B_matrix.reinit(sparsity_pattern);
  
  solution.reinit (dim);
  for (unsigned int i=0; i<dim; ++i)
  solution.block(i).reinit (dof_handler.n_dofs());
  solution.collect_sizes ();
  
  old_solution.reinit (solution);
  system_rhs.reinit (solution);
  old_sol.reinit (solution);
  
  // and in the RDE
  //CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
  DynamicSparsityPattern c_sparsity(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
				  c_sparsity,
				  constraints,
				  /*keep_constrained_dofs = */ true);
  sparsity_pattern_a.copy_from(c_sparsity);
  old_mass.reinit(sparsity_pattern_a);
  mass_matrix.reinit(sparsity_pattern_a);
  laplace_matrix.reinit(sparsity_pattern_a);

  displacement.reinit(dof_handler.n_dofs());
}

// Get parameters

template <int dim>
void CellMot<dim>::get_params()
{
time_step = parameters.get_double ("timestep");
youngs = parameters.get_double ("youngs");
nu = parameters.get_double ("nu");

c11 = youngs*(1-nu)/((1+nu)*(1-2*nu));
c12 = youngs*nu/((1+nu)*(1-2*nu));
c33 = youngs/(2*(1+nu));

pr = parameters.get_double ("pr");
}

template <int dim>
void CellMot<dim>::move_mesh ()
{
QGauss<dim>   quadrature_formula(2);
const unsigned int   n_q_points      = quadrature_formula.size();
FEValues<dim> fe_values (fe, quadrature_formula,
                         update_values    | update_gradients |
                         update_quadrature_points  | update_JxW_values);
std::vector<Vector<double> >      u_value(n_q_points, Vector<double>(dim));
std::vector<Vector<double> >      v_value(n_q_points, Vector<double>(dim));
std::vector<Vector<double> >      w_value(n_q_points, Vector<double>(dim));
std::vector<Vector<double> >      old_u_value(n_q_points, Vector<double>(dim));
std::vector<Vector<double> >      old_v_value(n_q_points, Vector<double>(dim));
std::vector<Vector<double> >      old_w_value(n_q_points, Vector<double>(dim));
std::vector<bool> vertex_touched (triangulation.n_vertices(),
                                  false);
//double vol=GridTools::volume(triangulation);
typename DoFHandler<dim>::active_cell_iterator
cell = dof_handler.begin_active(),
endc = dof_handler.end();
for (; cell!=endc; ++cell)
   {
   fe_values.reinit (cell);
   fe_values.get_function_values (old_sol.block(0), old_u_value);
   fe_values.get_function_values (old_sol.block(1), old_v_value);
   fe_values.get_function_values (old_sol.block(2), old_w_value);
   fe_values.get_function_values (solution.block(0), u_value);
   fe_values.get_function_values (solution.block(1), v_value);
   fe_values.get_function_values (solution.block(2), w_value);

   for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
      {
      if (vertex_touched[cell->vertex_index(v)] == false)
        {
        vertex_touched[cell->vertex_index(v)] = true;
        Point<dim> v_disp;
        v_disp[0]=0;
        v_disp[1]=0;
	v_disp[2]=0;
	for (unsigned int q=0; q<n_q_points; ++q){
   	v_disp[0]+=(u_value[q](0)-old_u_value[q](0))/*vol/vol1*/;
       	v_disp[1]+=(v_value[q](0)-old_v_value[q](0))/*vol/vol1*/;
	v_disp[2]+=(w_value[q](0)-old_w_value[q](0))/*vol/vol1*/;}
	cell->vertex(v) += v_disp;
	}
      }
    }
}

template <int dim>
void CellMot<dim>::makenewkm ()
{
mass_matrix=0;
laplace_matrix=0;
MatrixCreator::create_mass_matrix(dof_handler,
                                      QGauss<dim>(2),
                                      mass_matrix);
MatrixCreator::create_laplace_matrix(dof_handler,
                                         QGauss<dim>(2),
                                         laplace_matrix);
}

// Assemble system

template <int dim>
void CellMot<dim>::assemble_system_parts ()
{
// initialise system matrices
A_matrix=0;
B_matrix=0;
system_matrix = 0;
system_rhs = 0;

double p1;

// Defining the type of quadrature
QGauss<dim>   quadrature_formula(2);
QGauss<dim-1>  face_quadrature_formula(2);

// update FEValues
FEValues<dim> fe_values (fe, quadrature_formula,
                         update_values    | update_gradients |
                         update_quadrature_points  | update_JxW_values);
FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                  update_values    | update_gradients | update_normal_vectors |
                                  update_quadrature_points  | update_JxW_values);
const unsigned int   dofs_per_cell   = fe.dofs_per_cell;

const unsigned int   n_q_points      = quadrature_formula.size();
const unsigned int   n_face_q_points = face_quadrature_formula.size();

std::vector<types::global_dof_index>  local_face_dof_indices  (fe.dofs_per_face);
std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);


Vector<double> cell_1rhs (dofs_per_cell);
Vector<double> cell_2rhs (dofs_per_cell);
Vector<double> cell_3rhs (dofs_per_cell);

//FullMatrix<double>   cell_matrix_mass(dofs_per_cell, dofs_per_cell);
FullMatrix<double> cell_a11 (dofs_per_cell,dofs_per_cell);
FullMatrix<double> cell_a12 (dofs_per_cell,dofs_per_cell);
FullMatrix<double> cell_a13 (dofs_per_cell,dofs_per_cell);
FullMatrix<double> cell_a21 (dofs_per_cell,dofs_per_cell);
FullMatrix<double> cell_a22 (dofs_per_cell,dofs_per_cell);
FullMatrix<double> cell_a23 (dofs_per_cell,dofs_per_cell);
FullMatrix<double> cell_a31 (dofs_per_cell,dofs_per_cell);
FullMatrix<double> cell_a32 (dofs_per_cell,dofs_per_cell);
FullMatrix<double> cell_a33 (dofs_per_cell,dofs_per_cell);
FullMatrix<double> cell_b11 (dofs_per_cell,dofs_per_cell);
FullMatrix<double> cell_b12 (dofs_per_cell,dofs_per_cell);
FullMatrix<double> cell_b13 (dofs_per_cell,dofs_per_cell);
FullMatrix<double> cell_b21 (dofs_per_cell,dofs_per_cell);
FullMatrix<double> cell_b22 (dofs_per_cell,dofs_per_cell);
FullMatrix<double> cell_b23 (dofs_per_cell,dofs_per_cell);
FullMatrix<double> cell_b31 (dofs_per_cell,dofs_per_cell);
FullMatrix<double> cell_b32 (dofs_per_cell,dofs_per_cell);
FullMatrix<double> cell_b33 (dofs_per_cell,dofs_per_cell);
Vector<double>     cell_rhs (dofs_per_cell);
Vector<double> f1 (dofs_per_cell);
Vector<double> f2 (dofs_per_cell);
Vector<double> f3 (dofs_per_cell);

MatrixCreator::create_mass_matrix(dof_handler,
                                      QGauss<dim>(2),
                                      old_mass);

//double vol = GridTools::volume(triangulation);

// u v and a values at quadrature points
std::vector<Vector<double> >      u_value(n_q_points, Vector<double>(dim));
std::vector<Vector<double> >      v_value(n_q_points, Vector<double>(dim));
std::vector<Vector<double> >      w_value(n_q_points, Vector<double>(dim));
std::vector<Vector<double> >      old_u_value(n_q_points, Vector<double>(dim));
std::vector<Vector<double> >      old_v_value(n_q_points, Vector<double>(dim));
std::vector<Vector<double> >      old_w_value(n_q_points, Vector<double>(dim));

std::vector<std::vector<Tensor<1,dim> > >  u_grad(n_q_points,
        std::vector<Tensor<1,dim> > (dim));
std::vector<std::vector<Tensor<1,dim> > >  v_grad(n_q_points,
        std::vector<Tensor<1,dim> > (dim));
std::vector<std::vector<Tensor<1,dim> > >  w_grad(n_q_points,
        std::vector<Tensor<1,dim> > (dim));

std::vector<bool> vertex_touched (triangulation.n_vertices(),
                                  false);
typename DoFHandler<dim>::active_cell_iterator
cell = dof_handler.begin_active(),
endc = dof_handler.end();
for (; cell!=endc; ++cell)
   {
   // initialising cell values
   fe_values.reinit (cell);
//   cell_matrix_mass = 0;
   f1=0;
   f2=0;
   f3=0;
   cell_rhs = 0;
   cell_a11 = 0;
   cell_a12 = 0;
   cell_a13 = 0;
   cell_a21 = 0;
   cell_a22 = 0;
   cell_a23 = 0;
   cell_a31 = 0;
   cell_a32 = 0;
   cell_a33 = 0;
   cell_b11 = 0;
   cell_b12 = 0;
   cell_b13 = 0;
   cell_b21 = 0;
   cell_b22 = 0;
   cell_b23 = 0;
   cell_b31 = 0;
   cell_b32 = 0;
   cell_b33 = 0;
   cell_1rhs = 0;
   cell_2rhs = 0;
   cell_3rhs = 0;

   // specifying the value of delta (if in the area of the membrane)
   double delta = 0;
   if (cell->user_flag_set())
   delta = 1;
/*
   for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
        {
        Point<dim> delt;
        delt = cell->vertex(v);
        double dist = sqrt(delt[0]*delt[0]+delt[1]*delt[1]);
	double expel = exp(-dist/0.25)/(1+dist*dist/0.125);
	}
*/  
 // u v and a values at quadrature points of this cell
   fe_values.get_function_gradients (old_sol.block(0), u_grad);
   fe_values.get_function_gradients (old_sol.block(1), v_grad);
   fe_values.get_function_gradients (old_sol.block(2), w_grad);
   fe_values.get_function_values (old_sol.block(0), old_u_value);
   fe_values.get_function_values (old_sol.block(1), old_v_value);
   fe_values.get_function_values (old_sol.block(2), old_w_value);
   fe_values.get_function_values (solution.block(0), u_value);
   fe_values.get_function_values (solution.block(1), v_value);
   fe_values.get_function_values (solution.block(2), w_value);

   for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
      {
	const double dilation = u_grad[q_index][0][0]
                               +v_grad[q_index][1][1]
                               +w_grad[q_index][2][2];
	double JxW = fe_values.JxW (q_index);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
           {
	   const double phi_i = fe_values.shape_value (i, q_index);
	   const double dphi_i_dx = fe_values.shape_grad (i, q_index).operator[](0);
	   const double dphi_i_dy = fe_values.shape_grad (i, q_index).operator[](1);
           const double dphi_i_dz = fe_values.shape_grad (i, q_index).operator[](2);

           p1 = 0;
           f1(i) += -p1 * dphi_i_dx * JxW;
           f2(i) += -p1 * dphi_i_dy * JxW;
           f3(i) += -p1 * dphi_i_dz * JxW;

            for (unsigned int j=0; j<dofs_per_cell; ++j)
              {
		const double dphi_j_dx = fe_values.shape_grad (j, q_index).operator[](0);
             	const double dphi_j_dy = fe_values.shape_grad (j, q_index).operator[](1);
                const double dphi_j_dz = fe_values.shape_grad (j, q_index).operator[](2);
		                cell_a11(i,j) += (d11 * dphi_i_dx * dphi_j_dx +
                                  d33 * (dphi_i_dy * dphi_j_dy + dphi_i_dz * dphi_j_dz)) * JxW;

                cell_a22(i,j) += (d11 * dphi_i_dy * dphi_j_dy +
                                  d33 * (dphi_i_dx * dphi_j_dx + dphi_i_dz * dphi_j_dz)) * JxW;

                cell_a33(i,j) += (d11 * dphi_i_dz * dphi_j_dz +
                                  d33 * (dphi_i_dy * dphi_j_dy + dphi_i_dx * dphi_j_dx)) * JxW;

                cell_a12(i,j) += (d12 * dphi_i_dx * dphi_j_dy + d33 * dphi_i_dy * dphi_j_dx) * JxW;

                cell_a21(i,j) += (d12 * dphi_j_dx * dphi_i_dy + d33 * dphi_j_dy * dphi_i_dx) * JxW;

                cell_a13(i,j) += (d12 * dphi_i_dx * dphi_j_dz + d33 * dphi_i_dz * dphi_j_dx) * JxW;

                cell_a31(i,j) += (d12 * dphi_j_dx * dphi_i_dz + d33 * dphi_j_dz * dphi_i_dx) * JxW;

                cell_a23(i,j) += (d12 * dphi_i_dy * dphi_j_dz + d33 * dphi_i_dz * dphi_j_dy) * JxW;

                cell_a32(i,j) += (d12 * dphi_j_dy * dphi_i_dz + d33 * dphi_j_dz * dphi_i_dy) * JxW;


                cell_b11(i,j) += (c11 * dphi_i_dx * dphi_j_dx +
                                  c33 * (dphi_i_dy * dphi_j_dy + dphi_i_dz * dphi_j_dz)) * JxW;

                cell_b22(i,j) += (c11 * dphi_i_dy * dphi_j_dy +
                                  c33 * (dphi_i_dx * dphi_j_dx + dphi_i_dz * dphi_j_dz)) * JxW;

                cell_b33(i,j) += (c11 * dphi_i_dz * dphi_j_dz +
                                  c33 * (dphi_i_dy * dphi_j_dy + dphi_i_dx * dphi_j_dx)) * JxW;

                cell_b12(i,j) += (c12 * dphi_i_dx * dphi_j_dy + c33 * dphi_i_dy * dphi_j_dx) * JxW;

                cell_b21(i,j) += (c12 * dphi_j_dx * dphi_i_dy + c33 * dphi_j_dy * dphi_i_dx) * JxW;

                cell_b13(i,j) += (c12 * dphi_i_dx * dphi_j_dz + c33 * dphi_i_dz * dphi_j_dx) * JxW;

                cell_b31(i,j) += (c12 * dphi_j_dx * dphi_i_dz + c33 * dphi_j_dz * dphi_i_dx) * JxW;

                cell_b23(i,j) += (c12 * dphi_i_dy * dphi_j_dz + c33 * dphi_i_dz * dphi_j_dy) * JxW;

                cell_b32(i,j) += (c12 * dphi_j_dy * dphi_i_dz + c33 * dphi_j_dz * dphi_i_dy) * JxW;
	
	      }//end j
	   }//end i
}//end q

// make boundary terms
for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
   {if (cell->face(face)->at_boundary())
      {
      fe_face_values.reinit (cell, face);

      for (unsigned int q_index=0; q_index<n_face_q_points; ++q_index)
	 {
     	 for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
            const double dilation = u_grad[q_index][0][0]+v_grad[q_index][1][1]
                                        +w_grad[q_index][2][2];
            p1 = 0;

            f1(i) += p1 * fe_face_values.shape_value (i, q_index) * 
		  fe_face_values.JxW (q_index) * fe_face_values.normal_vector(q_index)[0];
            f2(i) += p1 * fe_face_values.shape_value (i, q_index) * 
		  fe_face_values.JxW (q_index) * fe_face_values.normal_vector(q_index)[1];
            f3(i) += p1 * fe_face_values.shape_value (i, q_index) *
                  fe_face_values.JxW (q_index) * fe_face_values.normal_vector(q_index)[2];
	    }//end i
          
	 }//end q
       }// end if at boundary
   }// end face
// contributions into the global matrix and right hand side vector:
cell->get_dof_indices (local_dof_indices);
constraints.distribute_local_to_global (cell_a11,local_dof_indices,A_matrix.block(0,0));
constraints.distribute_local_to_global (cell_a12,local_dof_indices,A_matrix.block(0,1));
constraints.distribute_local_to_global (cell_a13,local_dof_indices,A_matrix.block(0,2));
constraints.distribute_local_to_global (cell_a21,local_dof_indices,A_matrix.block(1,0));
constraints.distribute_local_to_global (cell_a22,local_dof_indices,A_matrix.block(1,1));
constraints.distribute_local_to_global (cell_a23,local_dof_indices,A_matrix.block(1,2));
constraints.distribute_local_to_global (cell_a31,local_dof_indices,A_matrix.block(2,0));
constraints.distribute_local_to_global (cell_a32,local_dof_indices,A_matrix.block(2,1));
constraints.distribute_local_to_global (cell_a33,local_dof_indices,A_matrix.block(2,2));

constraints.distribute_local_to_global (cell_b11,local_dof_indices,B_matrix.block(0,0));
constraints.distribute_local_to_global (cell_b12,local_dof_indices,B_matrix.block(0,1));
constraints.distribute_local_to_global (cell_b13,local_dof_indices,B_matrix.block(0,2));
constraints.distribute_local_to_global (cell_b21,local_dof_indices,B_matrix.block(1,0));
constraints.distribute_local_to_global (cell_b22,local_dof_indices,B_matrix.block(1,1));
constraints.distribute_local_to_global (cell_b23,local_dof_indices,B_matrix.block(1,2));
constraints.distribute_local_to_global (cell_b31,local_dof_indices,B_matrix.block(2,0));
constraints.distribute_local_to_global (cell_b32,local_dof_indices,B_matrix.block(2,1));
constraints.distribute_local_to_global (cell_b33,local_dof_indices,B_matrix.block(2,2));
//constraints.distribute_local_to_global (cell_matrix_mass,local_dof_indices,old_mass);

for (unsigned int i=0; i<dofs_per_cell; ++i)
   {
   system_rhs.block(0)(local_dof_indices[i]) += f1(i);
   system_rhs.block(1)(local_dof_indices[i]) += f2(i);
   system_rhs.block(2)(local_dof_indices[i]) += f3(i);
   }
}
}

// assemble LHS AND RHS for FBE
template <int dim>
void CellMot<dim>::assemble_system ()
{
system_matrix.copy_from(A_matrix);
system_matrix.add(time_step,B_matrix);

system_rhs.operator*=(time_step);
A_matrix.vmult_add(system_rhs,old_solution);
}

// SOLVE BLOCK SYSTEM (multiple codes to reduce tolerance over time)
template <int dim>
void CellMot<dim>::solve (BlockVector<double> &return_vector)
{
double tol = parameters.get_double ("tol");
std::cout << "solve" << std::endl;
SolverControl                      solver_control (10000, tol);

SolverGMRES<BlockVector<double> >  solver(solver_control);
solver.solve (system_matrix, return_vector, system_rhs,
              PreconditionIdentity());
}
template <int dim>
void CellMot<dim>::solve2 (BlockVector<double> &return_vector)
{
std::cout << "solve" << std::endl;
SolverControl                      solver_control (3000, 2e-4);

SolverGMRES<BlockVector<double> >  solver(solver_control);
solver.solve (system_matrix, return_vector, system_rhs,
              PreconditionIdentity());
}

template <int dim>
void CellMot<dim>::solve3 (BlockVector<double> &return_vector)
{
std::cout << "solve" << std::endl;
SolverControl                      solver_control (10000, 4e-4);

SolverGMRES<BlockVector<double> >  solver(solver_control);
solver.solve (system_matrix, return_vector, system_rhs,
              PreconditionIdentity());
}
template <int dim>
void CellMot<dim>::solve4 (BlockVector<double> &return_vector)
{
std::cout << "solve" << std::endl;
SolverControl                      solver_control (3000, 9e-4);

SolverGMRES<BlockVector<double> >  solver(solver_control);
solver.solve (system_matrix, return_vector, system_rhs,
              PreconditionIdentity());
}

template <int dim>
void CellMot<dim>::solve5 (BlockVector<double> &return_vector)
{
std::cout << "solve" << std::endl;
SolverControl                      solver_control (3000, 1.5e-3);

SolverGMRES<BlockVector<double> >  solver(solver_control);
solver.solve (system_matrix, return_vector, system_rhs,
              PreconditionIdentity());
}

template <int dim>
void CellMot<dim>::solve6 (BlockVector<double> &return_vector)
{
std::cout << "solve" << std::endl;
SolverControl                      solver_control (3000, 3e-3);

SolverGMRES<BlockVector<double> >  solver(solver_control);
solver.solve (system_matrix, return_vector, system_rhs,
              PreconditionIdentity());
}




// output displacement
template<int dim>
void CellMot<dim>::output_results_f() const
{
DataOut<dim> data_out;
data_out.attach_dof_handler(dof_handler);
data_out.add_data_vector(displacement,"D");
data_out.build_patches();

int decp = (int)abs(pr*1000);


const std::string filename = "grt3dfb_asq_a20m-20_2_t0-005_ref"
                                  + Utilities::int_to_string(timestep_number, 9) + 
                                  "disp_" +
                                  ".vtk";
std::ofstream output(filename.c_str());
data_out.write_vtk(output);
}

  template <int dim>
  void CellMot<dim>::output_results_u () const
  {
    DataOut<dim> data_out;

    data_out.attach_dof_handler (dof_handler);

    for (unsigned int i=0; i<dim; ++i)
      data_out.add_data_vector (solution.block(i),
                                std::string("u_") +
                                Utilities::int_to_string(i));
data_out.build_patches();

int decp = (int)abs(pr*1000);


const std::string filename = "grt3dfb_asq_0-01rad_t0-005_ref" +
                                  Utilities::int_to_string(refine, 1) +
                                  "_p_"
                                  + Utilities::int_to_string(decp, 5) +                           
                                 "-U-"
                                  + Utilities::int_to_string(timestep_number, 9) +
                                  ".vtk";

std::ofstream output(filename.c_str());
data_out.write_vtk(output);
}


// CALCULATING THE L2 ERROR
template <int dim>
void CellMot<dim>::l2_diff ()
{
//Vector<double> difference_vector_u;
//Vector<double> difference_vector_v;
Vector<double> difference_vector_d;
//double L2_error_u;
//double L2_error_v;
//double L2_error_a;
//*double L2_error_m;
double L2_error_d;
std::ofstream myfile;

/*difference_vector_u.operator=(solution.block(0));
difference_vector_u.add(-1,old_solution.block(0));
difference_vector_v.operator=(solution.block(1));
difference_vector_v.add(-1,old_solution.block(1));

L2_error_u = difference_vector_u.l2_norm();
L2_error_v = difference_vector_v.l2_norm();*/

difference_vector_d.operator=(displacement);
difference_vector_d.add(-1,old_displacement);
L2_error_d = difference_vector_d.l2_norm();

myfile.open ("l2_diff.txt", std::ofstream::app);
myfile << "d l2: " << L2_error_d/time_step << "  ";

myfile.close();
}

// set a flag if cell is in the vicinity of the membrane
template <int dim>
void CellMot<dim>::on_membrane ()
{
typename DoFHandler<dim>::active_cell_iterator
cell = dof_handler.begin_active(),
endc = dof_handler.end();
for (; cell!=endc; ++cell)
   {
   for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
        {
        Point<dim> delt;
        delt = cell->vertex(v);
        double dist = sqrt(delt[0]*delt[0]+delt[1]*delt[1]+delt[2]*delt[2]);
/*	if (delt[0]>0)
	x = 1;
	else 
	x=0;*/
	x=-delt[0];
        if (dist > 0.8)
           cell->set_user_flag();
        }
    }
}


// RUN EVERYTHING
template <int dim>
void CellMot<dim>::run ()
{

make_grid_and_dofs();
get_params();
on_membrane();
// set initial values

old_solution.operator=(0);
solution = old_solution;
old_sol = old_solution;
old_displacement = displacement;
timestep_number = 0;
time            = 0;

// output initial conditions
output_results_f();

//output_results_u();

// time loop
while (time <= 1000000.0)	
	{
	time += time_step;
	++timestep_number;
	std::cout << "time = " << time << std::endl;	    

	assemble_system_parts ();
        assemble_system ();
        old_sol = old_solution;

        std::cout << "solving u .. ";
//	if (time<0.19)
        solve (solution);
/*	else if (time<0.35)
	solve2 (solution);
	else if (time<0.78)
	solve3 (solution);
	else if (time<1.77)
	solve4 (solution);
	else if (time<2.76)
	solve5 (solution);
	else
	solve6 (solution);*/
	move_mesh();
        makenewkm();

	

	
	// OUTPUT VOLUME OF SHAPE
	std::ofstream myfile;
	myfile.open ("volume.txt", std::ofstream::app);
	myfile << GridTools::volume(triangulation) << std::endl;
	myfile.close();
	
	for(unsigned int i=0;i<displacement.size();i++)
	{
	displacement(i)=sqrt(solution.block(0)(i)*solution.block(0)(i)
			+ solution.block(1)(i)*solution.block(1)(i)
                        + solution.block(2)(i)*solution.block(2)(i));
	old_displacement(i)=sqrt(old_solution.block(0)(i)*old_solution.block(0)(i)
                        + old_solution.block(1)(i)*old_solution.block(1)(i)
                        + old_solution.block(2)(i)*old_solution.block(2)(i));
	}
        old_solution = solution;

        if (timestep_number<10 || timestep_number%10==0)
                {
        	l2_diff();
		output_results_f();
 //       output_results_u();
		}

	}// end time loop
  }// end run



// MAIN function
int main ()
{
  try
    {
      using namespace dealii;

      deallog.depth_console (2);

      CellMot<3> movement("cmm.prm");
      movement.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
