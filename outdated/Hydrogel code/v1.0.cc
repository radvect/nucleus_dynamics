#include <deal.II/grid/reference_cell.h>
#include <deal.II/lac/matrix_out.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <cmath>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_refinement.h>
#include <map>
#include <deal.II/base/derivative_form.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/qprojector.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/differentiation/sd/symengine_math.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>
#include <fstream>
#include <iostream>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/base/mpi.h>
#include <deal.II/grid/filtered_iterator.h>
#include <string>
#include <iomanip>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/fe/mapping_manifold.h>
#include <deal.II/grid/grid_generator.h>
#include <limits>
#include <locale>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/base/function.h>
#include <deal.II/base/table.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/gmsh/utilities.h>
#include <deal.II/opencascade/manifold_lib.h>
#include <deal.II/opencascade/utilities.h>
#include <Standard_Stream.hxx>
#include <TopTools.hxx>
#include <TopoDS_Shape.hxx>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace HP_ALE
{
    using namespace dealii;

    enum class TestCase
    {
        geo1,
        geo2
    };
    static const char *enum_str[] = {"geo1","geo2"};

template <int dim>
class ExtensionStokesVelocity : public Function<dim>
{
public:
  ExtensionStokesVelocity()
    : Function<dim>(dim * 4 + 2)
  {}
  virtual void
  vector_value(const Point<dim> &point, Vector<double> &value) const override;
};

template <int dim>
void
ExtensionStokesVelocity<dim>::vector_value(const Point<dim> &point,
                                           Vector<double> &  value) const
{
  Assert(dim >= 2, ExcNotImplemented());
  //const double extension_rate = 0.05;
  const double x              = point[0];
  const double y              = point[1];
  value[dim + dim + dim + 1]  = -100.*y*(y-0.2);   //case-6, v(x)
  value[dim + dim + dim + 2]  = y * 0.; //case-6, v(y)
  
}


template <int dim>
class InitialDisplacement : public Function<dim>
{
public:
  InitialDisplacement()
    : Function<dim>(dim * 4 + 2)
  {}
  virtual void
  vector_value(const Point<dim> &point, Vector<double> &value) const override;
};

template <int dim>
void
InitialDisplacement<dim>::vector_value(const Point<dim> &point,
                                       Vector<double> &  value) const
{
  Assert(dim >= 2, ExcNotImplemented());

  const double x     = point[0];
  const double x_b   = 4.;
  const double x_i   = 2.;
  const double u_max = 1.;

  value[0] = u_max - u_max * (x - x_i) / (x_b - x_i);
}

    template <int dim>
    class FluidStructureProblem
    {
    public:
        FluidStructureProblem(const unsigned int velocity_degree,
                              const unsigned int pressure_degree,
                              const unsigned int volume_degree,
                              const TestCase &   test_case);
        void run();
        void
        create_dofs_map(const std::string &filename, const ComponentMask mask);

    private:
        enum
        {
            fluid_domain_id,
            hydrogel_domain_id
        };

        using MeshType     = DoFHandler<dim>;
        using CellIterator = typename DoFHandler<dim>::cell_iterator;

        TestCase test_case;

        static bool
        cell_is_in_fluid_domain(
                const typename DoFHandler<dim>::cell_iterator &cell);

        static bool
        cell_is_in_hydrogel_domain(
                const typename DoFHandler<dim>::cell_iterator &cell);

        std::vector<const FiniteElement<dim> *>
        create_stokes_fe_list(const unsigned int velocity_degree,
                              const unsigned int pressure_degree);

        std::vector<const FiniteElement<dim> *>
        create_hydrogel_fe_list(const unsigned int velocity_degree,
                                const unsigned int pressure_degree);

        std::vector<unsigned int>
        create_fe_multiplicities();

        void apply_initial_condition_hp();
        void print_variables();
        void make_grid(const unsigned int n_refinement);
        void set_active_fe_indices();
        void setup_dofs();
        void setup_hp_sparse_matrix(const IndexSet &hp_index_set,
                                    const IndexSet &hp_relevant_set);

        void make_coupling(Table<2, DoFTools::Coupling> &cell_coupling,
                           Table<2, DoFTools::Coupling> &face_coupling,
                           const bool                    print_pattern = false);

        void
        make_boundary_constraints_hp(const IndexSet &hp_relevant_set);

        std::vector<Point<dim>>
        get_physical_face_points(
                const typename DoFHandler<dim>::cell_iterator &cell,
                const unsigned int                             face_no,
                const std::vector<Point<dim - 1>> &            unit_face_points);
        void
        add_interface_constraints(const FESystem<dim> &                      fe,
                                  const types::global_dof_index              row,
                                  const std::vector<types::global_dof_index> cols,
                                  const Tensor<1, dim>                       normal,
                                  const double                   coefficient,
                                  const uint                     line_comp,
                                  const uint                     starting_comp,
                                  const uint                     num_comp,
                                  const Point<dim> &             gsp,
                                  const std::vector<Point<dim>> &fsp,
                                  AffineConstraints<double> &    constraints_flux);

        void
        make_flux_constraints(AffineConstraints<double> &constraints_flux);

        void newton_iteration();
        void output_results(const unsigned int refinement_cycle) const;
        void update_constraints(const IndexSet &hp_relevant_set);
        /* void
         set_interface_dofs_flag(std::vector<unsigned int> &flag);*/
        void
        set_interface_dofs_flag(TrilinosWrappers::MPI::Vector &flag,
                                const IndexSet &hp_relevant_set);
        void compute_cauchy_stress();

        const unsigned int velocity_degree;
        const unsigned int pressure_degree;
        const unsigned int volume_degree;

        MPI_Comm     mpi_communicator;
        const unsigned int n_mpi_processes;
        const unsigned int this_mpi_process;

        parallel::distributed::Triangulation<dim> triangulation;

        FESystem<dim>      stokes_fe;
        FESystem<dim>      hydrogel_fe;

        FE_Q<dim>          volume_fe; // volume fraction phi_s

        hp::FECollection<dim> fe_collection;
        DoFHandler<dim>       dof_handler;

        hp::FECollection<dim> volume_fe_collection;
        DoFHandler<dim>       volume_dof_handler;

        IndexSet volume_locally_owned_dofs;
        IndexSet volume_locally_relevant_dofs;
        IndexSet hp_index_set;
        IndexSet hp_relevant_set;

        AffineConstraints<double> constraints_hp_nonzero; // for fe system
        AffineConstraints<double> constraints_hp; // flux constraints + nonzero
        AffineConstraints<double> constraints_newton_update; // boundary constraints + interface constraints
        AffineConstraints<double> constraints_boundary; // newton update boundary
        AffineConstraints<double> constraints_volume;   // for phi_s
        AffineConstraints<double> constraints_flux;

        std::vector<bool> constrainted_flag;
        SparsityPattern      sparsity_pattern;

        //SparsityPattern      sparsity_pattern;
        TrilinosWrappers::SparseMatrix system_matrix;

        //SparsityPattern      volume_sparsity_pattern;
        TrilinosWrappers::SparseMatrix volume_system_matrix;

        TrilinosWrappers::MPI::Vector solution;         // newton, n+1  t=m
        TrilinosWrappers::MPI::Vector old_solution;     // t=m-1
        TrilinosWrappers::MPI::Vector current_solution; // newton iteration solution n, t=m

        TrilinosWrappers::MPI::Vector dis_solution;         // newton, n+1  t=m
        TrilinosWrappers::MPI::Vector dis_old_solution;     // t=m-1
        TrilinosWrappers::MPI::Vector dis_current_solution; // newton iteration solution n, t=m

        TrilinosWrappers::MPI::Vector system_rhs;
        TrilinosWrappers::MPI::Vector newton_update;
        TrilinosWrappers::MPI::Vector dis_newton_update;

        TrilinosWrappers::MPI::Vector volume_solution;
        TrilinosWrappers::MPI::Vector volume_old_solution;
        TrilinosWrappers::MPI::Vector volume_system_rhs;
        TrilinosWrappers::MPI::Vector dis_volume_solution;
        TrilinosWrappers::MPI::Vector dis_volume_old_solution;

        std::vector<TrilinosWrappers::MPI::Vector> strain;
        std::vector<TrilinosWrappers::MPI::Vector> stress_s;
        std::vector<TrilinosWrappers::MPI::Vector> stress_f;
        TrilinosWrappers::MPI::Vector el_energy;

        std::unique_ptr<MappingQ<dim>> mapping_pointer;

        hp::MappingCollection<dim>     mapping_collection;

        /*std::vector<unsigned int> interface_dofs_flag;*/
        TrilinosWrappers::MPI::Vector  interface_dofs_flag;

        const double viscosity;
        const double vis_BM; // viscousity inside hydrogel
        double mu_s;
        double lambda_s;                 // lame coefficient
        const double beta_0;                   // interface permeability  (tangent)
        const double beta_i;                   // interface permeability  (tangent)
        const double eta;                      // interface permeability  (normal)
        const double xi;                       // friction coe
        const double Ca;
        const double alpha;                    // mesh coe
        double       time_step, old_time_step; //\delta t
        unsigned int timestep_number;
        double       hmin;
        const double phi_s0 = 0.02;

        const FEValuesExtractors::Vector extractor_displacement;      // u
        const FEValuesExtractors::Vector extractor_mesh_velocity;     // vs
        const FEValuesExtractors::Vector extractor_hydrogel_velocity; // vf·
        const FEValuesExtractors::Scalar extractor_hydrogel_pressure; // p2
        const FEValuesExtractors::Vector extractor_stokes_velocity;   // V
        const FEValuesExtractors::Scalar extractor_stokes_pressure;   // P1

        ConditionalOStream pcout;
        TimerOutput        computing_timer;

        // scratch data
        struct ScratchData
        {
            hp::MappingCollection<dim> mapping_collection;
            hp::FEValues<dim>          hp_fe_values;

            FEFaceValues<dim>    stokes_fe_face_values;      //
            FEFaceValues<dim>    hydrogel_fe_face_values;    //
            FESubfaceValues<dim> stokes_fe_subface_values;   // gamma
            FESubfaceValues<dim> hydrogel_fe_subface_values; // gamma

            hp::FEValues<dim>    volume_fe_values;
            FEFaceValues<dim>    volume_fe_face_values;
            FESubfaceValues<dim> volume_fe_subface_values;

            ScratchData(hp::FECollection<dim> &     fe_collection,
                        hp::FECollection<dim> &     volume_fe_collection,
                        hp::MappingCollection<dim>  mapping_collection,
                        const FiniteElement<dim> &  stokes_fe,
                        const FiniteElement<dim> &  hydrogel_fe,
                        const FiniteElement<dim> &  volume_fe,
                        const hp::QCollection<dim> &q_collection,
                        const hp::QCollection<dim> &volume_q_collection,
                        const Quadrature<dim - 1> & face_quadrature,
                        const UpdateFlags &         hp_update_flags,
                        const UpdateFlags &         volume_update_flags,
                        const UpdateFlags &         face_update_flags)
                    : mapping_collection(mapping_collection)
                    , hp_fe_values(mapping_collection,
                                   fe_collection,
                                   q_collection,
                                   hp_update_flags)
                    , stokes_fe_face_values(mapping_collection[0],
                                            stokes_fe,
                                            face_quadrature,
                                            face_update_flags)
                    , hydrogel_fe_face_values(mapping_collection[0],
                                              hydrogel_fe,
                                              face_quadrature,
                                              face_update_flags)
                    , stokes_fe_subface_values(mapping_collection[0],
                                               stokes_fe,
                                               face_quadrature,
                                               face_update_flags)
                    , hydrogel_fe_subface_values(mapping_collection[0],
                                                 hydrogel_fe,
                                                 face_quadrature,
                                                 face_update_flags)
                    , volume_fe_values(mapping_collection,
                                       volume_fe_collection,
                                       volume_q_collection,
                                       volume_update_flags)
                    , volume_fe_face_values(mapping_collection[0],
                                            volume_fe,
                                            face_quadrature,
                                            face_update_flags)
                    , volume_fe_subface_values(mapping_collection[0],
                                               volume_fe,
                                               face_quadrature,
                                               face_update_flags)
            {}

            ScratchData(const ScratchData &scratch_data)
                    : mapping_collection(scratch_data.mapping_collection)
                    , hp_fe_values(scratch_data.mapping_collection,
                                   scratch_data.hp_fe_values.get_fe_collection(),
                                   scratch_data.hp_fe_values.get_quadrature_collection(),
                                   scratch_data.hp_fe_values.get_update_flags())
                    , stokes_fe_face_values(
                            scratch_data.mapping_collection[0],
                            scratch_data.stokes_fe_face_values.get_fe(),
                            scratch_data.stokes_fe_face_values.get_quadrature(),
                            scratch_data.stokes_fe_face_values.get_update_flags())
                    , hydrogel_fe_face_values(
                            scratch_data.mapping_collection[0],
                            scratch_data.hydrogel_fe_face_values.get_fe(),
                            scratch_data.hydrogel_fe_face_values.get_quadrature(),
                            scratch_data.hydrogel_fe_face_values.get_update_flags())
                    , stokes_fe_subface_values(
                            scratch_data.mapping_collection[0],
                            scratch_data.stokes_fe_subface_values.get_fe(),
                            scratch_data.stokes_fe_subface_values.get_quadrature(),
                            scratch_data.stokes_fe_subface_values.get_update_flags())
                    , hydrogel_fe_subface_values(
                            scratch_data.mapping_collection[0],
                            scratch_data.hydrogel_fe_subface_values.get_fe(),
                            scratch_data.hydrogel_fe_subface_values.get_quadrature(),
                            scratch_data.hydrogel_fe_subface_values.get_update_flags())
                    , volume_fe_values(
                            scratch_data.mapping_collection,
                            scratch_data.volume_fe_values.get_fe_collection(),
                            scratch_data.volume_fe_values.get_quadrature_collection(),
                            scratch_data.volume_fe_values.get_update_flags())
                    , volume_fe_face_values(
                            scratch_data.mapping_collection[0],
                            scratch_data.volume_fe_face_values.get_fe(),
                            scratch_data.volume_fe_face_values.get_quadrature(),
                            scratch_data.volume_fe_face_values.get_update_flags())
                    , volume_fe_subface_values(
                            scratch_data.mapping_collection[0],
                            scratch_data.volume_fe_subface_values.get_fe(),
                            scratch_data.volume_fe_subface_values.get_quadrature(),
                            scratch_data.volume_fe_subface_values.get_update_flags())

            {}
        };
        struct PerTaskData
        {
            bool                                 assemble_interface;
            FullMatrix<double>                   cell_matrix;
            Vector<double>                       cell_rhs;
            std::vector<types::global_dof_index> local_dof_indices;

            FullMatrix<double>                   volume_cell_matrix;
            Vector<double>                       volume_cell_rhs;
            std::vector<types::global_dof_index> volume_local_dof_indices;

            // four different kinds of coupling
            FullMatrix<double> interface_matrix1; //(stokes,stokes)
            Vector<double>     interface_rhs1;
            FullMatrix<double> interface_matrix2; //(stokes,hydrogel)
            Vector<double>     interface_rhs2;
            FullMatrix<double> interface_matrix3; //(hydrogel,stokes)
            Vector<double>     interface_rhs3;
            FullMatrix<double> interface_matrix4; //(hydrogel,hydrogel)
            Vector<double>     interface_rhs4;
            std::vector<types::global_dof_index> neighbor_dof_indices;
        };
        void
        solver(TrilinosWrappers::SparseMatrix &matrix,
                                           TrilinosWrappers::MPI::Vector &sol,
                                           TrilinosWrappers::MPI::Vector &rhs);
        void
        local_assemble_hp(
                const typename DoFHandler<dim>::active_cell_iterator &cell,
                ScratchData &                                         scratch,
                PerTaskData &                                         copy_data,
                const bool update_matrix = true);
        void
        copy_local_to_global_hp(const PerTaskData &copy_data,
                                const bool         update_matrix = true);
        void
        local_assemble_volume(
                const typename DoFHandler<dim>::active_cell_iterator &cell,
                ScratchData &                                         scratch,
                PerTaskData &                                         copy_data);
        void
        copy_local_to_global_volume(const PerTaskData &copy_data);

        void
        assemble_system_workstream(const bool update_matrix = true);
        void
        assemble_volume_system_workstream();
        void
        solve_volume();
    };

    template <int dim>
    std::vector<const FiniteElement<dim> *>
    FluidStructureProblem<dim>::create_hydrogel_fe_list(
            const unsigned int velocity_degree,
            const unsigned int pressure_degree)
    {
        std::vector<const FiniteElement<dim> *> fe_list;
        fe_list.push_back(new FE_Q<dim>(velocity_degree)); //(dim) displacement
        fe_list.push_back(
                new FE_Q<dim>(velocity_degree)); //(dim) mesh velocity (solid velocity)
        fe_list.push_back(
                new FE_Q<dim>(velocity_degree)); //(dim) hydrogel fluid velocity v_f
        fe_list.push_back(new FE_Q<dim>(pressure_degree)); //(1)   hydrogel pressure
        fe_list.push_back(new FE_Nothing<dim>()); //(dim) outer fluid velocity V
        fe_list.push_back(new FE_Nothing<dim>()); //(1)   pressure P
        return fe_list;
    }

    template <int dim>
    std::vector<const FiniteElement<dim> *>
    FluidStructureProblem<dim>::create_stokes_fe_list(
            const unsigned int velocity_degree,
            const unsigned int pressure_degree)
    {
        std::vector<const FiniteElement<dim> *> fe_list;
        fe_list.push_back(new FE_Q<dim>(velocity_degree)); //(dim) displacement
        fe_list.push_back(
                new FE_Nothing<dim>()); //(dim) mesh velocity (solid velocity)
        fe_list.push_back(
                new FE_Nothing<dim>()); //(dim) hydrogel fluid velocity v_f
        fe_list.push_back(new FE_Nothing<dim>()); //(1)   hydrogel pressure
        fe_list.push_back(
                new FE_Q<dim>(velocity_degree)); //(dim) outer fluid velocity V_f
        fe_list.push_back(new FE_Q<dim>(pressure_degree)); //(1)   pressure P
        return fe_list;
    }

    template <int dim>
    std::vector<unsigned int>
    FluidStructureProblem<dim>::create_fe_multiplicities()
    {
        // correspond to fe list
        std::vector<unsigned int> multiplicities;
        multiplicities.push_back(dim);
        multiplicities.push_back(dim);
        multiplicities.push_back(dim);
        multiplicities.push_back(1);
        multiplicities.push_back(dim);
        multiplicities.push_back(1);
        return multiplicities;
    }

    template <int dim>
    FluidStructureProblem<dim>::FluidStructureProblem(
            const unsigned int velocity_degree,
            const unsigned int pressure_degree,
            const unsigned int volume_degree,
            const TestCase &   test_case)
            : test_case(test_case)
            , velocity_degree(velocity_degree)
            , pressure_degree(pressure_degree)
            , volume_degree(volume_degree)
            , mpi_communicator(MPI_COMM_WORLD)
            , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
            , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
           , triangulation(mpi_communicator,
                            typename Triangulation<dim>::MeshSmoothing(
                                    Triangulation<dim>::smoothing_on_refinement |
                                    Triangulation<dim>::smoothing_on_coarsening))
            , stokes_fe(create_stokes_fe_list(velocity_degree, pressure_degree),
                        create_fe_multiplicities())
            , hydrogel_fe(create_hydrogel_fe_list(velocity_degree, pressure_degree),
                          create_fe_multiplicities())
            , volume_fe(volume_degree)
            , dof_handler(triangulation)
            , volume_dof_handler(triangulation)
            , viscosity(1)
            , vis_BM(1)
            , mu_s(1)//shear stress 333.3333
            , lambda_s(1)//lamei xishu 222.2222
            , beta_0(2e-2)//
            , beta_i(2e-2)//PERMEABILITY ANDF SLIP beta mu /l_0
            , eta(2e-2)// dimensionless eta, eta mu /l_0
            , xi(4.5e4)//dimensionless xi
            , Ca((3*1e-3)/(2000*250)) //capilary number
            , alpha(10)
            , time_step(1e-7)
            , old_time_step(time_step)
            , timestep_number(0)
            , extractor_displacement(0)
            , extractor_mesh_velocity(dim)
            , extractor_hydrogel_velocity(dim + dim)
            , extractor_hydrogel_pressure(dim + dim + dim)
            , extractor_stokes_velocity(dim + dim + dim + 1)
            , extractor_stokes_pressure(dim + dim + dim + 1 + dim)
            , pcout(std::cout,
                    (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
            , computing_timer(MPI_COMM_WORLD,
                              pcout,
                              TimerOutput::summary,
                              TimerOutput::wall_times)
    {
        // active fe id:
        // 0: fluid domain
        // 1: hydrogel domain
        fe_collection.push_back(stokes_fe);
        fe_collection.push_back(hydrogel_fe);

        volume_fe_collection.push_back(FE_Nothing<dim>());

        volume_fe_collection.push_back(volume_fe);
        print_variables();

        switch (test_case)
        {
            case TestCase::geo1:
            {
                const bool         use_on_all_cells = true;
                const unsigned int mapping_degree   = velocity_degree;
                mapping_pointer =
                        std::make_unique<MappingQ<dim>>(mapping_degree, use_on_all_cells);

                break;
            }
            case TestCase::geo2:
            {
                const bool         use_on_all_cells = true;
                const unsigned int mapping_degree   = velocity_degree;

                mapping_pointer =
                        std::make_unique<MappingQ<dim>>(mapping_degree, use_on_all_cells);

                break;
            }
            default:
                Assert(false, ExcNotImplemented());
        }
        mapping_collection.push_back(*mapping_pointer);
        mapping_collection.push_back(*mapping_pointer);
    }

    template <int dim>
    bool
    FluidStructureProblem<dim>::cell_is_in_fluid_domain(
            const typename DoFHandler<dim>::cell_iterator &cell)
    {
        return (cell->material_id() == fluid_domain_id);
    }

    template <int dim>
    bool
    FluidStructureProblem<dim>::cell_is_in_hydrogel_domain(
            const typename DoFHandler<dim>::cell_iterator &cell)
    {
        return (cell->material_id() == hydrogel_domain_id);
    }

    template <int dim>
    void
    FluidStructureProblem<dim>::print_variables()
    {
        pcout << "viscosity: " << viscosity << std::endl
              << "vis_BM   : " << vis_BM << std::endl
              << "mu_s     : " << mu_s << std::endl
              << "lambda_s : " << lambda_s << std::endl
              << "beta_0   : " << beta_0 << std::endl
              << "beta_i   : " << beta_i << std::endl
              << "eta      : " << eta << std::endl
              << "xi       : " << xi << std::endl
              << "alpha    : " << alpha << std::endl;
    }

    template <int dim>
    void
    FluidStructureProblem<dim>::make_grid(const unsigned int n_refinement)
    {
        TimerOutput::Scope timing_section(computing_timer, "make_grid");
        switch (test_case)
        {
            case TestCase::geo1:
            {
				const Point<2> center1(2, 1./std::sqrt(3.));
                
                const double   radius = 2.0/ std::sqrt(3.);
                const SphericalManifold<2> manifold1(center1);
                GridIn<2> gridin;
                gridin.attach_triangulation(triangulation);
                std::ifstream f("case1.msh");
                gridin.read_msh(f);

                triangulation.reset_all_manifolds();
                triangulation.set_all_manifold_ids(0);

                for (const auto &cell : triangulation.cell_iterators())
                {
                    for (const auto &face : cell->face_iterators())
                    {
                        bool face_at_sphere_boundary = true;
                        for (const auto v : face->vertex_indices())
                        {
                            const double distance_from_center1 =
                                    center1.distance(face->vertex(v));
                            if (std::fabs(distance_from_center1 - radius) >
                                1e-5 * radius) 
                            {
                                face_at_sphere_boundary = false;
                            }
                        }
                        if (face_at_sphere_boundary)
                            face->set_all_manifold_ids(1);
                    }

                }
                triangulation.set_manifold (1, manifold1);
    
                // set material id:
                for (const auto &cell : dof_handler.active_cell_iterators())
                {
                    if (cell->material_id() == 1)
                        cell->set_material_id(hydrogel_domain_id);
                    else
                        cell->set_material_id(fluid_domain_id);
                }
                triangulation.refine_global(n_refinement);
                break;
            }

            case TestCase::geo2:
            {
                const Point<2> center1(2.4, 0.5);
                const Point<2> center2(4.0, 0.5);
                const double   radius1 = 0.2;
                const double   radius2 = 0.2;
                const SphericalManifold<2> manifold1(center1);
                const SphericalManifold<2> manifold2(center2);
                GridIn<2> gridin;
                gridin.attach_triangulation(triangulation);
                std::ifstream f("case2.msh");
                gridin.read_msh(f);
                triangulation.reset_all_manifolds();
                triangulation.set_all_manifold_ids(0);
                for (const auto &cell : triangulation.cell_iterators())
                {
                    for (const auto &face : cell->face_iterators())
                    {
                        bool face_at_sphere_boundary = true;
                        for (const auto v : face->vertex_indices())
                        {
                            const double distance_from_center1 =
                                    center1.distance(face->vertex(v));
                            if (std::fabs(distance_from_center1 - radius1) >
                                1e-5 * radius1)
                            {
                                face_at_sphere_boundary = false;
                            }
                        }
                        if (face_at_sphere_boundary)
                            face->set_all_manifold_ids(1);
                    }
                    for (const auto &face : cell->face_iterators())
                    {
                        bool face_at_sphere_boundary = true;
                        for (const auto v : face->vertex_indices())
                        {
                            const double distance_from_center2 =
                                    center2.distance(face->vertex(v));

                            if (std::fabs(distance_from_center2 - radius2) >
                                1e-5 * radius2)
                            {
                                face_at_sphere_boundary = false;
                            }
                        }
                        if (face_at_sphere_boundary)
                            face->set_all_manifold_ids(2);
                    }
                }
                triangulation.set_manifold (1, manifold1);
                triangulation.set_manifold (2, manifold2);
                // set material id:
                for (const auto &cell : dof_handler.active_cell_iterators())
                {
                    if (cell->material_id() == 1)
                        cell->set_material_id(hydrogel_domain_id);
                    else
                        cell->set_material_id(fluid_domain_id);
                }
                triangulation.refine_global(n_refinement);
                break;
            }

            default:
            AssertThrow(false, ExcNotImplemented());
        }


         hmin =
      GridTools::minimal_cell_diameter(triangulation) / std::sqrt(1. * dim);
      const double hmax =
      GridTools::maximal_cell_diameter(triangulation) / std::sqrt(1. * dim);
    pcout << " hmin = " << hmin << " hmax = " << hmax << std::endl;
    }

    template <int dim>
    void
    FluidStructureProblem<dim>::set_active_fe_indices()
    {
        typename DoFHandler<dim>::active_cell_iterator
                cell_hp     = dof_handler.begin_active(),
                cell_volume = volume_dof_handler.begin_active(), endc = dof_handler.end();

        for (; cell_hp != endc; ++cell_volume, ++cell_hp)
        {
            if (cell_hp->is_locally_owned())
            {
                if (cell_is_in_fluid_domain(cell_hp))
                {
                    cell_hp->set_active_fe_index(0);
                    cell_volume->set_active_fe_index(0);
                }
                else if (cell_is_in_hydrogel_domain(cell_hp))
                {
                    cell_hp->set_active_fe_index(1);
                    cell_volume->set_active_fe_index(1);
                }
                else
                        Assert(false, ExcNotImplemented());
            }
        }
    }

    template <int dim>
    void
    FluidStructureProblem<dim>::make_boundary_constraints_hp(const IndexSet &hp_relevant_set)
    {
        TimerOutput::Scope timing_section(computing_timer, "boundary constraints");
        const unsigned int n_components_total = fe_collection.n_components();
        constraints_boundary.clear();
        constraints_hp_nonzero.clear();
        constraints_boundary.reinit(hp_relevant_set);
        constraints_hp_nonzero.reinit(hp_relevant_set);
        DoFTools::make_hanging_node_constraints(dof_handler, constraints_boundary);
        DoFTools::make_hanging_node_constraints(dof_handler,
                                                constraints_hp_nonzero);
        switch (test_case)
        {
  
            case TestCase::geo1:
            {
                {
                    // left boundary (id 0):
                    // stokes_vel(0)= parabolic, stokes_vel(1)=0, displacement(0,1) = (0,0)
                    // component mask for velocity u, stokes_vel(0)  stokes_vel(1)
                    ComponentMask stokes_v_component_mask(n_components_total, false);
                    stokes_v_component_mask.set(
                            extractor_stokes_velocity.first_vector_component, true);
                    stokes_v_component_mask.set(
                            extractor_stokes_velocity.first_vector_component + 1, true);
                    
                    VectorTools::interpolate_boundary_values(
                            *mapping_pointer,
                            dof_handler,
                            0,
                            ExtensionStokesVelocity<dim>(),
                            constraints_hp_nonzero,
                            stokes_v_component_mask);

                 // newton iteration boundary for v
                    VectorTools::interpolate_boundary_values(
                            *mapping_pointer,
                            dof_handler,
                            0,
                            Functions::ZeroFunction<dim>(n_components_total),
                            constraints_boundary,
                            stokes_v_component_mask);

                    //displacement(0,1) = 0
                    ComponentMask non_zero_component_mask(n_components_total, false);

                    non_zero_component_mask.set(
                            extractor_displacement.first_vector_component, true);
                    non_zero_component_mask.set(
                            extractor_displacement.first_vector_component + 1, true);

                    VectorTools::interpolate_boundary_values(
                            *mapping_pointer,
                            dof_handler,
                            0,
                            Functions::ZeroFunction<dim>(n_components_total),
                            constraints_hp_nonzero,
                            non_zero_component_mask);

                    // newton update set to 0 correspondingly
                    // displacement u(0,1) = 0
                    ComponentMask zero_component_mask(n_components_total, false);
                    zero_component_mask.set(
                            extractor_displacement.first_vector_component, true);
                    zero_component_mask.set(
                            extractor_displacement.first_vector_component + 1, true);


                    VectorTools::interpolate_boundary_values(
                            *mapping_pointer,
                            dof_handler,
                            0,
                            Functions::ZeroFunction<dim>(n_components_total),
                            constraints_boundary,
                            zero_component_mask);
                }
                {
                    // right boundary (id 1): displacement(0,1) = 0
                    VectorTools::interpolate_boundary_values(
                            *mapping_pointer,
                            dof_handler,
                            1,
                            Functions::ZeroFunction<dim>(n_components_total),
                            constraints_boundary,
                            fe_collection.component_mask(extractor_displacement));
                    VectorTools::interpolate_boundary_values(
                            *mapping_pointer,
                            dof_handler,
                            1,
                            Functions::ZeroFunction<dim>(n_components_total),
                            constraints_hp_nonzero,
                            fe_collection.component_mask(extractor_displacement));
                }
                {
                  //no-slip
                   //stokes_vel(0,1) = 0,  displacement u(0,1) = 0,
                    ComponentMask component_mask(n_components_total, false);
                    component_mask.set(
                            extractor_stokes_velocity.first_vector_component, true);
                    component_mask.set(
                            extractor_stokes_velocity.first_vector_component + 1, true);
                    component_mask.set(extractor_displacement.first_vector_component, true);
                    component_mask.set(extractor_displacement.first_vector_component +
                                       1, true);

                    const auto zero_function =
                            Functions::ZeroFunction<dim>(n_components_total);
                    const std::map<types::boundary_id, const Function<dim> *>
                            function_map = {{2, &zero_function} };
                    VectorTools::interpolate_boundary_values(*mapping_pointer,
                                                             dof_handler,
                                                             function_map,
                                                             constraints_hp_nonzero,
                                                             component_mask);
                    VectorTools::interpolate_boundary_values(*mapping_pointer,
                                                             dof_handler,
                                                             function_map,
                                                             constraints_boundary,
                                                             component_mask);
                }

                {
                    // The middle edge(id = 3):
                    //  mesh_vel(0,1)=0; (vs)  displacement(1,0)=0  (u)
                    // hydro_vel(0,1)=0  vf

                    ComponentMask component_mask(n_components_total, false);
                    component_mask.set(
                            extractor_displacement.first_vector_component, true);
                    component_mask.set(
                            extractor_displacement.first_vector_component+1, true);
                    component_mask.set(
                            extractor_hydrogel_velocity.first_vector_component, true);
                    component_mask.set(
                            extractor_hydrogel_velocity.first_vector_component + 1, true);
                    component_mask.set(
                            extractor_mesh_velocity.first_vector_component, true);
                    component_mask.set(
                            extractor_mesh_velocity.first_vector_component + 1, true);

                    VectorTools::interpolate_boundary_values(
                            *mapping_pointer,
                            dof_handler,
                            3,
                            Functions::ZeroFunction<dim>(n_components_total),
                            constraints_hp_nonzero,
                            component_mask);

                    // newton update set to 0 correspondingly
                    VectorTools::interpolate_boundary_values(
                            *mapping_pointer,
                            dof_handler,
                            3,
                            Functions::ZeroFunction<dim>(n_components_total),
                            constraints_boundary,
                            component_mask);
                }
                break;
            } //
            case TestCase::geo2:            {
                {
                    // left boundary (id 0):inlet v=v_0

                    ComponentMask stokes_v_component_mask(n_components_total, false);
                    stokes_v_component_mask.set(
                            extractor_stokes_velocity.first_vector_component, true);
                    stokes_v_component_mask.set(
                            extractor_stokes_velocity.first_vector_component + 1, true);

                    VectorTools::interpolate_boundary_values(
                            *mapping_pointer,
                            dof_handler,
                            0,
                            ExtensionStokesVelocity<dim>(),
                            constraints_hp_nonzero,
                            stokes_v_component_mask);

                    // newton iteration boundary for v
                    VectorTools::interpolate_boundary_values(
                            *mapping_pointer,
                            dof_handler,
                            0,
                            Functions::ZeroFunction<dim>(n_components_total),
                            constraints_boundary,
                            stokes_v_component_mask);

                    //displacement(0,1) = 0
                    ComponentMask non_zero_component_mask(n_components_total, false);

                    non_zero_component_mask.set(
                            extractor_displacement.first_vector_component, true);
                    non_zero_component_mask.set(
                            extractor_displacement.first_vector_component + 1, true);

                    VectorTools::interpolate_boundary_values(
                            *mapping_pointer,
                            dof_handler,
                            0,
                            Functions::ZeroFunction<dim>(n_components_total),
                            constraints_hp_nonzero,
                            non_zero_component_mask);

                    // newton update set to 0 correspondingly
                    // displacement u(0,1) = 0
                    ComponentMask zero_component_mask(n_components_total, false);
                    zero_component_mask.set(
                            extractor_displacement.first_vector_component, true);
                    zero_component_mask.set(
                            extractor_displacement.first_vector_component + 1, true);


                    VectorTools::interpolate_boundary_values(
                            *mapping_pointer,
                            dof_handler,
                            0,
                            Functions::ZeroFunction<dim>(n_components_total),
                            constraints_boundary,
                            zero_component_mask);
                }
                {
                    // right boundary (id 1): displacement(0,1) = 0
                    VectorTools::interpolate_boundary_values(
                            *mapping_pointer,
                            dof_handler,
                            1,
                            Functions::ZeroFunction<dim>(n_components_total),
                            constraints_boundary,
                            fe_collection.component_mask(extractor_displacement));
                    VectorTools::interpolate_boundary_values(
                            *mapping_pointer,
                            dof_handler,
                            1,
                            Functions::ZeroFunction<dim>(n_components_total),
                            constraints_hp_nonzero,
                            fe_collection.component_mask(extractor_displacement));
                }
                {
                    //symmetry condition
                    //stokes_vel(1) = 0, displacement u(1) = 0, vf_y=vs_y=0,
                    ComponentMask component_mask(n_components_total, false);
                    component_mask.set(
                            extractor_stokes_velocity.first_vector_component + 1, true);
                    component_mask.set(extractor_displacement.first_vector_component +
                                       1, true);
                    component_mask.set(
                            extractor_hydrogel_velocity.first_vector_component + 1, true);
                    component_mask.set(
                            extractor_mesh_velocity.first_vector_component + 1, true);

                    const auto zero_function =
                            Functions::ZeroFunction<dim>(n_components_total);
                    const std::map<types::boundary_id, const Function<dim> *>
                            function_map = {{2, &zero_function} };
                    VectorTools::interpolate_boundary_values(*mapping_pointer,
                                                             dof_handler,
                                                             function_map,
                                                             constraints_hp_nonzero,
                                                             component_mask);
                    VectorTools::interpolate_boundary_values(*mapping_pointer,
                                                             dof_handler,
                                                             function_map,
                                                             constraints_boundary,
                                                             component_mask);
                }
                {
                    // The middle edge(id = 3):
                    //  mesh_vel(0,1)=0; (vs)  displacement(0,1)=0  (u)
                    // hydro_vel(0,1)=0  vf
                    ComponentMask component_mask(n_components_total, false);
                    component_mask.set(
                            extractor_displacement.first_vector_component, true);
                    component_mask.set(
                            extractor_displacement.first_vector_component+1, true);
                    component_mask.set(
                            extractor_hydrogel_velocity.first_vector_component, true);
                    component_mask.set(
                            extractor_hydrogel_velocity.first_vector_component + 1, true);
                    component_mask.set(
                            extractor_mesh_velocity.first_vector_component, true);
                    component_mask.set(
                            extractor_mesh_velocity.first_vector_component + 1, true);

                    VectorTools::interpolate_boundary_values(
                            *mapping_pointer,
                            dof_handler,
                            3,
                            Functions::ZeroFunction<dim>(n_components_total),
                            constraints_hp_nonzero,
                            component_mask);

                    // newton update set to 0 correspondingly
                    VectorTools::interpolate_boundary_values(
                            *mapping_pointer,
                            dof_handler,
                            3,
                            Functions::ZeroFunction<dim>(n_components_total),
                            constraints_boundary,
                            component_mask);
                }
                break;
            } //
            default:
                Assert(false, ExcNotImplemented());
        }
        constraints_boundary.close();
        constraints_hp_nonzero.close();
    }

    template <int dim>
    void
    FluidStructureProblem<dim>::setup_dofs()
    {
        TimerOutput::Scope timing_section(computing_timer, "setup_dofs");
        const unsigned int n_components_total = fe_collection.n_components();
        set_active_fe_indices();
        {
            volume_dof_handler.distribute_dofs(volume_fe_collection);
            volume_locally_owned_dofs = volume_dof_handler.locally_owned_dofs();
            volume_locally_relevant_dofs =
                    DoFTools::extract_locally_relevant_dofs(volume_dof_handler);

            // volume fraction constraints
            constraints_volume.clear();
            constraints_volume.reinit(volume_locally_relevant_dofs);
            DoFTools::make_hanging_node_constraints(volume_dof_handler,
                                                    constraints_volume);
            constraints_volume.close();

            TrilinosWrappers::SparsityPattern sp(volume_locally_owned_dofs,
                                                 volume_locally_owned_dofs,
                                                 volume_locally_relevant_dofs,
                                                 MPI_COMM_WORLD);

            DoFTools::make_sparsity_pattern(volume_dof_handler,
                                            sp,
                                            constraints_volume,
                                            false,
                                            Utilities::MPI::this_mpi_process(
                                                    MPI_COMM_WORLD));
            sp.compress();

            volume_system_matrix.reinit(sp);
            volume_system_rhs.reinit(volume_locally_owned_dofs,
                                     //volume_locally_relevant_dofs,
                                     MPI_COMM_WORLD);
                                     //false);

            //volume constraints
            volume_solution.reinit(volume_locally_relevant_dofs,
                                   MPI_COMM_WORLD);
            /*volume_solution.reinit(volume_locally_owned_dofs,
                                   volume_locally_relevant_dofs,
                                   mpi_communicator);*/
            volume_old_solution.reinit(volume_solution);

            dis_volume_solution.reinit(volume_locally_owned_dofs,
                                       MPI_COMM_WORLD);
            dis_volume_old_solution.reinit(dis_volume_solution);

            VectorTools::interpolate(*mapping_pointer,
                                     volume_dof_handler,
                                     Functions::ConstantFunction<dim>(phi_s0),
                                     dis_volume_solution);

            constraints_volume.distribute(dis_volume_solution);
            volume_solution = dis_volume_solution;
            dis_volume_old_solution = dis_volume_solution;
            volume_old_solution = volume_solution;

            //std::vector<Vector<double>> stress_s
            {
                const unsigned int n_stress_components = 6;
                stress_f.resize(n_stress_components);
                for(unsigned int d = 0; d < n_stress_components; ++d)
                {
                    //stress_f[d].reinit(volume_dof_handler.n_dofs());
                    stress_f[d].reinit(volume_locally_owned_dofs, MPI_COMM_WORLD);
                }
            }
            {
                const unsigned int n_stress_components = 6;
                stress_s.resize(n_stress_components);
                for(unsigned int d = 0; d < n_stress_components; ++d)
                {
                    //stress_s[d].reinit(volume_dof_handler.n_dofs());
                    stress_s[d].reinit(volume_locally_owned_dofs, MPI_COMM_WORLD);
                }

            }
            {
                const unsigned int n_stress_components = 6;
                strain.resize(n_stress_components);
                for(unsigned int d = 0; d < n_stress_components; ++d)
                {
                    //strain[d].reinit(volume_dof_handler.n_dofs());
                    strain[d].reinit(volume_locally_owned_dofs, MPI_COMM_WORLD);
                }

            }
            {
                el_energy.reinit(volume_locally_owned_dofs, MPI_COMM_WORLD);
            }
        }

        {
            dof_handler.distribute_dofs(fe_collection);
            // renumbering
            DoFRenumbering::Cuthill_McKee(dof_handler);
            std::vector<unsigned int> block_component(n_components_total, 0);
            for (unsigned int d = dim; d < n_components_total; ++d)
            {
                if (d < 2 * dim)
                    block_component[d] = 1;
                else if (d < 3 * dim)
                    block_component[d] = 2;
                else if (d == 3 * dim)
                    block_component[d] = 3;
                else if (d < 4 * dim + 1)
                    block_component[d] = 4;
                else // if (d == 4 * dim +1)
                    block_component[d] = 5;
            }
            for (unsigned int d = 0; d < block_component.size(); ++d)
                pcout << block_component[d] << " block " << std::endl;
            // DoFRenumbering::component_wise(dof_handler, block_component);

            const std::vector<types::global_dof_index> dofs_per_block =
                    DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
            const unsigned int n_displacement = dofs_per_block[0];
            const unsigned int n_mesh_vel     = dofs_per_block[1];
            const unsigned int n_vf           = dofs_per_block[2];
            const unsigned int n_p_hydrogel   = dofs_per_block[3];
            const unsigned int n_V            = dofs_per_block[4];
            const unsigned int n_P            = dofs_per_block[5];
            // const unsigned int n_phi_s = volume_dof_handler.n_dofs();

            pcout << "   Number of active cells: "
                  << triangulation.n_active_cells() << std::endl
                  << "   Number of degrees of freedom for the system: "
                  << dof_handler.n_locally_owned_dofs() << std::endl
                  << "(" << n_displacement << "+" << n_mesh_vel << "+" << n_vf
                  << "+" << n_p_hydrogel << "+" << n_V << "+" << n_P << ")"
                  << "   Number of degrees of freedom for the volume fraction: "
                  << volume_dof_handler.n_locally_owned_dofs() << std::endl;

            hp_index_set = dof_handler.locally_owned_dofs();
            hp_relevant_set = DoFTools::extract_locally_relevant_dofs(dof_handler);

            /*solution.reinit(hp_relevant_set,
                            MPI_COMM_WORLD);*/

            solution.reinit(hp_index_set,
                            hp_relevant_set,
                            MPI_COMM_WORLD);
            old_solution.reinit(solution);
            current_solution.reinit(solution);
            newton_update.reinit(solution);

            system_rhs.reinit(hp_index_set,
                              MPI_COMM_WORLD);

            /*system_rhs.reinit(hp_index_set,
                   hp_relevant_set,
                              MPI_COMM_WORLD,
                              true);*/
            dis_solution.reinit(system_rhs);
            /*dis_solution.reinit(hp_index_set,
                                MPI_COMM_WORLD);*/
            dis_old_solution.reinit(dis_solution);
            dis_current_solution.reinit(dis_solution);
            dis_newton_update.reinit(dis_solution);
        }

        make_boundary_constraints_hp(hp_relevant_set);
        constraints_volume.close();

        update_constraints(hp_relevant_set);
        setup_hp_sparse_matrix(hp_index_set, hp_relevant_set);

        {
            apply_initial_condition_hp();

            constraints_hp.distribute(dis_solution);
            solution = dis_solution;
            dis_old_solution = dis_solution;
            old_solution = solution;
            //old_solution = dis_old_solution;
            pcout << " initial sol l2: " << dis_solution.l2_norm() << std::endl;
        }
    }

    template <int dim>
    void FluidStructureProblem<dim>::make_coupling(
            Table<2, DoFTools::Coupling> &cell_coupling,
            Table<2, DoFTools::Coupling> &face_coupling,
            const bool                    print_pattern)
    {
        cell_coupling.reinit(fe_collection.n_components(),
                             fe_collection.n_components());
        face_coupling.reinit(fe_collection.n_components(),
                             fe_collection.n_components());

        for (unsigned int c = 0; c < fe_collection.n_components(); ++c)
            for (unsigned int d = 0; d < fe_collection.n_components(); ++d)
            {
                if (((c < dim) && (d < 2 * dim)) ||
                    (((c > dim - 1) && (c < 2 * dim)) && (d < (3 * dim + 1))) ||
                    (((c > dim + 1) && (c < 3 * dim)) &&
                     ((d > dim - 1) && (d < 3 * dim + 1))) ||
                    ((c == 3 * dim) && ((d > dim - 1) && (d < 3 * dim))) ||
                    (((c > 3 * dim) && (c < 4 * dim + 1)) && (d > 3 * dim)) ||
                    ((c == 4 * dim + 1) && ((d > 3 * dim) && (d < 4 * dim + 1))))
                    cell_coupling[c][d] = DoFTools::always;

                // face coupling,boundary condition #2

                if ((c >= extractor_stokes_velocity.first_vector_component &&
                     c < extractor_stokes_velocity.first_vector_component + dim) &&
                    (d >= extractor_hydrogel_velocity.first_vector_component &&
                     d < extractor_hydrogel_velocity.first_vector_component + dim))
                    face_coupling[c][d] = DoFTools::always; // (V, vf) 3

                if ((d >= extractor_stokes_velocity.first_vector_component &&
                     d < extractor_stokes_velocity.first_vector_component + dim) &&
                    (c >= extractor_hydrogel_velocity.first_vector_component &&
                     c < extractor_hydrogel_velocity.first_vector_component + dim))
                    face_coupling[c][d] = DoFTools::always; // (vf, V) 2

                if ((c >= extractor_stokes_velocity.first_vector_component &&
                     c < extractor_stokes_velocity.first_vector_component + dim) &&
                    (d >= extractor_stokes_velocity.first_vector_component &&
                     d < extractor_stokes_velocity.first_vector_component + dim))
                    face_coupling[c][d] = DoFTools::always; // (V, V) 1

                if ((c >= extractor_mesh_velocity.first_vector_component &&
                     c < extractor_mesh_velocity.first_vector_component + dim) &&
                    (d >= extractor_mesh_velocity.first_vector_component &&
                     d < extractor_mesh_velocity.first_vector_component + dim))
                    face_coupling[c][d] = DoFTools::always; // (vs, vs) 7

                if ((c >= extractor_hydrogel_velocity.first_vector_component &&
                     c < extractor_hydrogel_velocity.first_vector_component + dim) &&
                    (d >= extractor_mesh_velocity.first_vector_component &&
                     d < extractor_mesh_velocity.first_vector_component + dim))
                    face_coupling[c][d] = DoFTools::always; // (vf, vs) 6

                if ((d >= extractor_hydrogel_velocity.first_vector_component &&
                     d < extractor_hydrogel_velocity.first_vector_component + dim) &&
                    (c >= extractor_mesh_velocity.first_vector_component &&
                     c < extractor_mesh_velocity.first_vector_component + dim))
                    face_coupling[c][d] = DoFTools::always; // (vs, vf) 5

                if ((c >= extractor_hydrogel_velocity.first_vector_component &&
                     c < extractor_hydrogel_velocity.first_vector_component + dim) &&
                    (d >= extractor_hydrogel_velocity.first_vector_component &&
                     d < extractor_hydrogel_velocity.first_vector_component + dim))
                    face_coupling[c][d] = DoFTools::always; // (vf, vf) 4

            }

        if (print_pattern)
        {
            // visualize coupling
            FullMatrix<double> cell_coupling_mat(fe_collection.n_components(),
                                                 fe_collection.n_components());
            FullMatrix<double> face_coupling_mat(fe_collection.n_components(),
                                                 fe_collection.n_components());
            for (unsigned int c = 0; c < fe_collection.n_components(); ++c)
            {
                for (unsigned int d = 0; d < fe_collection.n_components(); ++d)
                {
                    if (cell_coupling[c][d] == DoFTools::always)
                        cell_coupling_mat(c, d) = 1;
                    if (face_coupling[c][d] == DoFTools::always)
                        face_coupling_mat(c, d) = 1;
                }
            }
            cell_coupling_mat.print(std::cout);
            pcout << "face coupling:" << std::endl;
            face_coupling_mat.print(std::cout);
        }
    }

    template <int dim>
    void
    FluidStructureProblem<dim>::setup_hp_sparse_matrix(const IndexSet &hp_index_set,
                                                       const IndexSet &hp_relevant_set)
    {
        TimerOutput::Scope timing_section(computing_timer, "setup_hp_sparse_matrix");
        Table<2, DoFTools::Coupling> cell_coupling, face_coupling;
        const bool print_coupling_pattern = false;
        make_coupling(cell_coupling, face_coupling, print_coupling_pattern);

        TrilinosWrappers::SparsityPattern dsp(hp_index_set,
                                             hp_index_set,
                                             hp_relevant_set,
                                             MPI_COMM_WORLD);

        DoFTools::make_flux_sparsity_pattern(dof_handler,
                                             dsp,
                                             constraints_newton_update,
                                             true,
                                             cell_coupling,
                                             face_coupling,
                                             Utilities::MPI::this_mpi_process(
                                                MPI_COMM_WORLD));
        dsp.compress();
        system_matrix.reinit(dsp);

    }

    template <int dim>
    std::vector<Point<dim>>
    FluidStructureProblem<dim>::get_physical_face_points(
            const typename DoFHandler<dim>::cell_iterator &cell,
            const unsigned int                             face_no,
            const std::vector<Point<dim - 1>> &            unit_face_points)
    {
        std::vector<Point<dim>> return_vals(unit_face_points.size());
        Quadrature<dim - 1>     tmp_q(unit_face_points);
        std::vector<Point<dim>> tmp_p(tmp_q.size());
        QProjector<dim>::project_to_face(ReferenceCells::Quadrilateral,
                                         tmp_q,
                                         face_no,
                                         tmp_p);

        for (unsigned int i = 0; i < unit_face_points.size(); ++i)
        {
            return_vals[i] =
                    mapping_pointer->transform_unit_to_real_cell(cell, tmp_p[i]);
        }
        return return_vals;
    }

    template <int dim>
    void
    FluidStructureProblem<dim>::add_interface_constraints(
            const FESystem<dim> &                      fe,
            const types::global_dof_index              row, // add_line row
            const std::vector<types::global_dof_index> cols,
            const Tensor<1, dim>                       normal,
            const double                               coeff,
            const uint                                 line_comp, // comp of add_line
            const uint                                 starting_comp,
            const uint                                 num_comp, // number of comp (dim)
            const Point<dim> &                         gsp,
            const std::vector<Point<dim>> &            fsp,
            AffineConstraints<double> &                constraints_flux)
    {
        TimerOutput::Scope timing_section(computing_timer, "add constraints");
        const uint dofs_per_face = fe.dofs_per_face;
        std::vector<std::pair<types::global_dof_index, double>> col_vals_pair;
        for (unsigned int i = 0; i < dofs_per_face; ++i)
        {
            const unsigned int comp = fe.face_system_to_component_index(i).first;
            if (cols[i] != row && comp >= starting_comp &&
                comp < starting_comp + num_comp && gsp.distance(fsp[i]) < 1e-10)
            {
                const double entry =
                        coeff * normal[comp - starting_comp] / normal[line_comp];
                col_vals_pair.emplace_back(cols[i], entry);
                constrainted_flag[cols[i]] = true;
#if 0
                pcout << " constraint(" << row << "," << cols[i] << ")=" << entry
                << std::endl;
#endif
            }
        }
        constraints_flux.add_entries(row, col_vals_pair);
    }

    template <int dim>
    void
    FluidStructureProblem<dim>::set_interface_dofs_flag(
            TrilinosWrappers::MPI::Vector &flag, const IndexSet &hp_index_set)
    {
        flag.clear();
        flag.reinit(hp_index_set, MPI_COMM_WORLD);
        for (auto it = flag.begin(); it != flag.end();
             ++it)
        {
            *it = 1;
        }

        const unsigned int hydrogel_dofs_per_cell = hydrogel_fe.dofs_per_cell;
        for (const auto &cell : dof_handler.active_cell_iterators())
            if (cell->is_locally_owned())
            {
                if (cell_is_in_hydrogel_domain(cell))
                {
                    std::vector<types::global_dof_index> local_dof_indices(
                            hydrogel_dofs_per_cell);
                    cell->get_dof_indices(local_dof_indices);
                    for (unsigned int i = 0; i < hydrogel_dofs_per_cell; ++i)
                        flag[local_dof_indices[i]] = 0;
                }
            }
    }

    // This function uses volume_solution to compute the velocity constraint on
    // the hydrogel surface
    template <int dim>
    void
    FluidStructureProblem<dim>::make_flux_constraints(
            AffineConstraints<double> &constraints_flux)
    {
        TimerOutput::Scope timing_section(computing_timer, "make flux constraints");
        const auto &sfe =
                stokes_fe.get_sub_fe(extractor_stokes_velocity.first_vector_component, 1);
        const auto &generalized_unit_face_support_points =
                sfe.get_unit_face_support_points();

        // construct fe values to get normal vectors and components
        const Quadrature<dim - 1> st_q(generalized_unit_face_support_points);

        constrainted_flag.clear();
        constrainted_flag.resize(dof_handler.n_locally_owned_dofs(), false);

        FEFaceValues<dim>   stokes_fe_face_values(*mapping_pointer,
                                                  stokes_fe,
                                                  st_q,
                                                  update_quadrature_points |
                                                  update_normal_vectors |
                                                  update_values);
        FEFaceValues<dim>   hydrogel_fe_face_values(*mapping_pointer,
                                                    hydrogel_fe,
                                                    st_q,
                                                    update_quadrature_points |
                                                    update_values |
                                                    update_normal_vectors |
                                                    update_gradients);
        FEFaceValues<dim>   volume_fe_face_values(*mapping_pointer,
                                                  volume_fe,
                                                  st_q,
                                                  update_values);
        const uint          n_q_face = st_q.size();
        std::vector<double> volume_s_face(n_q_face);
        const auto          st_dofs_per_face = stokes_fe.dofs_per_face;
        const auto          el_dofs_per_face = hydrogel_fe.dofs_per_face;
        std::vector<types::global_dof_index> stokes_face_dof_indices(
                st_dofs_per_face);
        std::vector<types::global_dof_index> hydrogel_face_dof_indices(
                el_dofs_per_face);
        SymmetricTensor<2, dim> identity_tensor;
        for (unsigned int d = 0; d < dim; ++d)
            identity_tensor[d][d] = 1.;

        // copy from step 46
        for (const auto &cell : dof_handler.active_cell_iterators())
            if (cell->is_locally_owned())
            {
                if (cell_is_in_fluid_domain(cell))
                    /*for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell;
                         ++f) // f : face #*/
                    for (const auto &f : cell->face_indices())
                        if (!cell->at_boundary(f))
                        {
                            bool face_is_on_interface = false;
                            if ((cell->neighbor(f)->has_children() == false) &&
                                (cell_is_in_hydrogel_domain(cell->neighbor(f))))
                                face_is_on_interface = true;
                            else if (cell->neighbor(f)->has_children() == true)
                            {
                                for (unsigned int sf = 0; sf < cell->face(f)->n_children();
                                     ++sf)
                                    if (cell_is_in_hydrogel_domain(
                                            cell->neighbor_child_on_subface(f, sf)))
                                    {
                                        face_is_on_interface = true;
                                        break;
                                    }
                            }
                            if (face_is_on_interface)
                            {
                                const auto nbr_face_no = cell->neighbor_of_neighbor(f);
#if 0
                                const auto & face_center = cell->face(f)->center();
            pcout << " face center: " << cell->face(f)->center()
                      << " norm: " << face_center.norm()
                      << " face no  = " << f << " nbr face = " << nbr_face_no
                      << std::endl;
#endif
                                stokes_fe_face_values.reinit(cell, f);
                                const auto &neighbor = cell->neighbor(f);

                                // get phi_s
                                typename DoFHandler<dim>::active_cell_iterator
                                        neighbor_cell_volume(&triangulation,
                                                             neighbor->level(),
                                                             neighbor->index(),
                                                             &volume_dof_handler);
                                volume_fe_face_values.reinit(neighbor_cell_volume,
                                                             nbr_face_no);
                                volume_fe_face_values.get_function_values(volume_solution,
                                                                          volume_s_face);

                                hydrogel_fe_face_values.reinit(neighbor, nbr_face_no);

                                // get normal vectors
                                const std::vector<Tensor<1, dim>> &normal_vectors =
                                        hydrogel_fe_face_values.get_normal_vectors();

                                std::vector<Tensor<2, dim>> grad_u_face(n_q_face);
                                hydrogel_fe_face_values[extractor_displacement]
                                        .get_function_gradients(solution, grad_u_face);

                                // the following two have repeated support points

                                // The order of points in the array matches that returned by
                                // the cell->face(face)->get_dof_indices function.
                                const std::vector<Point<dim - 1>>
                                        &stokes_unit_support_points =
                                        stokes_fe.get_unit_face_support_points();
                                const std::vector<Point<dim - 1>>
                                        &elasticity_unit_support_points =
                                        hydrogel_fe.get_unit_face_support_points();

                                const std::vector<Point<dim>> &stokes_physical_face_points =
                                        get_physical_face_points(cell,
                                                                 f,
                                                                 stokes_unit_support_points);
                                const std::vector<Point<dim>> &hydrogel_physical_face_points =
                                        get_physical_face_points(neighbor,
                                                                 nbr_face_no,
                                                                 elasticity_unit_support_points);
                                const std::vector<Point<dim>> &
                                        generalized_physical_face_points = get_physical_face_points(
                                                cell, f, generalized_unit_face_support_points);

                                // //pair the global indices and the physical support points
                                // std::vector<std::pair<unsigned int, Point<dim>>>
                                // st_global_id_to_phy_sp; std::vector<std::pair<unsigned int,
                                // Point<dim>>> el_global_id_to_phy_sp;

                                // 0 for stokes, 1 for elasticity
                                // determined by the function set_active_fe_indices() in this
                                // class
                                neighbor->face(cell->neighbor_of_neighbor(f))
                                        ->get_dof_indices(hydrogel_face_dof_indices, 1);
                                cell->face(f)->get_dof_indices(stokes_face_dof_indices, 0);

                                // generalized support points are also the quadrature points
                                for (unsigned int k = 0;
                                     k < generalized_unit_face_support_points.size();
                                     ++k)
                                {
                                    // add one line at each generalized support point
                                    // make sure diagonal is nonzero

                                    // determin which line to add at this support point:
                                    // the comp correspond to the max entry of the normal
                                    // vector

                                    unsigned int   comp = 0;

                                    Tensor<2, dim> deformation_F(identity_tensor);
                                    deformation_F += grad_u_face[k];
                                    const Tensor<2, dim> F_T =
                                            transpose(deformation_F);                 // F^T
                                    const Tensor<2, dim> inv_F_T = invert(F_T); // F^{-T}
                                    const auto           fn = inv_F_T * normal_vectors[k];
                                    const auto           magnitude_m = fn.norm();
                                    const Tensor<1, dim> normal_v    = fn / magnitude_m;
                                    {
                                        double tmp = 1e-15;
                                        for (unsigned int d = 0; d < dim; ++d)
                                        {
                                            if (std::fabs(normal_v[d]) > tmp)
                                            {
                                                comp = d;
                                                tmp  = std::fabs(normal_v[d]);
                                            }
                                        }
                                    }

                                    // add line first, note that we just need one line on each
                                    // support point
                                    bool         add_entries = false;
                                    unsigned int add_line_id = st_dofs_per_face;
                                    for (unsigned int i = 0; i < st_dofs_per_face; ++i)
                                    {
                                        if (stokes_fe.face_system_to_component_index(i)
                                                    .first == comp + extractor_stokes_velocity
                                                .first_vector_component &&
                                            generalized_physical_face_points[k].distance(
                                                    stokes_physical_face_points[i]) < 1e-10)
                                        {
                                            if (!constrainted_flag
                                            [stokes_face_dof_indices[i]])
                                            {
                                                constraints_flux.add_line(
                                                        stokes_face_dof_indices[i]);
                                                add_line_id = i;
                                                add_entries = true;
                                                constrainted_flag
                                                [stokes_face_dof_indices[add_line_id]] =
                                                        true;
                                                break;
                                            }
                                        }
                                    }
                                    //              AssertIndexRange(add_line_id,
                                    //              st_dofs_per_face);
                                    // add entries
                                    // stokes:
                                    if (add_entries)
                                    {
                                        //                pcout<<" fn:
                                        //                "<<normal_v<<std::endl;

                                        add_interface_constraints(
                                                stokes_fe,
                                                stokes_face_dof_indices[add_line_id],
                                                stokes_face_dof_indices,
                                                normal_v, /*normal vector at the sp*/
                                                -1.,      /*coefficient*/
                                                comp,     /*comp of add_line*/
                                                extractor_stokes_velocity
                                                        .first_vector_component /*starting comp*/,
                                                dim /*number of comp*/,
                                                generalized_physical_face_points
                                                [k] /*support point correspond to this dof*/,
                                                stokes_physical_face_points,
                                                constraints_flux);

                                        add_interface_constraints(
                                                hydrogel_fe,
                                                stokes_face_dof_indices[add_line_id],
                                                hydrogel_face_dof_indices,
                                                normal_v,                /*normal vector at the sp*/
                                                (1. - volume_s_face[k]), /*coefficient*/
                                                comp,
                                                extractor_hydrogel_velocity
                                                        .first_vector_component /*starting comp*/,
                                                dim /*number of comp*/,
                                                generalized_physical_face_points
                                                [k] /*support point correspond to this dof*/,
                                                hydrogel_physical_face_points,
                                                constraints_flux);

                                        add_interface_constraints(
                                                hydrogel_fe,
                                                stokes_face_dof_indices[add_line_id],
                                                hydrogel_face_dof_indices,
                                                normal_v,         /*normal vector at the sp*/
                                                volume_s_face[k], /*coefficient*/
                                                comp,
                                                extractor_mesh_velocity
                                                        .first_vector_component /*starting comp*/,
                                                dim /*number of comp*/,
                                                generalized_physical_face_points
                                                [k] /*support point correspond to this dof*/,
                                                hydrogel_physical_face_points,
                                                constraints_flux);
                                    } // if add entries

                                } // q loop
                            }
                        }
            }

    }

    template <int dim>
    void
    FluidStructureProblem<dim>::solver(TrilinosWrappers::SparseMatrix &matrix,
                                       TrilinosWrappers::MPI::Vector &sol,
                                       TrilinosWrappers::MPI::Vector &rhs)
    {
        SolverControl                    solver_control;
        TrilinosWrappers::SolverDirect::AdditionalData data(false, "Amesos_Mumps");
        TrilinosWrappers::SolverDirect DirectSolver(solver_control, data);
        DirectSolver.solve(matrix, sol, rhs);
    }

    template <int dim>
    void
    FluidStructureProblem<dim>::local_assemble_volume(
            const typename DoFHandler<dim>::active_cell_iterator &cell_volume,
            ScratchData &                                         scratch,
            PerTaskData &                                         copy_data)
    {
        typename DoFHandler<dim>::active_cell_iterator cell_hp(&triangulation,
                                                               cell_volume->level(),
                                                               cell_volume->index(),
                                                               &dof_handler);

        if (cell_is_in_hydrogel_domain(cell_hp))
        {
            scratch.hp_fe_values.reinit(cell_hp);
            scratch.volume_fe_values.reinit(cell_volume);
            const FEValues<dim> &fe_values =
                    scratch.hp_fe_values.get_present_fe_values();
            const FEValues<dim> &volume_fe_values =
                    scratch.volume_fe_values.get_present_fe_values();
            const unsigned int n_q_points    = volume_fe_values.n_quadrature_points;
            const unsigned int dofs_per_cell = cell_volume->get_fe().dofs_per_cell;
            copy_data.volume_cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
            copy_data.volume_cell_rhs.reinit(dofs_per_cell);
            copy_data.volume_local_dof_indices.resize(dofs_per_cell);
            cell_volume->get_dof_indices(copy_data.volume_local_dof_indices);
            // declaring all test functions
            // volume
            std::vector<double> psi_phi_s(dofs_per_cell); // \psi_{\phi_s}
            SymmetricTensor<2, dim> identity_tensor; // identity tensor
            for (unsigned int d = 0; d < dim; ++d)
                identity_tensor[d][d] = 1.;
            Assert(dofs_per_cell == volume_fe.dofs_per_cell, ExcInternalError());
            // grad u(gradients of mesh displacement)
            std::vector<Tensor<2, dim>> grad_u(n_q_points);
            fe_values[extractor_displacement].get_function_gradients(solution,
                                                                     grad_u);
            // grad vs (mesh velocity)
            std::vector<Tensor<2, dim>> grad_vs(n_q_points);
            fe_values[extractor_mesh_velocity].get_function_gradients(solution,
                                                                      grad_vs);
            std::vector<double> volume_shape_values(n_q_points);
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                // compute F an J
                Tensor<2, dim> deformation_F(identity_tensor); //define && initialize
                deformation_F += grad_u[q];                               // F
                const double jacobian = determinant(deformation_F);       // J
                const auto   inv_F_T  = transpose(invert(deformation_F)); // F^{-T}
                // assemble local rhs
                double tmp_rhs, tmp_mat;
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    volume_shape_values[i] = volume_fe_values.shape_value(i, q);
                    // Sigma^star F_hat_inv_transpose_star J_hat_star
                    tmp_rhs = 0.;
                     //div(F_hat_inv . v_s J_hat)
                     //= J_hat F_hat_inv : grad_hat(v_s)
                    tmp_rhs += jacobian * scalar_product(inv_F_T, grad_vs[q]) *
                    volume_shape_values[i];
                    // no need to change this, pay attention to the negative sign
                    copy_data.volume_cell_rhs(i) -=
                            volume_fe_values.JxW(q) * tmp_rhs;
                }

                // assemble local matrix
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        tmp_mat = 0.;
                         tmp_mat += jacobian / time_step * volume_shape_values[j] *
                                volume_shape_values[i];
                        // add all the other terms here using the same format
                        // no need to change this
                        copy_data.volume_cell_matrix(i, j) +=
                                volume_fe_values.JxW(q) * tmp_mat;
                    }
            }
            // assemble local matrix
        }
    } // namespace Step46

    template <int dim>
    void
    FluidStructureProblem<dim>::copy_local_to_global_volume(
            const PerTaskData &copy_data)
    {
        constraints_volume.distribute_local_to_global(
                copy_data.volume_cell_matrix,
                copy_data.volume_cell_rhs,
                copy_data.volume_local_dof_indices,
                volume_system_matrix,
                volume_system_rhs);
    }

    template <int dim>
    void
    FluidStructureProblem<dim>::assemble_volume_system_workstream()
    {
        TimerOutput::Scope timing_section(computing_timer, "assemble volume system");
        pcout << " assembling volume...";
        volume_system_matrix = 0;
        volume_system_rhs    = 0;
        using CellFilter =
                FilteredIterator<typename DoFHandler<2>::active_cell_iterator>;

        const QGauss<dim> quadrature_formula(2 + volume_degree);
        // same quadrature for all
        const hp::QCollection<dim> q_collection{quadrature_formula,
                                                quadrature_formula};
        const QGauss<dim - 1>      face_quadrature_formula(2 + volume_degree);

        const UpdateFlags          hp_update_flags =
                update_values | update_gradients | update_quadrature_points;
        const UpdateFlags face_update_flags = update_values; // update_default
        const UpdateFlags volume_update_flag =
                update_values | update_JxW_values | update_quadrature_points;

        ScratchData sd(fe_collection,
                       volume_fe_collection,
                       mapping_collection,
                       stokes_fe,
                       hydrogel_fe,
                       volume_fe,
                       q_collection,
                       q_collection,
                       face_quadrature_formula,
                       hp_update_flags,
                       volume_update_flag,
                       face_update_flags);

        PerTaskData cp;
        auto worker =
                [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
                       ScratchData &                                         scratch,
                       PerTaskData &                                         copy_data) {
                    this->local_assemble_volume(cell, scratch, copy_data);
                };
        auto copier = [this](const PerTaskData &copy_data) {
            this->copy_local_to_global_volume(copy_data);
        };

        WorkStream::run(CellFilter(IteratorFilters::LocallyOwnedCell(),
                                   volume_dof_handler.begin_active()),
                        CellFilter(IteratorFilters::LocallyOwnedCell(),
                                   volume_dof_handler.end()),
                        worker,
                        copier,
                        sd,
                        cp);
        volume_system_matrix.compress(VectorOperation::add);
        volume_system_rhs.compress(VectorOperation::add);
#ifdef DEBUG_TIMING
        timer.stop();
    int old_precision = pcout.precision();
    pcout << "  time elapsed in parallel assemble_system_ch ="
              << std::setprecision(3) << timer() << "[CPU];"
              << timer.wall_time() << "[Wall]" << std::endl;
    pcout.precision(old_precision);
#endif
        pcout << " done!" << std::endl;
    }

    template <int dim>
    void
    FluidStructureProblem<dim>::solve_volume()
    {
        TimerOutput::Scope timing_section(computing_timer, "solve volume");
        pcout << " solving volume..." << std::endl;
        solver(volume_system_matrix, dis_volume_solution, volume_system_rhs);
        auto it_old = dis_volume_old_solution.begin();
        for (auto it = dis_volume_solution.begin(); it != dis_volume_solution.end();
             ++it, ++it_old)
        {
            *it = std::min((*it_old) * (std::exp(*it)), 0.99);
        }

        constraints_volume.distribute(dis_volume_solution);
        volume_solution = dis_volume_solution;
        pcout << " done!" << std::endl;
    }

    template <int dim>
    void
    FluidStructureProblem<dim>::local_assemble_hp(
            const typename DoFHandler<dim>::active_cell_iterator &cell_hp,
            ScratchData &                                         scratch,
            PerTaskData &                                         copy_data,
            const bool                                            update_matrix)
    {
        copy_data.assemble_interface = false;
        scratch.hp_fe_values.reinit(cell_hp);
        typename DoFHandler<dim>::active_cell_iterator cell_volume(
                &triangulation, cell_hp->level(), cell_hp->index(), &volume_dof_handler);

        scratch.volume_fe_values.reinit(cell_volume);
        const unsigned int stokes_dofs_per_cell   = stokes_fe.dofs_per_cell;
        const unsigned int hydrogel_dofs_per_cell = hydrogel_fe.dofs_per_cell;
        const FEValues<dim> &fe_values =
                scratch.hp_fe_values.get_present_fe_values();
        const FEValues<dim> &volume_fe_values =
                scratch.volume_fe_values.get_present_fe_values();

        const unsigned int n_q_points    = fe_values.n_quadrature_points;
        const unsigned int dofs_per_cell = cell_hp->get_fe().dofs_per_cell;

        copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
        copy_data.cell_rhs.reinit(dofs_per_cell);
        copy_data.local_dof_indices.resize(dofs_per_cell);
        cell_hp->get_dof_indices(copy_data.local_dof_indices);

        // declaring all test functions
        std::vector<Tensor<2, dim>> grad_shape_stokes_velocity(
                stokes_dofs_per_cell); // \nabla psi_V
        std::vector<double> div_shape_stokes_velocity(
                stokes_dofs_per_cell); // \nabla \cdot psi_V
        std::vector<double> shape_stokes_pressure(
                stokes_dofs_per_cell); // psi_P, shape_stokes_pressure

        // hydrogel
        std::vector<Tensor<1, dim>> shape_vf(hydrogel_dofs_per_cell); // psi_{v_f}
        std::vector<Tensor<2, dim>> shape_grad_vf(
                hydrogel_dofs_per_cell); // \nabla psi_{v_f}

        std::vector<Tensor<1, dim>> shape_vs(hydrogel_dofs_per_cell); // psi_{v_s}
        std::vector<Tensor<2, dim>> shape_grad_vs(
                hydrogel_dofs_per_cell); // \nabla psi_{v_s}

        std::vector<Tensor<1, dim>> shape_displacement(
                hydrogel_dofs_per_cell); // shape_displacement
        std::vector<Tensor<2, dim>> grad_shape_displacement(
                hydrogel_dofs_per_cell); // \nabla shape_displacement
        std::vector<double> div_shape_displacement(
                hydrogel_dofs_per_cell); // \nabla \cdot shape_displacement

        std::vector<double> shape_hydrogel_pressure(
                hydrogel_dofs_per_cell); // shape_hydrogel_pressure

        SymmetricTensor<2, dim> identity_tensor;
        for (unsigned int d = 0; d < dim; ++d)
            identity_tensor[d][d] = 1.;

        if (cell_is_in_fluid_domain(cell_hp))
        {
            // pcout<<"assembling stokes domain..."<<std::endl;
            Assert(dofs_per_cell == stokes_dofs_per_cell, ExcInternalError());

            // grad u(gradients of mesh displacement)
            std::vector<Tensor<2, dim>> grad_u(n_q_points); // \nabla u
            fe_values[extractor_displacement].get_function_gradients(
                    current_solution, grad_u);

            // get solution^star (current_solution, newton's iterative method, n;
            // solution, n+1)

            //\nabla V
            std::vector<Tensor<2, dim>> grad_v(n_q_points);
            fe_values[extractor_stokes_velocity].get_function_gradients(
                    current_solution, grad_v);

            // get values of P at n step
            std::vector<double> stokes_pressure(n_q_points);
            fe_values[extractor_stokes_pressure].get_function_values(
                    current_solution, stokes_pressure);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                // extract shape functions first
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                    // grad Psi_V
                    grad_shape_stokes_velocity[k] =
                            fe_values[extractor_stokes_velocity].gradient(
                                    k, q); // G(i,j) = partial phi/ partial xj

                    // Psi_P , shape function
                    shape_stokes_pressure[k] =
                            fe_values[extractor_stokes_pressure].value(k, q);

                    // displacement shape functions
                    grad_shape_displacement[k] =
                            fe_values[extractor_displacement].gradient(k, q);
                }

                // compute F an J
                Tensor<2, dim> deformation_F(identity_tensor);
                deformation_F += grad_u[q]; // F

                const double         jacobian = determinant(deformation_F); // J
                const auto           inv_F    = invert(deformation_F);    // F^{-1}
                const Tensor<2, dim> F_T      = transpose(deformation_F); // F^T
                const Tensor<2, dim> inv_F_T  = invert(F_T);              // F^{-T}

                const Tensor<2, dim> grad_vq =
                        grad_v[q]; // stokes_velocity_values at quadrature points
                const Tensor<2, dim> grad_vq_T = transpose(grad_vq);

                // assemble local rhs
                double tmp_rhs, tmp_mat;

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    // Sigma^star F_hat_inv_transpose_star J_hat_star
                    tmp_rhs = 0.;
                    tmp_rhs +=
                            Ca * viscosity * jacobian *
                            scalar_product((inv_F_T * grad_vq_T + grad_vq * inv_F) *
                                           inv_F_T,
                                           grad_shape_stokes_velocity[i]) - // A_1

                            stokes_pressure[q] * jacobian *
                            scalar_product(inv_F_T,
                                           grad_shape_stokes_velocity[i]) + // A_4

                            jacobian * scalar_product(inv_F_T, grad_vq) *
                            shape_stokes_pressure[i] + // A_6

                            alpha *
                            scalar_product(grad_u[q],
                                           grad_shape_displacement[i]) *
                            static_cast<double>(
                                    interface_dofs_flag[copy_data
                                            .local_dof_indices[i]]); // A_13

                    // no need to change this, pay attention to the negative sign
                    copy_data.cell_rhs(i) -= fe_values.JxW(q) * tmp_rhs;
                }

                // assemble local matrix
                if (update_matrix)
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                            tmp_mat = 0.;

                            tmp_mat +=
                                    Ca * viscosity * jacobian *
                                    scalar_product(
                                            (inv_F_T * transpose(grad_shape_stokes_velocity[j]) +
                                             grad_shape_stokes_velocity[j] * inv_F) *
                                            inv_F_T,
                                            grad_shape_stokes_velocity[i]) - // A'_1

                                    shape_stokes_pressure[j] * jacobian *
                                    scalar_product(inv_F_T,
                                                   grad_shape_stokes_velocity[i]) + // A'_4

                                    jacobian *
                                    scalar_product(inv_F_T, grad_shape_stokes_velocity[j]) *
                                    shape_stokes_pressure[i] + // A'_6

                                    alpha *
                                    scalar_product(grad_shape_displacement[j],
                                                   grad_shape_displacement[i]) *
                                    static_cast<double>(
                                            interface_dofs_flag
                                            [copy_data.local_dof_indices[i]]); // A'_13

                            // add all the other terms here using the same format
                            // no need to change this
                            copy_data.cell_matrix(i, j) += fe_values.JxW(q) * tmp_mat;
                        } // dofs loop
            }         // q loop
        } // outer domain (fluid domain)

        else /*(cell_is_in_hydrogel_domain)*/
        {
            // pcout<<"assembling hydrogel domain...\n";
            // same as in the stokes domain
            Assert(dofs_per_cell == hydrogel_dofs_per_cell, ExcInternalError());
            // grad u(gradients of mesh displacement)
            std::vector<Tensor<2, dim>> grad_u(n_q_points); // \nabla u
            fe_values[extractor_displacement].get_function_gradients(
                    current_solution, grad_u);

            std::vector<Tensor<1, dim>> displacement_u(n_q_points),
                    old_displacement_u(n_q_points);
            fe_values[extractor_displacement].get_function_values(current_solution,
                                                                  displacement_u);
            fe_values[extractor_displacement].get_function_values(
                    old_solution, old_displacement_u);

            // get vs current solution, n step in Newton's method
            std::vector<Tensor<1, dim>> vs(n_q_points);
            fe_values[extractor_mesh_velocity].get_function_values(current_solution,
                                                                   vs);
            // grad vs(gradients of mesh displacement)
            std::vector<Tensor<2, dim>> grad_vs(n_q_points);
            fe_values[extractor_mesh_velocity].get_function_gradients(
                    current_solution, grad_vs);

            // get vf current solution, n step in Newton's method
            std::vector<Tensor<1, dim>> vf(n_q_points);
            fe_values[extractor_hydrogel_velocity].get_function_values(
                    current_solution, vf);
            // grad vf(gradients of mesh displacement)
            std::vector<Tensor<2, dim>> grad_vf(n_q_points);
            fe_values[extractor_hydrogel_velocity].get_function_gradients(
                    current_solution, grad_vf);

            // hydrogel pressure
            std::vector<double> hydrogel_pressure(n_q_points);
            fe_values[extractor_hydrogel_pressure].get_function_values(
                    current_solution, hydrogel_pressure);

            // volume fraction
            std::vector<double> volume_s(n_q_points);
            volume_fe_values.get_function_values(volume_solution, volume_s);
            // grad volume fraction
            std::vector<Tensor<1, dim>> grad_volume_s(n_q_points);
            volume_fe_values.get_function_gradients(volume_solution, grad_volume_s);

            // pressure, n step in Newton's method
            // std::vector<double> p2(n_q_points);
            // fe_values[extractor_hydrogel_pressure].get_function_values(current_solution,
            // p2);

            // volume
            std::vector<double> psi_phi_s(stokes_dofs_per_cell); // \psi_{\phi_s}

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                //Get the value of mu_s and lambda_s
                //const auto &x_q = fe_values.quadrature_point(q);
                //mu_s=mu_s_value.value(x_q);
                //lambda_s=mu_s;
                //pcout<<x_q<<"  "<<mu_s <<std::endl;
                // extract shape functions first
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                    // Psi_vf and grad \Psi_vf
                    shape_vf[k] =
                            fe_values[extractor_hydrogel_velocity].value(k, q);
                    shape_grad_vf[k] =
                            fe_values[extractor_hydrogel_velocity].gradient(k, q);

                    // Psi_vs and grad \Psi_vs
                    shape_vs[k] = fe_values[extractor_mesh_velocity].value(k, q);
                    shape_grad_vs[k] =
                            fe_values[extractor_mesh_velocity].gradient(k, q);

                    // Psi_u and grad \Psi_u
                    shape_displacement[k] =
                            fe_values[extractor_displacement].value(k, q);
                    grad_shape_displacement[k] =
                            fe_values[extractor_displacement].gradient(k, q);
                    div_shape_displacement[k] =
                            fe_values[extractor_displacement].divergence(k, q);

                    // Psi_pressure
                    shape_hydrogel_pressure[k] =
                            fe_values[extractor_hydrogel_pressure].value(k, q);
                }
                // compute F an J
                Tensor<2, dim> deformation_F(identity_tensor);
                deformation_F += grad_u[q]; // F

                const double         jacoiban = determinant(deformation_F); // J
                const auto           inv_F    = invert(deformation_F); // F inverse
                const Tensor<2, dim> F_T = transpose(deformation_F); // F transpose
                const Tensor<2, dim> inv_F_T = invert(F_T); // (F transpose) inverse

                const Tensor<2, dim> grad_vfq   = grad_vf[q];          //
                const Tensor<2, dim> grad_vfq_T = transpose(grad_vfq); // grad vf T

                const Tensor<2, dim> grad_vsq = grad_vs[q]; // grad vs


                // assemble local rhs
                double tmp_rhs, tmp_mat;

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    // Sigma^star F_hat_inv_transpose_star J_hat_star
                    tmp_rhs = 0.;
                    tmp_rhs +=
                            Ca * (1. - volume_s[q]) * vis_BM * jacoiban *
                            scalar_product(((inv_F_T * grad_vfq_T + grad_vfq * inv_F) *
                                            inv_F_T),
                                           shape_grad_vf[i]) + // A_2

                            volume_s[q] * scalar_product(
                                    mu_s * (deformation_F - inv_F_T) +
                                    lambda_s * (jacoiban - 1) * jacoiban * inv_F_T,
                                    shape_grad_vs[i]) - //A_3

                            hydrogel_pressure[q] * jacoiban *
                            scalar_product(
                                    (outer_product(
                                            shape_vf[i],
                                            -grad_volume_s[q]) + // \shape_vf \otimes - grad \phi_s
                                     (1. - volume_s[q]) *
                                     shape_grad_vf[i] + // \psi_f * grad shape_vf
                                     outer_product(
                                             shape_vs[i],
                                             grad_volume_s[q]) + // \shape_vs \otimes grad \phi_s
                                     volume_s[q] *
                                     shape_grad_vs[i]), // \phi_s * grad shape_vs
                                    inv_F_T) +            // A_5

                            jacoiban *
                            scalar_product((1. - volume_s[q]) * grad_vfq + //\phi_f * vf
                                           outer_product(vf[q], -grad_volume_s[q]) +
                                           volume_s[q] * grad_vsq + //\phi_s * vs
                                           outer_product(vs[q], grad_volume_s[q]),
                                           inv_F_T) *
                            shape_hydrogel_pressure[i] - // A_7

                            Ca * xi * (1. - volume_s[q]) * volume_s[q] * jacoiban *
                            (vs[q] - vf[q]) * (shape_vf[i] - shape_vs[i]) + // A_8

                            ((displacement_u[q] - old_displacement_u[q]) / time_step -
                             vs[q]) *
                            shape_displacement[i]; // A_12

                    // add other terms here using the format above

                    // no need to change this, pay attention to the negative sign
                    copy_data.cell_rhs(i) -= fe_values.JxW(q) * tmp_rhs;
                }

                // assemble local matrix

                if (update_matrix)
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                            const Tensor<2, dim> d_F = grad_shape_displacement[j];
                            const Tensor<2, dim> d_inv_F_T= -inv_F_T*transpose(d_F)*inv_F_T;
                            const double d_jacoiban = jacoiban * scalar_product(inv_F_T,d_F);
                            tmp_mat = 0.;

                            tmp_mat +=
                                    Ca * (1. - volume_s[q]) * vis_BM * jacoiban *
                                    scalar_product(((inv_F_T * transpose(shape_grad_vf[j]) +
                                                     shape_grad_vf[j] * inv_F) *
                                                    inv_F_T),
                                                   shape_grad_vf[i]) + // A'_2

                                    volume_s[q] * scalar_product(
                                            mu_s * (d_F - d_inv_F_T) +
                                            lambda_s * ((2 * jacoiban - 1) * d_jacoiban * inv_F_T +
                                                        (jacoiban-1) * jacoiban * d_inv_F_T),
                                            shape_grad_vs[i]) -  //A'3

                                    shape_hydrogel_pressure[j] * jacoiban *
                                    scalar_product(
                                            (outer_product(
                                                    shape_vf[i],
                                                    -grad_volume_s[q]) + // \shape_vf \otimes
                                             // - grad \phi_s
                                             (1. - volume_s[q]) *
                                             shape_grad_vf[i] + // \phi_f * grad Psi_vf

                                             outer_product(
                                                     shape_vs[i],
                                                     grad_volume_s[q]) + // \shape_vs \otimes
                                             // grad \phi_s
                                             volume_s[q] *
                                             shape_grad_vs[i]), // \phi_s * grad shape_vs
                                            inv_F_T) +            // A'_5

                                    jacoiban *
                                    scalar_product(
                                            (1 - volume_s[q]) *
                                            shape_grad_vf[j] + //\phi_f * grad shape_vf
                                            outer_product(
                                                    shape_vf[j],
                                                    -grad_volume_s[q]) + // shape_vf \otimes grad
                                            // volume _f
                                            volume_s[q] *
                                            shape_grad_vs[j] + //\phi_s * grad shape_vs
                                            outer_product(shape_vs[j],
                                                          grad_volume_s[q]), // shape_vs \otimes
                                            // grad volume _s
                                            inv_F_T) *
                                    shape_hydrogel_pressure[i] - // A'_7

                                    Ca * xi * (1. - volume_s[q]) * volume_s[q] * jacoiban *
                                    (-shape_vf[j] + shape_vs[j]) *
                                    (shape_vf[i] - shape_vs[i]) + // A'_8

                                    (shape_displacement[j] / time_step - shape_vs[j]) *
                                    shape_displacement[i]; // A'_12

                            // add all the other terms here using the same format

                            // no need to change this
                            copy_data.cell_matrix(i, j) += fe_values.JxW(q) * tmp_mat;
                        }
            } // q loop


            // currently only works for uniform mesh;
            //for (unsigned int f : GeometryInfo<dim>::face_indices())
            for (const auto &f : cell_hp->face_indices())
                if (!cell_hp->at_boundary(f))
                    if (cell_is_in_fluid_domain(cell_hp->neighbor(f)))
                    {
                        const auto &neighbor = cell_hp->neighbor(f);
                        copy_data.neighbor_dof_indices.resize(stokes_dofs_per_cell);
                        neighbor->get_dof_indices(copy_data.neighbor_dof_indices);
                        if (neighbor->level() == cell_hp->level() &&
                            !neighbor->has_children())
                        {
                            // pcout<<"assembling interface terms...";
                            copy_data.assemble_interface = true;
                            scratch.hydrogel_fe_face_values.reinit(cell_hp, f);
                            scratch.stokes_fe_face_values.reinit(
                                    neighbor, cell_hp->neighbor_of_neighbor(f));

                            Assert(
                                    scratch.hydrogel_fe_face_values.n_quadrature_points ==
                                    scratch.stokes_fe_face_values.n_quadrature_points,
                                    ExcInternalError());
                            const unsigned int n_q_points_face =
                                    scratch.stokes_fe_face_values.n_quadrature_points;

                            std::vector<Tensor<2, dim>> grad_u_face(n_q_points_face);
                            scratch.stokes_fe_face_values[extractor_displacement]
                                    .get_function_gradients(current_solution, grad_u_face);

                            std::vector<Tensor<1, dim>> stokes_velocity_face(
                                    n_q_points_face);
                            scratch.stokes_fe_face_values[extractor_stokes_velocity]
                                    .get_function_values(current_solution,
                                                         stokes_velocity_face);

                            std::vector<Tensor<1, dim>> vs_face(n_q_points_face),
                                    vf_face(n_q_points_face);
                            scratch.hydrogel_fe_face_values[extractor_mesh_velocity]
                                    .get_function_values(current_solution, vs_face);
                            scratch.hydrogel_fe_face_values[extractor_hydrogel_velocity]
                                    .get_function_values(current_solution, vf_face);

                            const unsigned int stokes_dofs_per_cell =
                                    stokes_fe.dofs_per_cell;
                            const unsigned int hydrogel_dofs_per_cell =
                                    hydrogel_fe.dofs_per_cell;

                            //(stokes,stokes)
                            copy_data.interface_matrix1.reinit(stokes_dofs_per_cell,
                                                               stokes_dofs_per_cell);
                            copy_data.interface_rhs1.reinit(stokes_dofs_per_cell);
                            //(stokes,hydrogel)
                            copy_data.interface_matrix2.reinit(hydrogel_dofs_per_cell,
                                                               stokes_dofs_per_cell);
                            copy_data.interface_rhs2.reinit(hydrogel_dofs_per_cell);
                            //(hydrogel,stokes)
                            copy_data.interface_matrix3.reinit(stokes_dofs_per_cell,
                                                               hydrogel_dofs_per_cell);
                            copy_data.interface_rhs3.reinit(stokes_dofs_per_cell);
                            //(hydrogel,hydrogel)
                            copy_data.interface_matrix4.reinit(hydrogel_dofs_per_cell,
                                                               hydrogel_dofs_per_cell);
                            copy_data.interface_rhs4.reinit(hydrogel_dofs_per_cell);

                            for (unsigned int q = 0; q < n_q_points_face; ++q)
                            {
                                Tensor<2, dim> deformation_F(identity_tensor);
                                deformation_F += grad_u_face[q];                    // F
                                const double jacobian = determinant(deformation_F); // J
                                const auto   inv_F    = invert(deformation_F); // F^{-1}
                                const Tensor<2, dim> F_T =
                                        transpose(deformation_F);                 // F^T
                                const Tensor<2, dim> inv_F_T = invert(F_T); // F^{-T}

                                std::vector<Tensor<1, dim>> shape_stokes_velocity_face(
                                        stokes_dofs_per_cell);
                                std::vector<Tensor<1, dim>> shape_vs_face(
                                        hydrogel_dofs_per_cell); // mesh velocity
                                std::vector<Tensor<1, dim>> shape_vf_face(
                                        hydrogel_dofs_per_cell); // displacement
                                std::vector<Tensor<1, dim>> shape_u_face(
                                        hydrogel_dofs_per_cell);
                                std::vector<Tensor<2, dim>> shape_grad_u_face(
                                        hydrogel_dofs_per_cell);

                                for (unsigned int i = 0; i < stokes_dofs_per_cell; ++i)
                                {
                                    shape_stokes_velocity_face[i] =
                                            scratch
                                                    .stokes_fe_face_values
                                            [extractor_stokes_velocity]
                                                    .value(i, q);
                                    shape_u_face[i] =
                                            scratch
                                                    .stokes_fe_face_values[extractor_displacement]
                                                    .value(i, q);
                                    shape_grad_u_face[i] =
                                            scratch
                                                    .stokes_fe_face_values[extractor_displacement]
                                                    .gradient(i, q);
                                }
                                for (unsigned int i = 0; i < hydrogel_dofs_per_cell;
                                     ++i)
                                {
                                    shape_vs_face[i] = scratch
                                            .hydrogel_fe_face_values
                                    [extractor_mesh_velocity]
                                            .value(i, q);
                                    shape_vf_face[i] = scratch
                                            .hydrogel_fe_face_values
                                    [extractor_hydrogel_velocity]
                                            .value(i, q);
                                }

                                const auto normal_vector =
                                        scratch.hydrogel_fe_face_values.normal_vector(
                                                q); // normal vector in reference domain
                                const Tensor<1, dim> fn = inv_F_T * normal_vector;
                                const double         magnitude_m = fn.norm();
                                const Tensor<1, dim> fnn =
                                        fn / magnitude_m; // normalized n in physical domain
                                const double jxwq =
                                        scratch.hydrogel_fe_face_values.JxW(q);

                                //use boundary condition #2

                                const auto term9 =
                                        [&](const Tensor<1, dim> trial_function,
                                            const Tensor<1, dim> test_function) {
                                            return jxwq *
                                                   ((inv_F * trial_function) * normal_vector) *
                                                   ((inv_F * test_function) * normal_vector) *
                                                   jacobian / ((eta/Ca) * magnitude_m);
                                        };
                                const auto term10 =
                                        [&](const Tensor<1, dim> trial_function,
                                            const Tensor<1, dim> test_function) {
                                            return jxwq *
                                                   (trial_function -
                                                    (trial_function * fnn) * fnn) *
                                                   test_function * magnitude_m * jacobian /
                                                   (beta_0/Ca);
                                        };
                                const auto term11 =
                                        [&](const Tensor<1, dim> trial_function,
                                            const Tensor<1, dim> test_function) {
                                            return jxwq * volume_s[q] * volume_s[q] *
                                                   (trial_function -
                                                    (trial_function * fnn) * fnn) *
                                                   test_function * magnitude_m * jacobian /
                                                   (beta_i/Ca);
                                        };

                                //(stokes,stokes)
                                for (unsigned int i = 0; i < stokes_dofs_per_cell; ++i)
                                {
                                    copy_data.interface_rhs1(i) -=
                                            term9(stokes_velocity_face[q],
                                                  shape_stokes_velocity_face[i]) +
                                            term10(stokes_velocity_face[q],
                                                   shape_stokes_velocity_face[i]);

                                    if (update_matrix)
                                        for (unsigned int j = 0; j < stokes_dofs_per_cell;
                                             ++j)
                                        {
                                            copy_data.interface_matrix1(i, j) +=
                                                    term9(shape_stokes_velocity_face[j],
                                                          shape_stokes_velocity_face[i]) +
                                                    term10(shape_stokes_velocity_face[j],
                                                           shape_stokes_velocity_face[i]);
                                        }
                                }

                                //(stokes,hydrogel)
                                for (unsigned int i = 0; i < hydrogel_dofs_per_cell;
                                     ++i)
                                {
                                    copy_data.interface_rhs2(i) -=
                                            term9(stokes_velocity_face[q],
                                                  -shape_vf_face[i]) +
                                            term10(stokes_velocity_face[q],
                                                   -shape_vf_face[i]);
                                    if (update_matrix)
                                        for (unsigned int j = 0; j < stokes_dofs_per_cell;
                                             ++j)
                                        {
                                            copy_data.interface_matrix2(i, j) +=
                                                    term9(shape_stokes_velocity_face[j],
                                                          -shape_vf_face[i]) +
                                                    term10(shape_stokes_velocity_face[j],
                                                           -shape_vf_face[i]);
                                        }
                                }

                                //(hydrogel,stokes)
                                for (unsigned int i = 0; i < stokes_dofs_per_cell; ++i)
                                {
                                    copy_data.interface_rhs3(i) -=
                                            term9(-vf_face[q],
                                                  shape_stokes_velocity_face[i]) +
                                            term10(-vf_face[q],
                                                   shape_stokes_velocity_face[i]);
                                    if (update_matrix)
                                        for (unsigned int j = 0;
                                             j < hydrogel_dofs_per_cell;
                                             ++j)
                                        {
                                            copy_data.interface_matrix3(i, j) +=
                                                    term9(-shape_vf_face[j],
                                                          shape_stokes_velocity_face[i]) +
                                                    term10(-shape_vf_face[j],
                                                           shape_stokes_velocity_face[i]);
                                        }
                                }

                                //(hydrogel,hydrogel)
                                for (unsigned int i = 0; i < hydrogel_dofs_per_cell;
                                     ++i)
                                {
                                    copy_data.interface_rhs4(i) -=
                                            term9(vf_face[q], shape_vf_face[i]) +
                                            term10(vf_face[q], shape_vf_face[i]) +
                                            term11(vf_face[q] - vs_face[q],
                                                   shape_vf_face[i] - shape_vs_face[i]); //+
                                    //                      term14(grad_u_face[q],
                                    //                      shape_u_face[i]);
                                    if (update_matrix)
                                        for (unsigned int j = 0;
                                             j < hydrogel_dofs_per_cell;
                                             ++j)
                                        {
                                            copy_data.interface_matrix4(i, j) +=
                                                    term9(shape_vf_face[j], shape_vf_face[i]) +
                                                    term10(shape_vf_face[j], shape_vf_face[i]) +
                                                    term11(shape_vf_face[j] - shape_vs_face[j],
                                                           shape_vf_face[i] -
                                                           shape_vs_face[i]); // +
                                            //                          term14(shape_grad_u_face[j],
                                            //                          shape_u_face[i]);
                                        }
                                }


                            } // quadrature loop


                        }
                    } // face loop

        } // hydrogel domain
    }

    // constraints on interface terms matrix and rhs
    template <int dim>
    void
    FluidStructureProblem<dim>::copy_local_to_global_hp(
            const PerTaskData &copy_data,
            const bool         update_matrix)
    {
        if (update_matrix)
        {
            constraints_newton_update.distribute_local_to_global(
                    copy_data.cell_matrix,
                    copy_data.cell_rhs,
                    copy_data.local_dof_indices,
                    system_matrix,
                    system_rhs);
            if (copy_data.assemble_interface)
            {
                //(stokes,stokes)
                constraints_newton_update.distribute_local_to_global(
                        copy_data.interface_matrix1,
                        copy_data.interface_rhs1,
                        copy_data.neighbor_dof_indices,
                        system_matrix,
                        system_rhs);

                //(stokes,hydrogel)
                constraints_newton_update.distribute_local_to_global(
                        copy_data.interface_matrix2,
                        copy_data.local_dof_indices,
                        copy_data.neighbor_dof_indices,
                        system_matrix);
                //(stokes,hydrogel)
                constraints_newton_update.distribute_local_to_global(
                        copy_data.interface_rhs2,
                        copy_data.local_dof_indices,
                        system_rhs);

                //(hydrogel,stokes)
                constraints_newton_update.distribute_local_to_global(
                        copy_data.interface_matrix3,
                        copy_data.neighbor_dof_indices,
                        copy_data.local_dof_indices,
                        system_matrix);
                //(hydrogel,stokes)
                constraints_newton_update.distribute_local_to_global(
                        copy_data.interface_rhs3,
                        copy_data.neighbor_dof_indices,
                        system_rhs);

                //(hydrogel,hydrogel)
                constraints_newton_update.distribute_local_to_global(
                        copy_data.interface_matrix4,
                        copy_data.interface_rhs4,
                        copy_data.local_dof_indices,
                        system_matrix,
                        system_rhs);
            }
        }
        else
        {
            constraints_newton_update.distribute_local_to_global(
                    copy_data.cell_rhs, copy_data.local_dof_indices, system_rhs);
            if (copy_data.assemble_interface)
            {
                //(stokes,stokes)
                constraints_newton_update.distribute_local_to_global(
                        copy_data.interface_rhs1,
                        copy_data.neighbor_dof_indices,
                        system_rhs);
                //(stokes,hydrogel)
                constraints_newton_update.distribute_local_to_global(
                        copy_data.interface_rhs2,
                        copy_data.local_dof_indices,
                        system_rhs);
                //(hydrogel,stokes)
                constraints_newton_update.distribute_local_to_global(
                        copy_data.interface_rhs3,
                        copy_data.neighbor_dof_indices,
                        system_rhs);
                //(hydrogel,hydrogel)
                constraints_newton_update.distribute_local_to_global(
                        copy_data.interface_rhs4,
                        copy_data.local_dof_indices,
                        system_rhs);
            }
        }
    }

    template <int dim>
    void
    FluidStructureProblem<dim>::assemble_system_workstream(
            const bool update_matrix)
    {
        TimerOutput::Scope timing_section(computing_timer, "assemble hp system");
        if (update_matrix)
            system_matrix = 0;

        system_rhs = 0;

        using CellFilter =
                FilteredIterator<typename DoFHandler<2>::active_cell_iterator>;
         const QGauss<dim> quadrature_formula(2 + 2 * velocity_degree);
        // same quadrature for all
        const hp::QCollection<dim> q_collection{quadrature_formula,
                                                quadrature_formula};
        const QGauss<dim - 1>      face_quadrature_formula(2 + 2 * velocity_degree);

        const UpdateFlags hp_update_flags = update_values | update_gradients |
                                            update_JxW_values |
                                            update_quadrature_points;
        const UpdateFlags face_update_flags =
                update_values | update_gradients | update_normal_vectors |
                update_JxW_values | update_quadrature_points;
        const UpdateFlags volume_update_flag = update_values | update_gradients |
                                               update_JxW_values |
                                               update_quadrature_points;

        ScratchData sd(fe_collection,
                       volume_fe_collection,
                       mapping_collection,
                       stokes_fe,
                       hydrogel_fe,
                       volume_fe,
                       q_collection,
                       q_collection,
                       face_quadrature_formula,
                       hp_update_flags,
                       volume_update_flag,
                       face_update_flags);
        PerTaskData cp;
        auto worker =
                [this, &update_matrix = std::as_const(update_matrix)](const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                                      ScratchData &                                         scratch,
                                                                      PerTaskData &                                         copy_data) {
                    this->local_assemble_hp(cell, scratch, copy_data, update_matrix);
                };
        auto copier = [this, &update_matrix = std::as_const(update_matrix)](const PerTaskData &copy_data) {
                   this->copy_local_to_global_hp(copy_data, update_matrix);
        };

        WorkStream::run(CellFilter(IteratorFilters::LocallyOwnedCell(),
                                   dof_handler.begin_active()),
                        CellFilter(IteratorFilters::LocallyOwnedCell(),
                                   dof_handler.end()),
                        worker, copier, sd, cp);

        system_matrix.compress(VectorOperation::add);
        system_rhs.compress(VectorOperation::add);
        //pcout << "Number of non-zero elements: " << system_matrix.n_nonzero_elements() << std::endl;
    }

    template <int dim>
    void
    FluidStructureProblem<dim>::
    compute_cauchy_stress()
    {
        const auto &unit_support_points =
                volume_fe.get_unit_support_points();

        // construct fe values to get normal vectors and components
        const Quadrature<dim> supporting_quadrature(
                unit_support_points);
        hp::QCollection<dim> q_collection;
        q_collection.push_back(supporting_quadrature);
        q_collection.push_back(supporting_quadrature);

        hp::FEValues<dim>      fe_values(mapping_collection,
                                         fe_collection,
                                         q_collection,
                                         update_values |
                                         update_gradients |
                                         update_quadrature_points);
        /*hp::FEValues<dim>      volume_fe_values(mapping_collection,
                                                volume_fe_collection,
                                                q_collection,
                                                update_values |
                                                update_quadrature_points);*/

        std::vector<bool> dof_is_touched(dof_handler.n_dofs(),
                                         false); // set to true if is computed

        for (const auto &cell_hp : dof_handler.active_cell_iterators())
        {
            if (cell_hp->is_locally_owned())
            {
                if (cell_is_in_hydrogel_domain(cell_hp))
                {
                    typename DoFHandler<dim>::active_cell_iterator cell_volume(
                            &triangulation, cell_hp->level(), cell_hp->index(), &volume_dof_handler);

                    fe_values.reinit(cell_hp);
                    //volume_fe_values.reinit(cell_volume);

                    const FEValues<dim> & hydrogel_fe_values = fe_values.get_present_fe_values();
                    //const FEValues<dim> & volume_fe_values = volume_fe_values.get_present_fe_values();

                    const unsigned int n_q_points =
                            hydrogel_fe_values.n_quadrature_points;

                    //get displacement values
                    std::vector<Tensor<1, dim>> displacement(n_q_points);
                    hydrogel_fe_values[extractor_displacement].get_function_values(
                            solution, displacement);
                    //get grad u
                    std::vector<Tensor<2, dim>> grad_u(n_q_points); // \nabla u
                    hydrogel_fe_values[extractor_displacement].get_function_gradients(
                            solution, grad_u);
                    //get vf values
                    std::vector<Tensor<1, dim>> vf(n_q_points);
                    hydrogel_fe_values[extractor_hydrogel_velocity].get_function_values(
                            solution, vf);
                    //gradient vf
                    std::vector<Tensor<2, dim>> grad_vf(n_q_points);
                    hydrogel_fe_values[extractor_hydrogel_velocity].get_function_gradients(
                            solution, grad_vf);
                    //get interior pressure p_2
                    std::vector<double> p2(n_q_points);
                    hydrogel_fe_values[extractor_hydrogel_pressure].get_function_values(
                            solution, p2);

                    std::vector<Tensor<1, dim>> grad_p(n_q_points);
                    hydrogel_fe_values[extractor_hydrogel_pressure].get_function_gradients(
                            solution, grad_p);

                    //get volume fraction
                    /*std::vector<double> volume_s(n_q_points);
                    volume_fe_values.get_function_values(volume_solution, volume_s);*/

                    //hydrogel domain
                    const auto &qpoints = hydrogel_fe_values.get_quadrature_points();

                    const unsigned int volume_dofs_per_cell =
                            cell_volume->get_fe().dofs_per_cell;
                    std::vector<types::global_dof_index> volume_local_dof_indices(
                            volume_dofs_per_cell);
                    cell_volume->get_dof_indices(volume_local_dof_indices);

                    SymmetricTensor<2, dim> identity_tensor;
                    for (unsigned int d = 0; d < dim; d++)
                    {
                        identity_tensor[d][d] = 1.0;
                    }

                    for(unsigned int q=0; q<n_q_points; ++q)
                    {
                        Tensor<2,dim> deformation_F(identity_tensor);
                        deformation_F += grad_u[q];
                        const double jacobian = determinant(deformation_F);
                        const auto           inv_F = invert(deformation_F); // F inverse
                        const auto           inv_F_T = transpose(inv_F);
                        const Tensor<2, dim> gradu_Finv   = grad_u[q] * inv_F;
                        const Tensor<2, dim> gradu_Finv_T = transpose(gradu_Finv);

                        const Tensor<2, dim> grad_vfq = grad_vf[q];
                        const Tensor<2, dim> grad_vfq_t = transpose(grad_vfq);
                        const Tensor<2, dim> gradvf_Finv   = grad_vf[q] * inv_F;
                        const Tensor<2, dim> gradvf_Finv_T = transpose(gradvf_Finv);

                        Tensor<2, dim> solid_stress_2d = 2000.* 1./jacobian *
                                                          (   mu_s * (deformation_F - inv_F_T) +
                                                              lambda_s * jacobian * (jacobian - 1.0) *
                                                              inv_F_T              ) *
                                                          transpose(deformation_F) ;
                        // almansi Strain 0.5(I- F^ -T * F^-1)
                        /*Tensor<2, dim> strain_2d(identity_tensor) ;
                        strain_2d -= inv_F_T * inv_F;
                        strain_2d *= 0.5;*/

                        //Green strain 0.5*(F^T - F - I)
                        Tensor<2, dim> strain_2d(identity_tensor) ;
                        strain_2d -=  transpose(deformation_F) * deformation_F;
                        strain_2d *= -0.5;

                        Tensor<2, dim> fluid_stress_2d =
                               2000. * 1e-3 * vis_BM * (gradvf_Finv + gradvf_Finv_T);

                        /*Tensor<2, dim> stress_total_2d =
                                volume_s[q] * solid_stress_2d +
                                (1.0 - volume_s[q]) * fluid_stress_2d -
                                p2[q] * identity_tensor;*/
                    // energy W = c1 (I1 - 3 - 2 ln J) + D1 (J-1)^2
                        Tensor<2, dim> right_green = transpose(deformation_F) * deformation_F;
                        auto strain_energy = 0.5 * mu_s * (trace(right_green) - 3.0 -2 * log(jacobian))
                                             + 0.5 * lambda_s * (jacobian - 1.) * (jacobian - 1.);

                        const unsigned int volume_global_id = volume_local_dof_indices[q];
                        if (!dof_is_touched[volume_global_id])
                        {
                            el_energy(volume_global_id)  = strain_energy;
                            stress_s[0](volume_global_id) = solid_stress_2d[0][0];
                            stress_s[1](volume_global_id) = solid_stress_2d[1][1];
                            stress_s[2](volume_global_id) = solid_stress_2d[0][1];
                            stress_s[3](volume_global_id) = solid_stress_2d[0][0] +
                                                            solid_stress_2d[1][1];
                            stress_s[4](volume_global_id) = solid_stress_2d[0][0] *
                                                            solid_stress_2d[1][1] - solid_stress_2d[0][1] *
                                                                                    solid_stress_2d[0][1];
                            /*stress_s[3](volume_global_id) = grad_p[q][0];
                            stress_s[4](volume_global_id) = grad_p[q][1];*/
                            stress_s[5](volume_global_id) = std::pow( (
                                                                              std::pow((solid_stress_2d[0][0] + solid_stress_2d[1][1]), 2)
                                                                              -3 * (solid_stress_2d[0][0] * solid_stress_2d[1][1]
                                                                                    - solid_stress_2d[0][1] * solid_stress_2d[0][1]) ),  0.5);

                            stress_f[0](volume_global_id) = fluid_stress_2d[0][0];
                            stress_f[1](volume_global_id) = fluid_stress_2d[1][1];
                            stress_f[2](volume_global_id) = fluid_stress_2d[0][1];
                            stress_f[3](volume_global_id) = fluid_stress_2d[0][0] +
                                                            fluid_stress_2d[1][1];
                            stress_f[4](volume_global_id) = fluid_stress_2d[0][0] *
                                                            fluid_stress_2d[1][1] - fluid_stress_2d[0][1] *
                                                                                    fluid_stress_2d[0][1];
                            stress_f[5](volume_global_id) = std::pow( (
                                                                              std::pow((fluid_stress_2d[0][0] + fluid_stress_2d[1][1]), 2)
                                                                              -3 * (fluid_stress_2d[0][0] * fluid_stress_2d[1][1]
                                                                                    - fluid_stress_2d[0][1] * fluid_stress_2d[0][1]) ),  0.5);

                            /*stress_T[0](volume_global_id) = stress_total_2d[0][0];
                            stress_T[1](volume_global_id) = stress_total_2d[1][1];
                            stress_T[2](volume_global_id) = stress_total_2d[0][1];
                            stress_T[3](volume_global_id) = stress_total_2d[0][0] +
                                                              stress_total_2d[1][1];
                            stress_T[4](volume_global_id) = stress_total_2d[0][0] *
                                    stress_total_2d[1][1] - stress_total_2d[0][1] *
                                    stress_total_2d[0][1];
                            stress_T[5](volume_global_id) = std::pow( (
                                    std::pow((stress_total_2d[0][0] + stress_total_2d[1][1]), 2)
                                    -3 * (stress_total_2d[0][0] * stress_total_2d[1][1]
                                    - stress_total_2d[0][1] * stress_total_2d[0][1]) ),  0.5);*/

                            strain[0](volume_global_id) = strain_2d[0][0];
                            strain[1](volume_global_id) = strain_2d[1][1];
                            strain[2](volume_global_id) = strain_2d[0][1];
                            strain[3](volume_global_id) = strain_2d[0][0] + strain_2d[1][1];
                            strain[4](volume_global_id) = strain_2d[0][0] *
                                                          strain_2d[1][1] - strain_2d[0][1] *
                                                                            strain_2d[0][1];
                            strain[5](volume_global_id) = std::pow( (
                                                                            std::pow((strain_2d[0][0] + strain_2d[1][1]), 2)
                                                                            -3 * (strain_2d[0][0] * strain_2d[1][1]
                                                                                  - strain_2d[0][1] * strain_2d[0][1]) ),  0.5);
                            dof_is_touched[volume_global_id] = true;
                        }
                    }
                }

            }

        }
    }
//change this later for parallel computing
    template <int dim>
    void
    FluidStructureProblem<dim>::newton_iteration()
    {
        TimerOutput::Scope timing_section(computing_timer, "newton iteration");
        pcout << " newton iteration... " << std::endl;
        // set the Newton iterate v* to v_n
        current_solution = old_solution;
        dis_current_solution = dis_old_solution;
        // constrain v* using the flux constraint from phis_n+1
        constraints_hp.distribute(dis_current_solution);
        current_solution = dis_current_solution;

        const unsigned int max_iteration = 50;
        unsigned int       iteration     = 0;
        const double       tol           = 1e-10;
        const double       alpha_min     = 0.01;
        assemble_system_workstream(false);
        double residual_hp = system_rhs.l2_norm();

        double g1,g2,g3,g0,g_final; // the l2 norm for the rhs u_k
        double alpha_0, alpha_2, alpha_3, alpha_final;
        double h1, h2, h3;

        pcout << " initial residual = " << residual_hp << std::endl;
        if (residual_hp<tol)
        {
            pcout<<"Simulation has converged!"<<std::endl;
            abort();
        }

        while (iteration < max_iteration && residual_hp > tol)
        {
            alpha_3 = 1;
            TrilinosWrappers::MPI::Vector u_k(dis_current_solution);
            assemble_system_workstream(true);

            g1 = system_rhs.l2_norm();
            dis_newton_update = system_rhs;

            solver(system_matrix, dis_newton_update, system_rhs);

            constraints_newton_update.distribute(dis_newton_update);
            current_solution = dis_current_solution;
            TrilinosWrappers::MPI::Vector du_k(dis_newton_update);
            dis_current_solution += dis_newton_update;
            current_solution = dis_current_solution;
            assemble_system_workstream(true);
            g3 = system_rhs.l2_norm();
            alpha_final = 1;
            g_final = g3;

            if (g3>g1)
            {
                while(g3>g1)
                {
                    alpha_3 = alpha_3/2;
                    dis_current_solution = u_k;
                    current_solution = dis_current_solution;
                    du_k *= alpha_3;
                    dis_current_solution += du_k;
                    current_solution = dis_current_solution;
                    assemble_system_workstream(false);
                    g3 = system_rhs.l2_norm();
                    du_k = dis_newton_update;
                    if (alpha_3<alpha_min)
                    {
                        pcout<<"Newton Iteration is UNCOPELED!"<<std::endl;
                        abort();
                    }
                }

                alpha_2 = alpha_3/2;
                dis_current_solution = u_k;
                current_solution = dis_current_solution;
                du_k *= alpha_2;
                dis_current_solution += du_k;
                current_solution = dis_current_solution;
                assemble_system_workstream(false);
                g2 = system_rhs.l2_norm();
                du_k = dis_newton_update;


                h1 = (g2-g1)/alpha_2;
                h2 = (g3-g2)/(alpha_3-alpha_2);
                h3 = (h2-h1)/alpha_3;
                alpha_0 = 0.5*(alpha_2 - h1/h3);

                dis_current_solution = u_k;
                current_solution = dis_current_solution;
                du_k *= alpha_0;
                dis_current_solution += du_k;
                current_solution = dis_current_solution;
                assemble_system_workstream(false);
                g0 = system_rhs.l2_norm();
                du_k = dis_newton_update;

                alpha_final = alpha_0;
                g_final = g0;
                if (g2 < g0)
                {
                    alpha_final = alpha_2;
                    g_final = g2;
                    if (g3 < g2)
                    {
                        alpha_final = alpha_3;
                        g_final = g3;
                    }

                }
                else if (g3 < g0)
                {
                    alpha_final = alpha_3;
                    g_final = g3;
                    if (g2 < g3)
                    {
                        alpha_final = alpha_2;
                        g_final = g2;
                    }
                }
            }

            dis_current_solution = u_k;
            current_solution = dis_current_solution;
            du_k *= alpha_final;
            dis_current_solution += du_k;
            current_solution = dis_current_solution;
            residual_hp=g_final;

            pcout << " k= " << iteration << " residual = " << g_final
                  << " alpha= " << alpha_final
                  << std::endl;
            iteration++;
        }
        dis_solution = dis_current_solution;
        current_solution = dis_current_solution;
        solution = current_solution;
        pcout << " newton iteration done " << std::endl;
    }

    template <int dim>
    void
    FluidStructureProblem<dim>::output_results(
            const unsigned int step_number) const
    {
        //Timer timer;
       // std::string filename = filename_base;
     //   filename +="_quad";
        std::vector<std::string> solution_names(dim, "displacement");
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
                data_component_interpretation(
                dim, DataComponentInterpretation::component_is_part_of_vector);

        for (unsigned int d = extractor_mesh_velocity.first_vector_component;
             d < extractor_mesh_velocity.first_vector_component + dim;
             ++d)
        {
            solution_names.emplace_back("v_s");
            data_component_interpretation.push_back(
                    DataComponentInterpretation::component_is_part_of_vector);
        }

        for (unsigned int d = extractor_hydrogel_velocity.first_vector_component;
             d < extractor_hydrogel_velocity.first_vector_component + dim;
             ++d)
        {
            solution_names.emplace_back("v_f");
            data_component_interpretation.push_back(
                    DataComponentInterpretation::component_is_part_of_vector);
        }

        solution_names.emplace_back("p");
        data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);

        for (unsigned int d = extractor_stokes_velocity.first_vector_component;
             d < extractor_stokes_velocity.first_vector_component + dim;
             ++d)
        {
            solution_names.emplace_back("V");
            data_component_interpretation.push_back(
                    DataComponentInterpretation::component_is_part_of_vector);
        }

        solution_names.emplace_back("P");
        data_component_interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);

        std::vector<std::string> volume_names(1, "phi_s");
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
                volume_data_component_interpretation(
                1, DataComponentInterpretation::component_is_scalar);

        DataOut<dim> data_out;
        //    data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(
                dof_handler,
                solution,
                solution_names,
                data_component_interpretation);

        data_out.add_data_vector(
                volume_dof_handler,
                volume_solution,
                volume_names,
                volume_data_component_interpretation);

        std::vector<std::vector<std::string>> strain_names(6);
        strain_names[0] = {"strain_xx"};
        strain_names[1] = {"strain_yy"};
        strain_names[2] = {"strain_xy"};
        strain_names[3] = {"strain_I1"};
        strain_names[4] = {"strain_I2"};
        strain_names[5] = {"strain_von"};

        std::vector<std::vector<std::string>> stress_s_names(6);
        stress_s_names[0] = {"Stress_s_xx"};
        stress_s_names[1] = {"Stress_s_yy"};
        stress_s_names[2] = {"Stress_s_xy"};
        stress_s_names[3] = {"Stress_s_I1"};
        stress_s_names[4] = {"Stress_s_I2"};
        stress_s_names[5] = {"Stress_s_von"};

        std::vector<std::vector<std::string>> stress_f_names(6);
        stress_f_names[0] = {"Stress_f_xx"};
        stress_f_names[1] = {"Stress_f_yy"};
        stress_f_names[2] = {"Stress_f_xy"};
        stress_f_names[3] = {"Stress_f_I1"};
        stress_f_names[4] = {"Stress_f_I2"};
        stress_f_names[5] = {"Stress_f_von"};

        for(unsigned int d=0; d<6; ++d)
        {
            data_out.add_data_vector(
                    volume_dof_handler,
                    strain[d],
                    strain_names[d],
                    volume_data_component_interpretation);
            data_out.add_data_vector(
                    volume_dof_handler,
                    stress_f[d],
                    stress_f_names[d],
                    volume_data_component_interpretation);
            data_out.add_data_vector(
                    volume_dof_handler,
                    stress_s[d],
                    stress_s_names[d],
                    volume_data_component_interpretation);
        }

        /*std::vector<std::string> strain_energy(1, "el_energy");
        strain_energy = {"el_energy"};
        data_out.add_data_vector(
                volume_dof_handler,
                el_energy,
                strain_energy,
                volume_data_component_interpretation);*/

        data_out.build_patches();

        std::ofstream output("solution-" +
                             Utilities::int_to_string(step_number, 5) + ".vtk");
        data_out.write_vtk(output);
        /*data_out.write_vtu_with_pvtu_record(
                "./", "solution", step_number,
                MPI_COMM_WORLD, 5);*/

             //timer.stop();
        //std::cout << "done (" << timer.cpu_time() << "s)" << std::endl;
    }

    //3 kinds of constraints 1.interface flux  2.hp boundary 3.newton
    // Updates constraints using volume_solution (phi_n+1)
    template <int dim>
    void
    FluidStructureProblem<dim>::update_constraints(const IndexSet &hp_relevant_set)
    {
        constraints_flux.clear();
        constraints_flux.reinit(hp_relevant_set);
        make_flux_constraints(constraints_flux);
        constraints_flux.close();

        constraints_hp.clear();
        constraints_hp.reinit(hp_relevant_set);
        constraints_hp.merge(constraints_hp_nonzero);
        constraints_hp.merge(constraints_flux,
                             AffineConstraints<double>::MergeConflictBehavior::left_object_wins);
        constraints_hp.close();

        constraints_newton_update.clear();
        constraints_newton_update.reinit(hp_relevant_set);
        constraints_newton_update.merge(constraints_boundary);
        constraints_newton_update.merge(constraints_flux,
                                        AffineConstraints<double>::MergeConflictBehavior::left_object_wins);
        constraints_newton_update.close();
    }

    template <int dim>
    void
    FluidStructureProblem<dim>::apply_initial_condition_hp()
    {
        VectorTools::interpolate(*mapping_pointer,
                                 volume_dof_handler,
                                 Functions::ConstantFunction<dim>(phi_s0),
                                 dis_volume_solution);

        constraints_volume.distribute(dis_volume_solution);
        volume_solution = dis_volume_solution;
        dis_volume_old_solution = dis_volume_solution;
        volume_old_solution = volume_solution;
    }

    template <int dim>
    void
    FluidStructureProblem<dim>::create_dofs_map(const std::string & filename,
                                                const ComponentMask mask)
    {
        std::ofstream out(filename);
        out << "plot '-' using 1:2 with lines, "
            << "'-' with labels point pt 2 offset 1,1" << std::endl;
        GridOut().write_gnuplot(triangulation, out);
        out << "e" << std::endl;
        std::map<types::global_dof_index, Point<dim>> physical_support_points;

        DoFTools::map_dofs_to_support_points(mapping_collection,
                                             dof_handler,
                                             physical_support_points,
                                             mask);
        DoFTools::write_gnuplot_dof_support_point_info(out,
                                                       physical_support_points);
        out << "e" << std::endl;
    }

    template <int dim>
    void
    FluidStructureProblem<dim>::run()
    {
        const unsigned int n_refinement = 1;
        make_grid(n_refinement);
        setup_dofs();
        set_interface_dofs_flag(interface_dofs_flag, hp_index_set);
        output_results(0);

        uint         step       = 0;
        double       time       = 0.;
        const double final_time = 20.0;//20000. * static_cast<double>(time_step);

        std::string   fileNameBaseLsns;
        std::ofstream myfile;
        myfile.open(("output_variables" + fileNameBaseLsns + ".dat").c_str());
        myfile << std::fixed;
        pcout << "going to do loop" << std::endl;
        do
        {
            // compute time step?
            pcout << "\n step: " << step << " time: " << time << std::endl;
            {
                assemble_volume_system_workstream();
                solve_volume();
                // when phi_s changed, matching conditions are changed

                {
                    update_constraints(hp_relevant_set);
                    setup_hp_sparse_matrix(hp_index_set,
                                           hp_relevant_set);
                    pcout << "setup_hp_sparse_matrix" << std::endl;
                }
                newton_iteration();
            }

            dis_volume_old_solution = dis_volume_solution;
            volume_solution = dis_volume_solution;
            volume_old_solution = volume_solution;
            dis_old_solution = dis_solution;
            solution = dis_solution;
            old_solution        = solution;
            ++step;
            if (step % 1 == 0)
            {
                compute_cauchy_stress();
                output_results(step);
            }
            /*if ((timestep_number > 0) && (timestep_number % 4 == 0))
                computing_timer.print_summary();*/
            time += time_step;
            ++timestep_number;
        }
        while (time < final_time);
        myfile.close();
    }
} // namespace HP_ALE

int main(int argc, char *argv[])
{
    try
    {
        using namespace HP_ALE;
        //dealii::MultithreadInfo::set_thread_limit(32);
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                            numbers::invalid_unsigned_int);
        //Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 32);
        const TestCase test_case = TestCase::geo1;
        //pcout << "running " << enum_str[static_cast<int>(test_case)]
        //<< std::endl;

        FluidStructureProblem<2> flow_problem(2, 1, 2, test_case);
        flow_problem.run();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
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
        std::cerr << std::endl
                  << std::endl
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
