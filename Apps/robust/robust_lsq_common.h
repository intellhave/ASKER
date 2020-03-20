// -*- C++ -*-
#ifndef ROBUST_LSQ_COMMON_H
#define ROBUST_LSQ_COMMON_H

#include "Math/v3d_nonlinlsq.h"
#include "colamd.h"
#include "Math/v3d_ldl_private.h"
#include "Base/v3d_timer.h"
#include <cmath>
#include <iomanip>
#include <fstream>


namespace Robust_LSQ
{

   using namespace std;
   using namespace V3D;

   template <typename Vec>
   inline void
   displayVector_n(Vec const& v, int N, ostream& os = std::cout)
   {
      using namespace std;

      os << "[ ";
      for (int r = 0; r < N; ++r) os << v[r] << " ";
      os << "]" << endl;
   }

   constexpr double damping_value_min = 1e-8;

   struct Psi_Sqrt
   {
         static double fun(double r2)              
         { 
            double r = sqrt(r2);            
            return sqrt(r); 
         }

         static double weight_fun(double r2)       
         { 
            double r = sqrt(r2);

            return 1.0/(2.0*sqrt(r)*r); 
         }
         static double weight_fun_deriv(double r2) { return 0.0; }
         static double gamma_omega_fun(double r2)  { return 0.0; }
         static double get_convex_range()          { return 1e30; }
   };

   struct Psi_L1
   {
         static double fun(double r2)              { return sqrt(r2); }
         static double weight_fun(double r2)       { return 1; }
         static double weight_fun_deriv(double r2) { return 0.0; }
         static double gamma_omega_fun(double r2)  { return 0.0; }
         static double get_convex_range()          { return 1e30; }
   };

   struct Psi_Quadratic
   {
         // static double fun(double r2)              { return r2/2.0; }
         static double fun(double r2)              { return r2/2.0; }
         static double weight_fun(double r2)       { return 1.0; }
         static double weight_fun_deriv(double r2) { return 0.0; }
         static double gamma_omega_fun(double r2)  { return 0.0; }
         static double get_convex_range()          { return 1e30; }
   };

   struct Psi_SmoothTrunc
   {
         static double fun(double r2)              { return 0.25*((r2 <= 1.0) ? r2*(2.0 - r2) : 1.0); }
         static double weight_fun(double r2)       { return std::max(0.0, 1.0 - r2); }
         static double weight_fun_deriv(double r2) { return (r2 <= 1.0) ? -2.0*sqrt(r2) : 0.0; }
         static double gamma_omega_fun(double r2)  { return 0.25 * sqr(Psi_SmoothTrunc::weight_fun(r2)-1.0); }
         static double get_convex_range()          { return sqrt(0.333333333); }
   };

   struct Psi_Tukey
   {
         static double fun(double r2)              { double const u = std::max(0.0, 1.0 - r2); return (1.0-u*u*u)/6.0; }
         static double weight_fun(double r2)       { double const u = std::max(0.0, 1.0 - r2); return u*u; }
         static double weight_fun_deriv(double r2) { return (r2 <= 1.0) ? -4.0*sqrt(r2)*(1.0-r2) : 0.0; }
         static double gamma_omega_fun(double r2)  { return Psi_Tukey::fun(r2) - 0.5*r2*Psi_Tukey::weight_fun(r2); }
   };

   struct Psi_Cauchy
   {
         static double fun(double r2)              { return 0.5*log(1.0 + r2); }
         static double weight_fun(double r2)       { return 1.0/(1.0 + r2); }
         static double weight_fun_deriv(double r2) { return -2.0*sqrt(r2) / sqr(1.0+r2); }
         static double gamma_omega_fun(double r2)  { return Psi_Cauchy::fun(r2) - 0.5*r2*Psi_Cauchy::weight_fun(r2); }
         static double get_convex_range()          { return 1.0; }
   };

   struct Psi_Geman
   {
         static double fun(double r2)              { return r2/(1.0 + r2)/2.0; }
         static double weight_fun(double r2)       { return 1.0/sqr(1.0 + r2); }
         static double weight_fun_deriv(double r2) { return -4.0*sqrt(r2) / (sqr(1.0+r2)*(1.0+r2)); }
         static double gamma_omega_fun(double r2)  { return Psi_Geman::fun(r2) - 0.5*r2*Psi_Geman::weight_fun(r2); }
         static double get_convex_range()          { return sqrt(0.333333333);; }
   };

   struct Psi_Huber
   {
         static double fun(double r2)              { return (r2 <= 1.0) ? 0.5*r2 : sqrt(r2)-0.5; }
         static double weight_fun(double r2)       { return (r2 <= 1.0) ? 1.0 : 1.0/sqrt(r2); }
         static double weight_fun_deriv(double r2) { return (r2 <= 1.0) ? 0.0 : -1.0/r2; }
         static double gamma_omega_fun(double r2)  { return Psi_Huber::fun(r2) - 0.5*r2*Psi_Huber::weight_fun(r2); }
         static double get_convex_range()          { return 1e30; }
   };

   struct Psi_Welsch
   {
#if 0
         static constexpr double r2_max = 1e30;
         static double fun(double r2)              { double const rr2 = std::min(r2, r2_max); return (1.0 - exp(-rr2))/2.0; }
         static double weight_fun(double r2)       { double const rr2 = std::min(r2, r2_max); return std::max(0.0, exp(-rr2)); }
#else
         static double fun(double r2)              { return 0.5*(1.0 - exp(-r2)); }
         static double weight_fun(double r2)       { return exp(-r2); }
         static double weight_fun_deriv(double r2) { return -2.0*sqrt(r2)*exp(-r2); }
         static double gamma_omega_fun(double r2)  { double const ex = exp(-r2); return 0.5*(1.0 - ex - ex*r2); }
         //static double gamma_omega_fun(double r2)  { return Psi_Welsch::fun(r2) - 0.5*r2*Psi_Welsch::weight_fun(r2); }
#endif
         static double get_convex_range()          { return sqrt(0.5); }

         // static double fun(double r2)
         // {
         //    double const rr2 = std::min(r2, r2_max), v = (1.0 - exp(-rr2))/2.0;
         //    int status = std::fpclassify(v);
         //    if (status == FP_NAN)
         //    {
         //       cout << "v = " << v << " is NaN" << endl;
         //    }
         //    else if (status == FP_SUBNORMAL)
         //    {
         //       cout << "v = " << v << " is subnormal" << endl;
         //    }
         //    else if (status == FP_INFINITE)
         //    {
         //       cout << "v = " << v << " is infinite" << endl;
         //    }
         //    return v;
         // }
         // static double weight_fun(double r2)
         // {
         //    double const rr2 = std::min(r2, r2_max), w = std::exp(-rr2);
         //    int status = std::fpclassify(w);
         //    if (status == FP_NAN)
         //    {
         //       cout << "w = " << w << " is NaN" << endl;
         //    }
         //    else if (status == FP_SUBNORMAL)
         //    {
         //       cout << "w = " << w << " is subnormal" << endl;
         //    }
         //    else if (status == FP_INFINITE)
         //    {
         //       cout << "w = " << w << " is infinite" << endl;
         //    }
         //    return w;
         // }
   };

//**********************************************************************

   struct Robust_NLSQ_CostFunction_Base
   {
         Robust_NLSQ_CostFunction_Base(NLSQ_CostFunction &costFun)
            : _costFun(costFun), _nMeasurements(costFun._nMeasurements)
         { }

         virtual void cache_residuals(NLSQ_Residuals &residuals, Vector<double> &errors) const = 0;
         virtual void cache_IRLS_weights(double const scale2, Vector<double> const& errors, Vector<double> &weights) const = 0;
         virtual void cache_One_weights(double const scale2, Vector<double> const& errors, Vector<double> &weights) const = 0;
         virtual void cache_target_costs(double const scale2, Vector<double> const& errors, Vector<double> &costs) const = 0;

         virtual void cache_weight_fun(double const scale2, Vector<double> const& errors, Vector<double> &weights) const = 0;
         virtual void cache_weight_fun_deriv(double const scale2, Vector<double> const& errors, Vector<double> &vals) const = 0;
         virtual void cache_sqrt_psi_values(double const scale2, Vector<double> const& errors, Vector<double> &values) const = 0;
         virtual void cache_gamma_values(double const scale2, Vector<double> const& errors, Vector<double> &values) const = 0;

         virtual double eval_target_cost(double const scale2, Vector<double> const& errors) const = 0;
         virtual double eval_weights_sensitivity(double const scale2, Vector<double> const& errors) const = 0;

         virtual double eval_target_fun(double const scale2, double const r2) const = 0;
         virtual double eval_target_weight(double const scale2, double const r2) const = 0;

         virtual double get_convex_range(double const scale2) const = 0;
         virtual double get_cost_weighting() const { return 1.0; }
         virtual double get_tau2() const { return 1.0; }

         NLSQ_CostFunction &_costFun;
         int const _nMeasurements;
   }; // end Robust_NLSQ_CostFunction_Base

   template <typename Psi>
   struct Robust_NLSQ_CostFunction : public Robust_NLSQ_CostFunction_Base
   {
         Robust_NLSQ_CostFunction(NLSQ_CostFunction &costFun, double tau, double alpha = 1.0)
            : Robust_NLSQ_CostFunction_Base(costFun), _tau(tau), _tau2(tau*tau), _alpha(alpha)
         { }

         virtual void cache_residuals(NLSQ_Residuals &residuals, Vector<double> &errors) const
         {
            _costFun.preIterationCallback();
            _costFun.initializeResiduals();
            _costFun.evalAllResiduals(residuals._residuals);

            for (int k = 0; k < _costFun._nMeasurements; ++k) 
               errors[k] = sqrNorm_L2(residuals._residuals[k]);
         } // end cache_residuals()


         virtual void cache_IRLS_weights(double const scale2, Vector<double> const& errors, Vector<double> &weights) const
         {
            double const tau2 = _tau2 * scale2;
            for (int k = 0; k < errors.size(); ++k) weights[k] = _alpha * Psi::weight_fun(errors[k] / tau2);
            for (int k = 0; k < errors.size(); ++k) weights[k] = sqrt(weights[k]);
         }

         virtual void cache_One_weights(double const scale2, Vector<double> const& errors, Vector<double> &weights) const
         {
            double const tau2 = _tau2 * scale2;
            // for (int k = 0; k < errors.size(); ++k) weights[k] = _alpha * Psi::weight_fun(errors[k] / tau2);
            for (int k = 0; k < errors.size(); ++k) 
            {               
               weights[k] = 1.0;
            }
         }


         virtual void cache_target_costs(double const scale2, Vector<double> const& errors, Vector<double> &costs) const
         {
            double const tau2 = _tau2 * scale2, W = _alpha * tau2;
            for (int k = 0; k < errors.size(); ++k) costs[k] = W * Psi::fun(errors[k] / tau2);
         }

         virtual void cache_weight_fun(double const scale2, Vector<double> const& errors, Vector<double> &weights) const
         {
            double const tau2 = _tau2 * scale2;
            for (int k = 0; k < errors.size(); ++k) weights[k] = Psi::weight_fun(errors[k] / tau2);
         }

         virtual void cache_weight_fun_deriv(double const scale2, Vector<double> const& errors, Vector<double> &vals) const
         {
            double const tau2 = _tau2 * scale2;
            for (int k = 0; k < errors.size(); ++k) vals[k] = Psi::weight_fun_deriv(errors[k] / tau2);
         }

         virtual void cache_sqrt_psi_values(double const scale2, Vector<double> const& errors, Vector<double> &values) const
         {
            double const tau2 = _tau2 * scale2;
            for (int k = 0; k < values.size(); ++k) values[k] = sqrt(Psi::fun(errors[k] / tau2));
         }

         virtual void cache_gamma_values(double const scale2, Vector<double> const& errors, Vector<double> &values) const
         {
            double const tau2 = _tau2 * scale2;
            for (int k = 0; k < values.size(); ++k) values[k] = Psi::gamma_omega_fun(errors[k] / tau2);
         }

         virtual double eval_target_cost(double const scale2, Vector<double> const& errors) const
         {
            double const tau2 = _tau2 * scale2;
            double res = 0;
            for (int k = 0; k < errors.size(); ++k) res += Psi::fun(errors[k] / tau2);
            return _alpha * tau2 * res;
         }

         virtual double eval_weights_sensitivity(double const scale2, Vector<double> const& errors) const
         {
            double const tau2 = _tau2 * scale2;
            double res = 0;
            for (int k = 0; k < errors.size(); ++k) res += sqr(Psi::weight_fun_deriv(errors[k] / tau2));
            return _alpha * res;
         }

         virtual double eval_target_fun(double const scale2, double const r2) const
         {
            double const tau2 = _tau2 * scale2;
            return _alpha * tau2 * Psi::fun(r2 / tau2);
         }

         virtual double eval_target_weight(double const scale2, double const r2) const
         {
            double const tau2 = _tau2 * scale2;
            return Psi::weight_fun(r2 / tau2);
         }

         virtual double get_convex_range(double const scale2) const
         {
            double const tau2 = _tau2 * scale2;
            return Psi::get_convex_range() * sqrt(tau2);
         }

         virtual double get_cost_weighting() const { return _alpha; }

         virtual double get_tau2() const { return _tau2; }

         double const _tau, _tau2, _alpha;
   }; // end struct Robust_NLSQ_CostFunction

//**********************************************************************

   struct Robust_LSQ_Optimizer_Base : public NLSQ_LM_Optimizer
   {
         Robust_LSQ_Optimizer_Base(NLSQ_ParamDesc const& paramDesc,
                                   std::vector<NLSQ_CostFunction *> const& costFunctions,
                                   std::vector<Robust_NLSQ_CostFunction_Base *> const& robustCostFunctions)
            : NLSQ_LM_Optimizer(paramDesc, costFunctions), _robustCostFunctions(robustCostFunctions)
         { }

         virtual void copyToAllParameters(double * dst) = 0;
         virtual void copyFromAllParameters(double const * src) = 0;

         bool solve_JtJ(Vector<double> &Jt_e, Vector<double> &deltaPerm, Vector<double> &delta)
         {
            bool success_LDL = true;

            int const nCols = _JtJ_Parent.size();
            //int const nnz   = _JtJ.getNonzeroCount();
            int const lnz   = _JtJ_Lp.back();

            vector<int> Li(lnz);
            vector<double> Lx(lnz);
            vector<double> D(nCols), Y(nCols);
            vector<int> workPattern(nCols), workFlag(nCols);

            int * colStarts = (int *)_JtJ.getColumnStarts();
            int * rowIdxs   = (int *)_JtJ.getRowIndices();
            double * values = _JtJ.getValues();

            int const d = LDL_numeric(nCols, colStarts, rowIdxs, values,
                                      &_JtJ_Lp[0], &_JtJ_Parent[0], &_JtJ_Lnz[0],
                                      &Li[0], &Lx[0], &D[0],
                                      &Y[0], &workPattern[0], &workFlag[0]);

            if (d == nCols)
            {
               LDL_perm(nCols, &deltaPerm[0], &Jt_e[0], &_perm_JtJ[0]);
               LDL_lsolve(nCols, &deltaPerm[0], &_JtJ_Lp[0], &Li[0], &Lx[0]);
               LDL_dsolve(nCols, &deltaPerm[0], &D[0]);
               LDL_ltsolve(nCols, &deltaPerm[0], &_JtJ_Lp[0], &Li[0], &Lx[0]);
               LDL_permt(nCols, &delta[0], &deltaPerm[0], &_perm_JtJ[0]);
            }
            else
            {
               if (optimizerVerbosenessLevel >= 2)
               {
                  cout << "Robust_LSQ_Optimizer_Base: LDL decomposition failed with d = " << d << ". Increasing lambda." << endl;
               }
               success_LDL = false;
            }
            return success_LDL;
         } // end solve_JtJ()

         std::vector<Robust_NLSQ_CostFunction_Base *> const& _robustCostFunctions;
   }; // end struct Robust_LSQ_Optimizer_Base

//**********************************************************************

   struct Robust_LSQ_Customized_Weights_Optimizer_Base : public Robust_LSQ_Optimizer_Base
   {
         Robust_LSQ_Customized_Weights_Optimizer_Base(NLSQ_ParamDesc const& paramDesc,
                                                      std::vector<NLSQ_CostFunction *> const& costFunctions,
                                                      std::vector<Robust_NLSQ_CostFunction_Base *> const& robustCostFunctions)
            : Robust_LSQ_Optimizer_Base(paramDesc, costFunctions, robustCostFunctions)
         { }

         typedef Robust_LSQ_Optimizer_Base Base;

         void fillHessian(vector<Vector<double> > const& given_weights)
         {
            // Set Hessian to zero
            _hessian.setZero();

            int const nObjs = _costFunctions.size();
            for (int obj = 0; obj < nObjs; ++obj)
            {
               NLSQ_CostFunction& costFun = *_costFunctions[obj];
               
               int const nParamTypes = costFun._usedParamTypes.size();
               int const nMeasurements = costFun._nMeasurements, residualDim = costFun._measurementDimension;

               Matrix<double> H(residualDim, residualDim);

               MatrixArray<int> const& index = *_hessianIndices[obj];
               NLSQ_Residuals   const& residuals = *_residuals[obj];

               Vector<double> const& weights = given_weights[obj];

               vector<int> const& usedParamTypes = costFun._usedParamTypes;

               for (int i1 = 0; i1 < nParamTypes; ++i1)
               {
                  int const t1 = usedParamTypes[i1], dim1 = _paramDesc.dimension[t1];

                  MatrixArray<double> const& Js1 = *residuals._Js[i1];

                  for (int i2 = 0; i2 < nParamTypes; ++i2)
                  {
                     int const t2 = usedParamTypes[i2], dim2 = _paramDesc.dimension[t2];

                     MatrixArray<double> const& Js2 = *residuals._Js[i2];

                     Matrix<double> J1tJ2(dim1, dim2), H_J2(residualDim, dim2);

                     // Ignore non-existent Hessians lying in the lower triangular part.
                     if (!_hessian.Hs[t1][t2]) continue;

                     MatrixArray<double>& Hs = *_hessian.Hs[t1][t2];

                     for (int k = 0; k < nMeasurements; ++k)
                     {
                        int const ix1 = costFun._correspondingParams[k][i1];
                        int const id1 = this->getParamId(t1, ix1);
                        int const ix2 = costFun._correspondingParams[k][i2];
                        int const id2 = this->getParamId(t2, ix2);

#if 1 || defined(ONLY_UPPER_TRIANGULAR_HESSIAN)
                        if (id1 > id2) continue; // only store the upper diagonal blocks
#endif
                        int const n = index[k][i1][i2];
                        assert(n < Hs.count());

                        multiply_At_B(Js1[k], Js2[k], J1tJ2);
                        scaleMatrixIP(weights[k], J1tJ2);

                        addMatricesIP(J1tJ2, Hs[n]);
                     } // end for (k)
                  } // end for (i2)
               } // end for (i1)
            } // end for (obj)
         } // end fillHessian()

         void eval_Jt_e(vector<Vector<double> > const& given_weights, Vector<double>& Jt_e) const
         {
            makeZeroVector(Jt_e);

            int const nObjs = _costFunctions.size();
            for (int obj = 0; obj < nObjs; ++obj)
            {
               NLSQ_CostFunction& costFun = *_costFunctions[obj];
               NLSQ_Residuals& residuals = *_residuals[obj];
               Vector<double> const& weights = given_weights[obj];

               int const nParamTypes   = costFun._usedParamTypes.size();
               int const nMeasurements = costFun._nMeasurements;

               for (int i = 0; i < nParamTypes; ++i)
               {
                  int const paramType = costFun._usedParamTypes[i];
                  int const paramDim = _paramDesc.dimension[paramType];

                  MatrixArray<double> const& J = *residuals._Js[i];

                  Vector<double> Jkt_e(paramDim);

                  for (int k = 0; k < nMeasurements; ++k)
                  {
                     int const id = costFun._correspondingParams[k][i];
                     int const dstRow = _paramTypeRowStart[paramType] + id*paramDim;

                     multiply_At_v(J[k], residuals._residuals[k], Jkt_e);
                     scaleVectorIP(weights[k], Jkt_e);
                     for (int l = 0; l < paramDim; ++l) Jt_e[dstRow + l] += Jkt_e[l];
                  } // end for (k)
               } // end for (i)
            } // end for (obj)
         } // end evalJt_e()
   }; // end struct Robust_LSQ_Customized_Weights_Optimizer_Base

//**********************************************************************

   template <bool init_with_optimal_weights = true>
   struct IRLS_Optimizer : public Robust_LSQ_Optimizer_Base
   {
         IRLS_Optimizer(NLSQ_ParamDesc const& paramDesc,
                           std::vector<NLSQ_CostFunction *> const& costFunctions,
                           std::vector<Robust_NLSQ_CostFunction_Base *> const& robustCostFunctions)
            : Robust_LSQ_Optimizer_Base(paramDesc, costFunctions, robustCostFunctions)
         { }

         double eval_robust_objective() const
         {
            int const nObjs = _robustCostFunctions.size();
            vector<Vector<double> > cached_errors(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
            for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
            return this->eval_current_cost(cached_errors);
         }

         double eval_current_cost(vector<Vector<double> > const& cached_errors) const
         {
            int const nObjs = _robustCostFunctions.size();
            double cost = 0;
            for (int obj = 0; obj < nObjs; ++obj) cost += _robustCostFunctions[obj]->eval_target_cost(1.0, cached_errors[obj]);
            return cost;
         }

         void minimize()
         {
            status = LEVENBERG_OPTIMIZER_TIMEOUT;

            if (_totalParamCount == 0)
            {
               // No degrees of freedom, nothing to optimize.
               if (optimizerVerbosenessLevel >= 2) cout << "IRLS_Optimizer: exiting since d.o.f is zero." << endl;
               status = LEVENBERG_OPTIMIZER_CONVERGED;
               return;
            }

            int const totalParamDimension = _JtJ.num_cols();

            vector<double> x_saved(totalParamDimension);

            Vector<double> Jt_e(totalParamDimension);
            Vector<double> delta(totalParamDimension);
            Vector<double> deltaPerm(totalParamDimension);

            int const nObjs = _robustCostFunctions.size();

            double damping_value = this->tau;

            vector<Vector<double> > cached_errors(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);

            int LDL_failures = 0;
            bool initial_success = false; // true, if at least one iteration was successful

            for (int obj = 0; obj < nObjs; ++obj) fillVector(1.0, _residuals[obj]->_weights);

            for (currentIteration = 0; currentIteration < maxIterations; ++currentIteration)
            {
               for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
               double const initial_cost = this->eval_current_cost(cached_errors);

               if (optimizerVerbosenessLevel >= 1)
                  cout << "IRLS_Optimizer: iteration: " << setw(3) << currentIteration << ", initial |residual|^2 = " << setw(12) << initial_cost << " lambda = " << damping_value << endl;

               bool success_LDL = true;

               for (int obj = 0; obj < nObjs; ++obj)
               {
                  NLSQ_Residuals& residuals = *_residuals[obj];
                  if (init_with_optimal_weights || initial_success)
                     _robustCostFunctions[obj]->cache_IRLS_weights(1.0, cached_errors[obj], residuals._weights);

                  int const K = residuals._weights.size();
                  for (int k = 0; k < K; ++k) scaleVectorIP(residuals._weights[k], residuals._residuals[k]);
               }

               this->fillJacobians();

               this->evalJt_e(Jt_e);
               scaleVectorIP(-1.0, Jt_e);
               double const norm_Linf_Jt_e = norm_Linf(Jt_e);

               if (this->allowStoppingCriteria() && this->applyGradientStoppingCriteria(norm_Linf_Jt_e))
               {
                  if (optimizerVerbosenessLevel >= 2) cout << "IRLS_Optimizer: converged due to gradient stopping," << "norm_Linf_Jt_e = " << norm_Linf_Jt_e << endl;
                  break;
               }

               this->fillHessian();
               success_LDL = 0;

               if (!success_LDL)
               {
                  // Augment the diagonals
                  for (int paramType = 0; paramType < _paramDesc.nParamTypes; ++paramType)
                  {
                     MatrixArray<double>& Hs = *_hessian.Hs[paramType][paramType];
                     vector<pair<int, int> > const& nzPairs = _hessian.nonzeroPairs[paramType][paramType];
                     int const dim = Hs.num_cols(), count = Hs.count();

                     // Only augment those with the same parameter id
                     for (int n = 0; n < count; ++n)
                     {
                        if (nzPairs[n].first != nzPairs[n].second) continue;
                        for (int l = 0; l < dim; ++l) Hs[n][l][l] += damping_value;
                     }
                  } // end for (paramType)
                  this->fillJtJ();
                  success_LDL = this->solve_JtJ(Jt_e, deltaPerm, delta);
                  if (!success_LDL) ++LDL_failures;
               } // end if

               bool success_decrease = false;
               this->copyToAllParameters(&x_saved[0]);

               if (success_LDL)
               {
                  double const deltaSqrLength = sqrNorm_L2(delta);
                  double const paramLength = this->getParameterLength();

                  if (optimizerVerbosenessLevel >= 3)
                     cout << "IRLS_Optimizer: ||delta|| = " << sqrt(deltaSqrLength) << " ||paramLength|| = " << paramLength << endl;

                  if (this->allowStoppingCriteria() && this->applyUpdateStoppingCriteria(paramLength, sqrt(deltaSqrLength)))
                  {
                     if (optimizerVerbosenessLevel >= 2) cout << "IRLS_Optimizer: converged due to small update, deltaSqrLength = " << deltaSqrLength << endl;
                     break;
                  }

                  for (int paramType = 0; paramType < _paramDesc.nParamTypes; ++paramType)
                  {
                     int const paramDim = _paramDesc.dimension[paramType];
                     int const count    = _paramDesc.count[paramType];
                     int const rowStart = _paramTypeRowStart[paramType];

                     VectorArrayAdapter<double> deltaParam(count, paramDim, &delta[0] + rowStart);
                     this->updateParameters(paramType, deltaParam);
                  } // end for (paramType)
                  this->finishUpdateParameters();

                  for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
               } // end if (success_LDL)

               if (success_LDL)
               {
                  // Check if new cost is better than best one at current level
                  double const current_cost = this->eval_current_cost(cached_errors);

                  if (optimizerVerbosenessLevel >= 2) cout << "IRLS_Optimizer: success_LDL = " << int(success_LDL) << " new cost = " << current_cost << endl;

                  success_decrease = true; //(current_cost < initial_cost);
               } // end if (success_LDL)

               if (success_decrease)
               {
                  damping_value = std::max(damping_value_min, damping_value / 10);
               }
               else
               {
                  if (success_LDL) this->copyFromAllParameters(&x_saved[0]);
                  damping_value *= 10;
               }

               initial_success = (initial_success || success_decrease);
            } // end for (currentIteration)

            if (optimizerVerbosenessLevel >= 1 && currentIteration+1 >= maxIterations)
            {
               cout << "IRLS_Optimizer: reached maximum number of iterations, exiting." << endl;
            }

            if (optimizerVerbosenessLevel >= 2)
               cout << "Leaving IRLS_Optimizer::minimize(): LDL_failures = " << LDL_failures << endl;
         } // end minimize()
   }; // end struct IRLS_Optimizer

//**********************************************************************

   // This one is just IRLS, but using the "customized weights" code path
   // It solely exists to compare the two code paths w.r.t. their numerical behavior.
   template <bool init_with_optimal_weights = true>
   struct IRLS_Optimizer2 : public Robust_LSQ_Customized_Weights_Optimizer_Base
   {
         IRLS_Optimizer2(NLSQ_ParamDesc const& paramDesc,
                         std::vector<NLSQ_CostFunction *> const& costFunctions,
                         std::vector<Robust_NLSQ_CostFunction_Base *> const& robustCostFunctions
                         )
            : Robust_LSQ_Customized_Weights_Optimizer_Base(paramDesc, costFunctions, robustCostFunctions)
         { }

         double eval_robust_objective() const
         {
            int const nObjs = _robustCostFunctions.size();
            vector<Vector<double> > cached_errors(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
            for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
            return this->eval_current_cost(cached_errors);
         }

         double eval_current_cost(vector<Vector<double> > const& cached_errors) const
         {
            int const nObjs = _robustCostFunctions.size();
            double cost = 0;
            for (int obj = 0; obj < nObjs; ++obj) cost += _robustCostFunctions[obj]->eval_target_cost(1.0, cached_errors[obj]);
            return cost;
         }

         void minimize()
         {
            status = LEVENBERG_OPTIMIZER_TIMEOUT;

            if (_totalParamCount == 0)
            {
               // No degrees of freedom, nothing to optimize.
               if (optimizerVerbosenessLevel >= 2) cout << "IRLS_Optimizer: exiting since d.o.f is zero." << endl;
               status = LEVENBERG_OPTIMIZER_CONVERGED;
               return;
            }

            int const totalParamDimension = _JtJ.num_cols();

            vector<double> x_saved(totalParamDimension);

            Vector<double> Jt_e(totalParamDimension);
            Vector<double> delta(totalParamDimension);
            Vector<double> deltaPerm(totalParamDimension);

            int const nObjs = _robustCostFunctions.size();

            double damping_value = this->tau;

            vector<Vector<double> > cached_errors(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);

            vector<Vector<double> > cached_weights(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) cached_weights[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
            for (int obj = 0; obj < nObjs; ++obj) fillVector(1.0, cached_weights[obj]);

            int LDL_failures = 0;
            bool initial_success = false; // true, if at least one iteration was successful

            for (int obj = 0; obj < nObjs; ++obj) fillVector(1.0, _residuals[obj]->_weights);

            Timer t("Bundle");
            t.start();
            ofstream log_file("log_IRLS.txt");
            for (currentIteration = 0; currentIteration < maxIterations; ++currentIteration)
            {
               for (int obj = 0; obj < nObjs; ++obj) 
                  _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
               double const initial_cost = this->eval_current_cost(cached_errors);

               if (optimizerVerbosenessLevel >= 1)
               {
                  t.stop();
                  cout << "Time : " << t.getTime() << " IRLS_Optimizer: iteration: " << setw(3) << currentIteration << ", initial |residual|^2 = " << setw(12) << initial_cost << " lambda = " << damping_value << endl;                  
                  log_file << t.getTime() << "\t" << currentIteration << "\t" << initial_cost << endl;
                  t.start();
               }

               bool success_LDL = true;

               for (int obj = 0; obj < nObjs; ++obj)
               {
                  NLSQ_Residuals& residuals = *_residuals[obj];
                  if (init_with_optimal_weights || initial_success)
                  {
                     _robustCostFunctions[obj]->cache_weight_fun(1.0, cached_errors[obj], cached_weights[obj]);
                     scaleVectorIP(_robustCostFunctions[obj]->get_cost_weighting(), cached_weights[obj]);
                  }
               }

               this->fillJacobians();

               this->eval_Jt_e(cached_weights, Jt_e);
               scaleVectorIP(-1.0, Jt_e);
               double const norm_Linf_Jt_e = norm_Linf(Jt_e);

               if (this->allowStoppingCriteria() && this->applyGradientStoppingCriteria(norm_Linf_Jt_e))
               {
                  if (optimizerVerbosenessLevel >= 2) cout << "IRLS_Optimizer: converged due to gradient stopping," << "norm_Linf_Jt_e = " << norm_Linf_Jt_e << endl;
                  break;
               }

               this->fillHessian(cached_weights);
               success_LDL = 0;

               if (!success_LDL)
               {
                  // Augment the diagonals
                  for (int paramType = 0; paramType < _paramDesc.nParamTypes; ++paramType)
                  {
                     MatrixArray<double>& Hs = *_hessian.Hs[paramType][paramType];
                     vector<pair<int, int> > const& nzPairs = _hessian.nonzeroPairs[paramType][paramType];
                     int const dim = Hs.num_cols(), count = Hs.count();

                     // Only augment those with the same parameter id
                     for (int n = 0; n < count; ++n)
                     {
                        if (nzPairs[n].first != nzPairs[n].second) continue;
                        for (int l = 0; l < dim; ++l) Hs[n][l][l] += damping_value;
                     }
                  } // end for (paramType)
                  this->fillJtJ();
                  success_LDL = this->solve_JtJ(Jt_e, deltaPerm, delta);
                  if (!success_LDL) ++LDL_failures;
               } // end if

               bool success_decrease = false;
               this->copyToAllParameters(&x_saved[0]);

               if (success_LDL)
               {
                  double const deltaSqrLength = sqrNorm_L2(delta);
                  double const paramLength = this->getParameterLength();

                  if (optimizerVerbosenessLevel >= 3)
                     cout << "IRLS_Optimizer: ||delta|| = " << sqrt(deltaSqrLength) << " ||paramLength|| = " << paramLength << endl;

                  if (this->allowStoppingCriteria() && this->applyUpdateStoppingCriteria(paramLength, sqrt(deltaSqrLength)))
                  {
                     if (optimizerVerbosenessLevel >= 2) cout << "IRLS_Optimizer: converged due to small update, deltaSqrLength = " << deltaSqrLength << endl;
                     break;
                  }

                  for (int paramType = 0; paramType < _paramDesc.nParamTypes; ++paramType)
                  {
                     int const paramDim = _paramDesc.dimension[paramType];
                     int const count    = _paramDesc.count[paramType];
                     int const rowStart = _paramTypeRowStart[paramType];

                     VectorArrayAdapter<double> deltaParam(count, paramDim, &delta[0] + rowStart);
                     this->updateParameters(paramType, deltaParam);
                  } // end for (paramType)
                  this->finishUpdateParameters();

                  for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
               } // end if (success_LDL)

               if (success_LDL)
               {
                  // Check if new cost is better than best one at current level
                  double const current_cost = this->eval_current_cost(cached_errors);

                  if (optimizerVerbosenessLevel >= 2) cout << "IRLS_Optimizer: success_LDL = " << int(success_LDL) << " new cost = " << current_cost << endl;

                  //success_decrease = true;
                  success_decrease = (current_cost < initial_cost);
               } // end if (success_LDL)

               if (success_decrease)
                  damping_value = std::max(damping_value_min, damping_value / 10);
               else
               {
                  if (success_LDL) this->copyFromAllParameters(&x_saved[0]);
                  damping_value *= 10;
               }

               initial_success = (initial_success || success_decrease);
            } // end for (currentIteration)

            if (optimizerVerbosenessLevel >= 1 && currentIteration+1 >= maxIterations)
            {
               cout << "IRLS_Optimizer: reached maximum number of iterations, exiting." << endl;
            }

            if (optimizerVerbosenessLevel >= 2)
               cout << "Leaving IRLS_Optimizer::minimize(): LDL_failures = " << LDL_failures << endl;
         } // end minimize()
   }; // end struct IRLS_Optimizer

//**********************************************************************

   struct Min_Inliers_IRLS_Optimizer : public Robust_LSQ_Optimizer_Base
   {
         static constexpr bool force_residual_scaling = 0;

         Min_Inliers_IRLS_Optimizer(NLSQ_ParamDesc const& paramDesc,
                                    std::vector<NLSQ_CostFunction *> const& costFunctions,
                                    std::vector<Robust_NLSQ_CostFunction_Base *> const& robustCostFunctions,
                                    double const min_inlier_fraction = 0.5)
            : Robust_LSQ_Optimizer_Base(paramDesc, costFunctions, robustCostFunctions),
              _min_inlier_fraction(min_inlier_fraction)
         {
            int const nObjs = _robustCostFunctions.size();
            int nTotalResiduals = 0;
            for (int obj = 0; obj < nObjs; ++obj) nTotalResiduals += _residuals[obj]->_residuals.count();
            cout << "nTotalResiduals = " << nTotalResiduals << endl;
            _residuals_vec.resize(nTotalResiduals);
         }

         double eval_robust_objective() const
         {
            int const nObjs = _robustCostFunctions.size();
            vector<Vector<double> > cached_errors(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
            for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
            return this->eval_current_cost(cached_errors);
         }

         double eval_current_cost(vector<Vector<double> > const& cached_errors) const
         {
            int const nObjs = _robustCostFunctions.size();
            double cost = 0;
#if 0
            for (int obj = 0; obj < nObjs; ++obj) cost += _robustCostFunctions[obj]->eval_target_cost(1.0, cached_errors[obj]);
#else
            if (1 || force_residual_scaling)
            {
               for (int obj = 0; obj < nObjs; ++obj)
               {
                  auto const& errors = cached_errors[obj];
                  auto const& weights = _residuals[obj]->_weights;
                  int const K = weights.size();
                  double cost1 = 0;
                  for (int k = 0; k < K; ++k) cost1 += 0.5*errors[k]*sqr(weights[k]);
                  cost += cost1;
               }
            }
            else
            {
               // for (int obj = 0; obj < nObjs; ++obj)
               // {
               //    auto const& errors = cached_errors[obj];
               //    auto const& weights = _residuals[obj]->_weights;
               //    int const K = weights.size();

               //    auto costFun = _robustCostFunctions[obj];
               //    double const slope = costFun->eval_target_weight(1.0, cvx_x2) * cvx_x;
               //    double const alpha = std::max(0.0, sqrt(R2) - cvx_x);

               //    for (int k = 0; k < K; ++k)
               //    {
               //       double const r2 = errors[k], r = sqrt(r2);
               //       if (r2 <= cvx_x2)
               //       {
               //          cost1 += costFun->eval_target_fun(1.0, r2);
               //       }
               //       else if (r >= cvx_x + alpha)
               //       {
               //          cost1 += costFun->eval_target_fun(1.0, sqr(r-alpha)) + alpha*slope;
               //       }
               //       else
               //       {
               //          cost1 += psi_cvx_x + slope*(r - cvx_x);
               //       }
               //    } // end for (k)
               //    cost += cost1;
               // } // end for (k)
            } // end if
#endif
            return cost;
         }

         double const _min_inlier_fraction;
         //vector<pair<double, pair<int, int> > > _residuals_vec;
         vector<double> _residuals_vec;

         static constexpr double Tau = 0.5, Tau2 = Tau*Tau;

         double fill_weights(vector<Vector<double> > const& cached_errors)
         {
            int const nObjs = cached_errors.size();

            int pos = 0;
            for (int obj = 0; obj < nObjs; ++obj)
            {
               Vector<double> const& errors = cached_errors[obj];
               int const K = errors.size();
               //for (int k = 0; k < K; ++k) _residuals_vec[pos++] = make_pair(errors[k], make_pair(obj, k));
               for (int k = 0; k < K; ++k) _residuals_vec[pos++] = errors[k];
            } // end for (obj)
            std::sort(_residuals_vec.begin(), _residuals_vec.end());

            int const N0 = std::min(int(_residuals_vec.size())-1, int(ceil(_min_inlier_fraction*_residuals_vec.size())));
            //double const R2 = _residuals_vec[N0].first / tau2;

            if (force_residual_scaling)
            {
               double const R2 = _residuals_vec[N0] / Tau2;
               double const scale2 = std::max(1.0, 3.0*R2);
               for (int obj = 0; obj < nObjs; ++obj)
               {
                  NLSQ_Residuals& residuals = *_residuals[obj];
                  _robustCostFunctions[obj]->cache_IRLS_weights(scale2, cached_errors[obj], residuals._weights);
               }
               return scale2;
            }
            else
            {
               double const R2 = _residuals_vec[N0];

               double const cvx_x = _robustCostFunctions[0]->get_convex_range(1.0), cvx_x2 = cvx_x*cvx_x;
               double const slope = _robustCostFunctions[0]->eval_target_weight(1.0, cvx_x2) * cvx_x;
               //cout << "w = " << _robustCostFunctions[0]->eval_target_weight(1.0, cvx_x2) << " slope = " << slope << endl;
               double const alpha = std::max(0.0, sqrt(R2) - cvx_x);

               for (int obj = 0; obj < nObjs; ++obj)
               {
                  auto costFun = _robustCostFunctions[obj];
                  auto const& errors = cached_errors[obj];
                  auto &weights = _residuals[obj]->_weights;
                  int const K = errors.size();

                  for (int k = 0; k < K; ++k)
                  {
                     double const r2 = errors[k], r = sqrt(r2);
                     if (r2 <= cvx_x2)
                     {
                        weights[k] = costFun->eval_target_weight(1.0, r2);
                     }
                     else if (r >= cvx_x + alpha)
                     {
                        double const r1 = r - alpha, w = costFun->eval_target_weight(1.0, r1*r1);
                        weights[k] = w*r1 / r;
                     }
                     else
                     {
                        weights[k] = slope / r;
                     }
                  } // end for (k)
               } // end for (obj)
               return alpha;
            } // end if

            return -1;
         } // end fill_weights()

         void minimize()
         {
            status = LEVENBERG_OPTIMIZER_TIMEOUT;

            if (_totalParamCount == 0)
            {
               // No degrees of freedom, nothing to optimize.
               if (optimizerVerbosenessLevel >= 2) cout << "Min_Inliers_IRLS_Optimizer: exiting since d.o.f is zero." << endl;
               status = LEVENBERG_OPTIMIZER_CONVERGED;
               return;
            }

            int const totalParamDimension = _JtJ.num_cols();

            vector<double> x_saved(totalParamDimension);

            Vector<double> Jt_e(totalParamDimension);
            Vector<double> delta(totalParamDimension);
            Vector<double> deltaPerm(totalParamDimension);

            int const nObjs = _robustCostFunctions.size();

            double damping_value = this->tau;

            vector<Vector<double> > cached_errors(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);

            int LDL_failures = 0;

            for (int obj = 0; obj < nObjs; ++obj) fillVector(1.0, _residuals[obj]->_weights);

            bool success_decrease = false;

            double scale_info = 0;

            for (currentIteration = 0; currentIteration < maxIterations; ++currentIteration)
            {
               if (currentIteration == 0 || !success_decrease)
               {
                  for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
               }

               if (currentIteration == 0 || success_decrease) scale_info = this->fill_weights(cached_errors);

               double const initial_cost = this->eval_current_cost(cached_errors);

               // if (optimizerVerbosenessLevel >= 1)
               //    cout << "Min_Inliers_IRLS_Optimizer: iteration: " << currentIteration << ", initial |residual|^2 = " << initial_cost << " lambda = " << damping_value << endl;

               bool success_LDL = true;

               for (int obj = 0; obj < nObjs; ++obj)
               {
                  NLSQ_Residuals& residuals = *_residuals[obj];
                  int const K = residuals._weights.size();
                  for (int k = 0; k < K; ++k) scaleVectorIP(residuals._weights[k], residuals._residuals[k]);
               }

               this->fillJacobians();

               this->evalJt_e(Jt_e);
               scaleVectorIP(-1.0, Jt_e);
               double const norm_Linf_Jt_e = norm_Linf(Jt_e);

               if (this->allowStoppingCriteria() && this->applyGradientStoppingCriteria(norm_Linf_Jt_e))
               {
                  if (optimizerVerbosenessLevel >= 2) cout << "Min_Inliers_IRLS_Optimizer: converged due to gradient stopping," << "norm_Linf_Jt_e = " << norm_Linf_Jt_e << endl;
                  break;
               }

               this->fillHessian();
               success_LDL = 0;

               if (!success_LDL)
               {
                  // Augment the diagonals
                  for (int paramType = 0; paramType < _paramDesc.nParamTypes; ++paramType)
                  {
                     MatrixArray<double>& Hs = *_hessian.Hs[paramType][paramType];
                     vector<pair<int, int> > const& nzPairs = _hessian.nonzeroPairs[paramType][paramType];
                     int const dim = Hs.num_cols(), count = Hs.count();

                     // Only augment those with the same parameter id
                     for (int n = 0; n < count; ++n)
                     {
                        if (nzPairs[n].first != nzPairs[n].second) continue;
                        for (int l = 0; l < dim; ++l) Hs[n][l][l] += damping_value;
                     }
                  } // end for (paramType)
                  this->fillJtJ();
                  success_LDL = this->solve_JtJ(Jt_e, deltaPerm, delta);
                  if (!success_LDL) ++LDL_failures;
               } // end if

               success_decrease = false;
               this->copyToAllParameters(&x_saved[0]);

               if (success_LDL)
               {
                  double const deltaSqrLength = sqrNorm_L2(delta);
                  double const paramLength = this->getParameterLength();

                  if (optimizerVerbosenessLevel >= 3)
                     cout << "Min_Inliers_IRLS_Optimizer: ||delta|| = " << sqrt(deltaSqrLength) << " ||paramLength|| = " << paramLength << endl;

                  if (this->allowStoppingCriteria() && this->applyUpdateStoppingCriteria(paramLength, sqrt(deltaSqrLength)))
                  {
                     if (optimizerVerbosenessLevel >= 2) cout << "Min_Inliers_IRLS_Optimizer: converged due to small update, deltaSqrLength = " << deltaSqrLength << endl;
                     break;
                  }

                  for (int paramType = 0; paramType < _paramDesc.nParamTypes; ++paramType)
                  {
                     int const paramDim = _paramDesc.dimension[paramType];
                     int const count    = _paramDesc.count[paramType];
                     int const rowStart = _paramTypeRowStart[paramType];

                     VectorArrayAdapter<double> deltaParam(count, paramDim, &delta[0] + rowStart);
                     this->updateParameters(paramType, deltaParam);
                  } // end for (paramType)
                  this->finishUpdateParameters();

                  for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
               } // end if (success_LDL)

               if (success_LDL)
               {
                  // Check if new cost is better than best one at current level
                  double const current_cost = this->eval_current_cost(cached_errors);

                  if (optimizerVerbosenessLevel >= 1) cout << "IRLS_Optimizer: iteration: " << setw(3) << currentIteration << " success_LDL = " << int(success_LDL)
                                                           << " previous cost = " << setw(12) << initial_cost << " new cost = " << setw(12) << current_cost << " scale_info = " << scale_info << endl;
                  //if (optimizerVerbosenessLevel >= 2) cout << "IRLS_Optimizer: success_LDL = " << int(success_LDL) << " new cost = " << current_cost << endl;

                  success_decrease = true; //(current_cost < initial_cost);
               } // end if (success_LDL)

               if (success_decrease)
               {
                  damping_value = std::max(damping_value_min, damping_value / 10);
               }
               else
               {
                  if (success_LDL) this->copyFromAllParameters(&x_saved[0]);
                  damping_value *= 10;
               }
            } // end for (currentIteration)

            if (optimizerVerbosenessLevel >= 1 && currentIteration+1 >= maxIterations)
            {
               cout << "Min_Inliers_IRLS_Optimizer: reached maximum number of iterations, exiting." << endl;
            }

            if (optimizerVerbosenessLevel >= 2)
               cout << "Leaving Min_Inliers_IRLS_Optimizer::minimize(): LDL_failures = " << LDL_failures << endl;
         } // end minimize()
   }; // end struct Min_Inliers_IRLS_Optimizer

} // end namespace Robust_LSQ

#endif
