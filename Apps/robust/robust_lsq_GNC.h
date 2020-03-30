// -*- C++ -*-
#ifndef ROBUST_LSQ_GNC_H
#define ROBUST_LSQ_GNC_H

#include "Base/v3d_timer.h"
#include "robust_lsq_bt.h"
#include "Math/v3d_linear_lu.h"


namespace Robust_LSQ
{

   template <int NLevels>
   struct HOM_Optimizer_Base : public Robust_LSQ_Customized_Weights_Optimizer_Base
   {
         typedef Robust_LSQ_Customized_Weights_Optimizer_Base Base;

         enum {
            SCALED_KERNELS = 0,
            LINEAR_SEGMENTS = 1,
            CAUCHY_KERNEL = 2,
            GEMAN_KERNEL = 3,
            HUBER_KERNEL = 4
         };

         static int  constexpr GNC_mode = 0;
         static bool constexpr force_level_stopping = 1;

         static double constexpr alpha_multiplier = 2;
         static double constexpr eta_multiplier = 2;

         HOM_Optimizer_Base(NLSQ_ParamDesc const& paramDesc,
                            std::vector<NLSQ_CostFunction *> const& costFunctions,
                            std::vector<Robust_NLSQ_CostFunction_Base *> const& robustCostFunctions)
            : Base(paramDesc, costFunctions, robustCostFunctions),
              _cvx_xs(robustCostFunctions.size()), _alphas(robustCostFunctions.size())
         {
            this->fill_alphas_etas();
         }

         double _etas[NLevels], _etas2[NLevels];

         vector<double> _cvx_xs;
         vector<InlineVector<double, NLevels> > _alphas;

         virtual void fill_alphas_etas()
         {
            for (int obj = 0; obj < _robustCostFunctions.size(); ++obj)
            {
               double const cvx_x = _robustCostFunctions[obj]->get_convex_range(1.0);
               _cvx_xs[obj] = cvx_x;

               auto &alphas = _alphas[obj];

               alphas[0] = 1.0;
               for (int k = 1; k < NLevels; ++k) alphas[k] = alpha_multiplier * alphas[k-1];
               for (int k = 0; k < NLevels; ++k) alphas[k] -= 1.0;
               for (int k = 0; k < NLevels; ++k) alphas[k] *= cvx_x;
            } // end for (obj)

            _etas[0] = 1.0;
            for (int k = 1; k < NLevels; ++k) _etas[k]  = eta_multiplier * _etas[k-1];
            for (int k = 0; k < NLevels; ++k) _etas2[k] = sqr(_etas[k]);
         }

         double eval_current_cost(vector<Vector<double> > const& cached_errors) const
         {
            int const nObjs = _robustCostFunctions.size();
            double cost = 0;
            for (int obj = 0; obj < nObjs; ++obj) cost += _robustCostFunctions[obj]->eval_target_cost(1.0, cached_errors[obj]);
            return cost;
         }

         void cache_target_costs(int const obj, int const level, Vector<double> const& errors, Vector<double> &costs) const
         {
            switch (GNC_mode)
            {
               case SCALED_KERNELS:
                  _robustCostFunctions[obj]->cache_target_costs(_etas2[level], errors, costs);
                  break;
               case LINEAR_SEGMENTS:
               {
                  if (level == 0)
                     _robustCostFunctions[obj]->cache_target_costs(1.0, errors, costs);
                  else
                  {
                     auto costFun = _robustCostFunctions[obj];
                     double const alpha = _alphas[obj][level], alpha1 = 1.0 + alpha, alpha2 = sqr(alpha1);
                     double const cvx_x = _cvx_xs[obj], cvx_x2 = cvx_x*cvx_x;
                     double const psi_cvx_x = costFun->eval_target_fun(1.0, cvx_x2);
                     double const W = costFun->get_cost_weighting();
                     double const slope = W * costFun->eval_target_weight(1.0, cvx_x2) * cvx_x;

                     int const K = errors.size();
                     for (int k = 0; k < K; ++k)
                     {
                        double const r2 = errors[k], r = sqrt(r2);
                        if (r2 <= cvx_x2)
                        {
                           costs[k] = costFun->eval_target_fun(1.0, r2);
                        }
                        else if (r >= cvx_x + alpha)
                        {
                           costs[k] = costFun->eval_target_fun(1.0, sqr(r-alpha)) + alpha*slope;
                           //costs[k] = costFun->eval_target_fun(alpha2, r2) + psi_cvx_x*(1.0 - alpha2) + alpha*slope;
                        }
                        else
                        {
                           costs[k] = psi_cvx_x + slope*(r - cvx_x);
                        }
                     } // end for (k)
                  } // end if
                  break;
               }
               case CAUCHY_KERNEL:
               {
                  if (level == 0)
                     _robustCostFunctions[obj]->cache_target_costs(1.0, errors, costs);
                  else
                  {
                     auto costFun = _robustCostFunctions[obj];
                     double const W = costFun->get_cost_weighting(), tau2 = costFun->get_tau2(), W_tau2 = W*tau2;
                     int const K = errors.size();
                     for (int k = 0; k < K; ++k) costs[k] = W_tau2 * Psi_Cauchy::fun(errors[k] / tau2);
                  }
                  break;
               }
               case GEMAN_KERNEL:
               {
                  if (level == 0)
                     _robustCostFunctions[obj]->cache_target_costs(1.0, errors, costs);
                  else
                  {
                     auto costFun = _robustCostFunctions[obj];
                     double const W = costFun->get_cost_weighting(), tau2 = costFun->get_tau2(), W_tau2 = W*tau2;;
                     int const K = errors.size();
                     for (int k = 0; k < K; ++k) costs[k] = W_tau2 * Psi_Geman::fun(errors[k] / tau2);
                  }
                  break;
               }
               case HUBER_KERNEL:
               {
                  if (level == 0)
                     _robustCostFunctions[obj]->cache_target_costs(1.0, errors, costs);
                  else
                  {
                     auto costFun = _robustCostFunctions[obj];
                     double const W = costFun->get_cost_weighting(), tau2 = costFun->get_tau2(), W_tau2 = W*tau2;
                     int const K = errors.size();
                     for (int k = 0; k < K; ++k) costs[k] = W_tau2 * Psi_Huber::fun(errors[k] / tau2);
                  }
                  break;
               }
               default:
                  throw std::string("cache_target_costs(): unknown GNC_mode");
            } // end switch (GNC_mode)
         } // end cache_target_costs()

         double eval_level_cost(int const level, vector<Vector<double> > const& cached_errors) const
         {
            int const nObjs = _robustCostFunctions.size();
            double cost = 0;

            for (int obj = 0; obj < nObjs; ++obj)
            {
               auto costFun = _robustCostFunctions[obj];
               auto const& errors = cached_errors[obj];

               int const K = errors.size();
               Vector<double> costs(K);
               this->cache_target_costs(obj, level, errors, costs);
               double cost1 = 0; 
               for (int k = 0; k < K; ++k) 
                  cost1 += costs[k];
               cost += cost1;
            } // end for (obj)
            return cost;
         }

         void compute_weights(int obj, int level, Vector<double> const& errors, Vector<double>& weights) const
         {
            auto costFun = _robustCostFunctions[obj];
            int const K = errors.size();

            switch (GNC_mode)
            {
               case SCALED_KERNELS:
               {
                  double const s2 = _etas2[level];
                  for (int k = 0; k < K; ++k)
                  {
                     double const r2 = errors[k];
                     weights[k] = costFun->eval_target_weight(s2, r2);
                  } // end for (k)
                  break;
               }
               case LINEAR_SEGMENTS:
               {
                  if (level == 0)
                     _robustCostFunctions[obj]->cache_weight_fun(1.0, errors, weights);
                  else
                  {
                     double const cvx_x = costFun->get_convex_range(1.0), cvx_x2 = cvx_x*cvx_x;
                     double const alpha = _alphas[obj][level], alpha1 = 1.0 + alpha, alpha2 = sqr(alpha1);
                     double const W = costFun->get_cost_weighting();
                     double const slope = W * costFun->eval_target_weight(1.0, cvx_x2) * cvx_x;

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
                           //weights[k] = costFun->eval_target_weight(alpha2, r2);
                        }
                     } // end for (k)
                  } // end if
                  break;
               }
               case CAUCHY_KERNEL:
               {
                  if (level == 0)
                     _robustCostFunctions[obj]->cache_weight_fun(1.0, errors, weights);
                  else
                  {
                     auto costFun = _robustCostFunctions[obj];
                     double const tau2 = costFun->get_tau2();
                     for (int k = 0; k < K; ++k) weights[k] = Psi_Cauchy::weight_fun(errors[k] / tau2);
                  }
                  break;
               }
               case GEMAN_KERNEL:
               {
                  if (level == 0)
                     _robustCostFunctions[obj]->cache_weight_fun(1.0, errors, weights);
                  else
                  {
                     auto costFun = _robustCostFunctions[obj];
                     double const tau2 = costFun->get_tau2();
                     for (int k = 0; k < K; ++k) weights[k] = Psi_Geman::weight_fun(errors[k] / tau2);
                  }
                  break;
               }
               case HUBER_KERNEL:
               {
                  if (level == 0)
                     _robustCostFunctions[obj]->cache_weight_fun(1.0, errors, weights);
                  else
                  {
                     auto costFun = _robustCostFunctions[obj];
                     double const tau2 = costFun->get_tau2();
                     for (int k = 0; k < K; ++k) weights[k] = Psi_Huber::weight_fun(errors[k] / tau2);
                  }
                  break;
               }
            } // end switch

            scaleVectorIP(costFun->get_cost_weighting(), weights);
         } // end compute_weights()

         double eval_robust_objective() const
         {
            int const nObjs = _robustCostFunctions.size();
            vector<Vector<double> > cached_errors(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
            for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
            return this->eval_level_cost(0, cached_errors);
         }

         virtual bool allowStoppingCriteria() const { return true; }
   }; // end struct HOM_Optimizer_Base

//**********************************************************************

   // template <int NLevels, bool allow_early_level_stopping = true>
   // struct Fast_HOM_Optimizer2 : public HOM_Optimizer_Base<NLevels>
   // {
   //       typedef HOM_Optimizer_Base<NLevels> Base;

   //       using Base::status;
   //       using Base::_JtJ;
   //       using Base::_hessian;
   //       using Base::_paramDesc;
   //       using Base::_totalParamCount;
   //       using Base::_paramTypeRowStart;
   //       using Base::_residuals;
   //       using Base::currentIteration;
   //       using Base::maxIterations;
   //       using Base::_costFunctions;
   //       using Base::_robustCostFunctions;
   //       using Base::force_level_stopping;

   //       Fast_HOM_Optimizer2(NLSQ_ParamDesc const& paramDesc,
   //                          std::vector<NLSQ_CostFunction *> const& costFunctions,
   //                          std::vector<Robust_NLSQ_CostFunction_Base *> const& robustCostFunctions)
   //          : Base(paramDesc, costFunctions, robustCostFunctions)
   //       { }

   //       Vector2d eval_level_deltas(int const level, vector<Vector<double> > const& cached_errors, vector<Vector<double> > const& cached_errors0) const
   //       {
   //          int const nObjs = _robustCostFunctions.size();

   //          Vector2d res(0.0, 0.0);

   //          for (int obj = 0; obj < nObjs; ++obj)
   //          {
   //             auto const& errors  = cached_errors[obj];
   //             auto const& errors0 = cached_errors0[obj];

   //             int const K = errors.size();

   //             Vector<double> costs_old(K), costs_new(K);
   //             this->cache_target_costs(obj, level, errors0, costs_old);
   //             this->cache_target_costs(obj, level, errors,  costs_new);

   //             for (int k = 0; k < K; ++k)
   //             {
   //                res[0] += std::max(0.0, costs_old[k] - costs_new[k]);
   //                res[1] += std::max(0.0, costs_new[k] - costs_old[k]);
   //             } // end for (k)
   //          } // end for (obj)

   //          return res;
   //       } // end eval_level_deltas()

   //       // void report_lower_bound_psi(double const radius2) const
   //       // {
   //       //    double const radius = sqrt(radius2);

   //       //    double lb = 0;
   //       //    int n_invalid = 0;

   //       //    int const nObjs = _robustCostFunctions.size();
   //       //    for (int obj = 0; obj < nObjs; ++obj)
   //       //    {
   //       //       NLSQ_CostFunction& costFun = *_costFunctions[obj];
   //       //       NLSQ_Residuals& residuals = *_residuals[obj];

   //       //       int const nParamTypes   = costFun._usedParamTypes.size();
   //       //       int const nMeasurements = costFun._nMeasurements;
   //       //       int const dim           = costFun._measurementDimension;

   //       //       MatrixArray<double> JJt(nMeasurements, dim, dim);
   //       //       for (int k = 0; k < nMeasurements; ++k) makeZeroMatrix(JJt[k]);

   //       //       Matrix<double> JJt_tmp(dim, dim);

   //       //       for (int i = 0; i < nParamTypes; ++i)
   //       //       {
   //       //          MatrixArray<double> const& J = *residuals._Js[i];

   //       //          for (int k = 0; k < nMeasurements; ++k)
   //       //          {
   //       //             multiply_A_At(J[k], JJt_tmp);
   //       //             addMatricesIP(JJt_tmp, JJt[k]);
   //       //          } // end for (k)
   //       //       } // end for (i)

   //       //       VectorArray<double> invJJt_e(nMeasurements, dim);
   //       //       vector<bool> valid(nMeasurements, true);

   //       //       for (int k = 0; k < nMeasurements; ++k)
   //       //       {
   //       //          //for (int d = 0; d < dim; ++d) JJt[k][d][d] += 1e-6*matrixNormFrobenius(JJt[k]);

   //       //          LU<double> lu(JJt[k]);
   //       //          if (lu.isNonsingular())
   //       //             copyVector(lu.solve(residuals._residuals[k]), invJJt_e[k]);
   //       //          else
   //       //          {
   //       //             makeZeroVector(invJJt_e[k]);
   //       //             valid[k] = false;
   //       //             ++n_invalid;
   //       //          }
   //       //       } // end for (k)

   //       //       vector<double> denoms(nMeasurements, 0.0);
   //       //       for (int i = 0; i < nParamTypes; ++i)
   //       //       {
   //       //          int const paramType = costFun._usedParamTypes[i];
   //       //          int const paramDim  = _paramDesc.dimension[paramType];
   //       //          Vector<double> Jx(paramDim);
   //       //          MatrixArray<double> const& J = *residuals._Js[i];

   //       //          for (int k = 0; k < nMeasurements; ++k)
   //       //          {
   //       //             multiply_At_v(J[k], invJJt_e[k], Jx); denoms[k] += sqrNorm_L2(Jx);
   //       //          } // end for (k)
   //       //       } // end for (i)

   //       //       // for (int k = 0; k < 12; ++k) cout << denoms[k] << " "; cout << endl;
   //       //       for (int k = 0; k < nMeasurements; ++k)
   //       //       {
   //       //          if (valid[k])
   //       //             denoms[k] = (denoms[k] <= radius2) ? 1.0 : (radius / sqrt(denoms[k]));
   //       //          else
   //       //             denoms[k] = 0.0;
   //       //       }
   //       //       // for (int k = 0; k < 12; ++k) cout << denoms[k] << " "; cout << endl;
   //       //       // for (int k = 0; k < 12; ++k) cout << matrixNormFrobenius(JJt[k]) << " "; cout << endl;

   //       //       Vector<double> errors(nMeasurements), costs(nMeasurements);

   //       //       for (int k = 0; k < nMeasurements; ++k) errors[k] = sqr(1.0 - denoms[k]) * sqrNorm_L2(residuals._residuals[k]);
   //       //       _robustCostFunctions[obj]->cache_target_costs(1.0, errors, costs);
   //       //       double lb1 = 0; for (int k = 0; k < nMeasurements; ++k) lb1 += costs[k];
   //       //       lb += lb1;
   //       //    } // end for (obj)
   //       //    cout << "radius = " << radius << " lb = " << lb << " n_invalid = " << n_invalid << endl;
   //       // } // end report_lower_bound_psi()

   //       void minimize()
   //       {
   //          double const rho_stopping = 0.95;

   //          status = LEVENBERG_OPTIMIZER_TIMEOUT;

   //          if (_totalParamCount == 0)
   //          {
   //             // No degrees of freedom, nothing to optimize.
   //             if (optimizerVerbosenessLevel >= 2) cout << "BT_Parallel_IRLS_Optimizer: exiting since d.o.f is zero." << endl;
   //             status = LEVENBERG_OPTIMIZER_CONVERGED;
   //             return;
   //          }

   //          int const totalParamDimension = _JtJ.num_cols();

   //          vector<double> x0(totalParamDimension), x_saved(totalParamDimension);

   //          Vector<double> Jt_e(totalParamDimension);
   //          Vector<double> delta(totalParamDimension);
   //          Vector<double> deltaPerm(totalParamDimension);

   //          int const nObjs = _robustCostFunctions.size();

   //          double damping_value = this->tau;

   //          vector<Vector<double> > cached_errors(nObjs), cached_errors0(nObjs);
   //          for (int obj = 0; obj < nObjs; ++obj) cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
   //          for (int obj = 0; obj < nObjs; ++obj) cached_errors0[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);

   //          vector<Vector<double> > cached_weights(nObjs);
   //          for (int obj = 0; obj < nObjs; ++obj) cached_weights[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);

   //          if (use_residual_scaling)
   //          {
   //             _robustCostFunctions[obj]->cache_target_costs(_etas2[level], errors, costs);
   //          }
   //          else
   //          {
   //             if (level == 0)
   //                _robustCostFunctions[obj]->cache_target_costs(1.0, errors, costs);
   //             else
   //             {
   //                auto costFun = _robustCostFunctions[obj];
   //                double const alpha = _alphas[obj][level];
   //                double const cvx_x = _cvx_xs[obj], cvx_x2 = cvx_x*cvx_x;
   //                double const psi_cvx_x = costFun->eval_target_fun(1.0, cvx_x2);
   //                double const slope = costFun->eval_target_weight(1.0, cvx_x2) * cvx_x;

   //                int const K = errors.size();
   //                for (int k = 0; k < K; ++k)
   //                {
   //                   double const r2 = errors[k], r = sqrt(r2);
   //                   if (r2 <= cvx_x2)
   //                   {
   //                      costs[k] = costFun->eval_target_fun(1.0, r2);
   //                   }
   //                   else if (r >= cvx_x + alpha)
   //                   {
   //                      costs[k] = costFun->eval_target_fun(1.0, sqr(r-alpha)) + alpha*slope;
   //                   }
   //                   else
   //                   {
   //                      costs[k] = psi_cvx_x + slope*(r - cvx_x);
   //                   }
   //                } // end for (k)
   //             } // end if
   //          } // end if
   //       } // end cache_target_costs()

   //       double eval_level_cost(int const level, vector<Vector<double> > const& cached_errors) const
   //       {
   //          int const nObjs = _robustCostFunctions.size();
   //          double cost = 0;

   //          for (int obj = 0; obj < nObjs; ++obj)
   //          {
   //             auto costFun = _robustCostFunctions[obj];
   //             auto const& errors = cached_errors[obj];

   //             int const K = errors.size();
   //             Vector<double> costs(K);
   //             this->cache_target_costs(obj, level, errors, costs);
   //             double cost1 = 0; for (int k = 0; k < K; ++k) cost1 += costs[k];
   //             cost += cost1;
   //          } // end for (obj)
   //          return cost;
   //       }

   //       void compute_weights(int obj, int level, Vector<double> const& errors, Vector<double>& weights) const
   //       {
   //          auto costFun = _robustCostFunctions[obj];

   //          if (1 && level == 0)
   //          {
   //             costFun->cache_weight_fun(1.0, errors, weights);
   //          }
   //          else
   //          {
   //             int const K = errors.size();

   //             if (use_residual_scaling)
   //             {
   //                double const s2 = _etas2[level];
   //                for (int k = 0; k < K; ++k)
   //                {
   //                   double const r2 = errors[k];
   //                   weights[k] = costFun->eval_target_weight(s2, r2);
   //                } // end for (k)
   //             }
   //             else
   //             {
   //                double const cvx_x = costFun->get_convex_range(1.0), cvx_x2 = cvx_x*cvx_x;
   //                double const slope = costFun->eval_target_weight(1.0, cvx_x2) * cvx_x;
   //                double const alpha = _alphas[obj][level];

   //                for (int k = 0; k < K; ++k)
   //                {
   //                   double const r2 = errors[k], r = sqrt(r2);
   //                   if (r2 <= cvx_x2)
   //                   {
   //                      weights[k] = costFun->eval_target_weight(1.0, r2);
   //                   }
   //                   else if (r >= cvx_x + alpha)
   //                   {
   //                      double const r1 = r - alpha, w = costFun->eval_target_weight(1.0, r1*r1);
   //                      weights[k] = w*r1 / r;
   //                   }
   //                   else
   //                   {
   //                      weights[k] = slope / r;
   //                   }
   //                } // end for (k)
   //             } // end if
   //          } /// end if

   //          scaleVectorIP(costFun->get_cost_weighting(), weights);
   //       } // end compute_weights()

   //       double eval_robust_objective() const
   //       {
   //          int const nObjs = _robustCostFunctions.size();
   //          vector<Vector<double> > cached_errors(nObjs);
   //          for (int obj = 0; obj < nObjs; ++obj) cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
   //          for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
   //          return this->eval_level_cost(0, cached_errors);
   //       }

   //       virtual bool allowStoppingCriteria() const { return true; }
   // }; // end struct HOM_Optimizer_Base

//**********************************************************************

   template <int NLevels, bool allow_early_level_stopping = true>
   struct Fast_HOM_Optimizer : public HOM_Optimizer_Base<NLevels>
   {
         typedef HOM_Optimizer_Base<NLevels> Base;

         using Base::status;
         using Base::_JtJ;
         using Base::_hessian;
         using Base::_paramDesc;
         using Base::_totalParamCount;
         using Base::_paramTypeRowStart;
         using Base::_residuals;
         using Base::currentIteration;
         using Base::maxIterations;
         using Base::_costFunctions;
         using Base::_robustCostFunctions;
         using Base::force_level_stopping;

         Fast_HOM_Optimizer(NLSQ_ParamDesc const& paramDesc,
                            std::vector<NLSQ_CostFunction *> const& costFunctions,
                            std::vector<Robust_NLSQ_CostFunction_Base *> const& robustCostFunctions)
            : Base(paramDesc, costFunctions, robustCostFunctions)
         { }

         Vector2d eval_level_deltas(int const level, vector<Vector<double> > const& cached_errors, vector<Vector<double> > const& cached_errors0) const
         {
            int const nObjs = _robustCostFunctions.size();

            Vector2d res(0.0, 0.0);

            for (int obj = 0; obj < nObjs; ++obj)
            {
               auto const& errors  = cached_errors[obj];
               auto const& errors0 = cached_errors0[obj];

               int const K = errors.size();

               Vector<double> costs_old(K), costs_new(K);
               this->cache_target_costs(obj, level, errors0, costs_old);
               this->cache_target_costs(obj, level, errors,  costs_new);

               for (int k = 0; k < K; ++k)
               {
                  res[0] += std::max(0.0, costs_old[k] - costs_new[k]);
                  res[1] += std::max(0.0, costs_new[k] - costs_old[k]);
               } // end for (k)
            } // end for (obj)

            return res;
         } // end eval_level_deltas()

         // void report_lower_bound_psi(double const radius2) const
         // {
         //    double const radius = sqrt(radius2);

         //    double lb = 0;
         //    int n_invalid = 0;

         //    int const nObjs = _robustCostFunctions.size();
         //    for (int obj = 0; obj < nObjs; ++obj)
         //    {
         //       NLSQ_CostFunction& costFun = *_costFunctions[obj];
         //       NLSQ_Residuals& residuals = *_residuals[obj];

         //       int const nParamTypes   = costFun._usedParamTypes.size();
         //       int const nMeasurements = costFun._nMeasurements;
         //       int const dim           = costFun._measurementDimension;

         //       MatrixArray<double> JJt(nMeasurements, dim, dim);
         //       for (int k = 0; k < nMeasurements; ++k) makeZeroMatrix(JJt[k]);

         //       Matrix<double> JJt_tmp(dim, dim);

         //       for (int i = 0; i < nParamTypes; ++i)
         //       {
         //          MatrixArray<double> const& J = *residuals._Js[i];

         //          for (int k = 0; k < nMeasurements; ++k)
         //          {
         //             multiply_A_At(J[k], JJt_tmp);
         //             addMatricesIP(JJt_tmp, JJt[k]);
         //          } // end for (k)
         //       } // end for (i)

         //       VectorArray<double> invJJt_e(nMeasurements, dim);
         //       vector<bool> valid(nMeasurements, true);

         //       for (int k = 0; k < nMeasurements; ++k)
         //       {
         //          //for (int d = 0; d < dim; ++d) JJt[k][d][d] += 1e-6*matrixNormFrobenius(JJt[k]);

         //          LU<double> lu(JJt[k]);
         //          if (lu.isNonsingular())
         //             copyVector(lu.solve(residuals._residuals[k]), invJJt_e[k]);
         //          else
         //          {
         //             makeZeroVector(invJJt_e[k]);
         //             valid[k] = false;
         //             ++n_invalid;
         //          }
         //       } // end for (k)

         //       vector<double> denoms(nMeasurements, 0.0);
         //       for (int i = 0; i < nParamTypes; ++i)
         //       {
         //          int const paramType = costFun._usedParamTypes[i];
         //          int const paramDim  = _paramDesc.dimension[paramType];
         //          Vector<double> Jx(paramDim);
         //          MatrixArray<double> const& J = *residuals._Js[i];

         //          for (int k = 0; k < nMeasurements; ++k)
         //          {
         //             multiply_At_v(J[k], invJJt_e[k], Jx); denoms[k] += sqrNorm_L2(Jx);
         //          } // end for (k)
         //       } // end for (i)

         //       // for (int k = 0; k < 12; ++k) cout << denoms[k] << " "; cout << endl;
         //       for (int k = 0; k < nMeasurements; ++k)
         //       {
         //          if (valid[k])
         //             denoms[k] = (denoms[k] <= radius2) ? 1.0 : (radius / sqrt(denoms[k]));
         //          else
         //             denoms[k] = 0.0;
         //       }
         //       // for (int k = 0; k < 12; ++k) cout << denoms[k] << " "; cout << endl;
         //       // for (int k = 0; k < 12; ++k) cout << matrixNormFrobenius(JJt[k]) << " "; cout << endl;

         //       Vector<double> errors(nMeasurements), costs(nMeasurements);

         //       for (int k = 0; k < nMeasurements; ++k) errors[k] = sqr(1.0 - denoms[k]) * sqrNorm_L2(residuals._residuals[k]);
         //       _robustCostFunctions[obj]->cache_target_costs(1.0, errors, costs);
         //       double lb1 = 0; for (int k = 0; k < nMeasurements; ++k) lb1 += costs[k];
         //       lb += lb1;
         //    } // end for (obj)
         //    cout << "radius = " << radius << " lb = " << lb << " n_invalid = " << n_invalid << endl;
         // } // end report_lower_bound_psi()

         void minimize()
         {
            double const rho_stopping = 0.95;

            status = LEVENBERG_OPTIMIZER_TIMEOUT;

            if (_totalParamCount == 0)
            {
               // No degrees of freedom, nothing to optimize.
               if (optimizerVerbosenessLevel >= 2) cout << "BT_Parallel_IRLS_Optimizer: exiting since d.o.f is zero." << endl;
               status = LEVENBERG_OPTIMIZER_CONVERGED;
               return;
            }

            int const totalParamDimension = _JtJ.num_cols();

            vector<double> x0(totalParamDimension), x_saved(totalParamDimension);

            Vector<double> Jt_e(totalParamDimension);
            Vector<double> delta(totalParamDimension);
            Vector<double> deltaPerm(totalParamDimension);

            int const nObjs = _robustCostFunctions.size();

            double damping_value = this->tau;

            vector<Vector<double> > cached_errors(nObjs), cached_errors0(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
            for (int obj = 0; obj < nObjs; ++obj) cached_errors0[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);

            vector<Vector<double> > cached_weights(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) cached_weights[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);

            for (int obj = 0; obj < nObjs; ++obj) fillVector(1.0, _residuals[obj]->_weights);

            for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
            double best_cost = this->eval_current_cost(cached_errors);
            
            Timer t("Bundle");
            ofstream log_file("log_GNC.txt");            
            cout << "Initialization cost  = " << best_cost << endl;
            
            int LDL_failures = 0;

            this->copyToAllParameters(&x0[0]);

            currentIteration = 0;
            t.start();
            for (int level = NLevels-1; level >= 0; --level)
            {
               bool success_LDL = true;

               int const remainingIterations = maxIterations - currentIteration;
               int const n_inner_iterations = force_level_stopping ? int(0.5+double(remainingIterations)/(level+1)) : remainingIterations;

               for (int iter = 0; iter < n_inner_iterations; ++iter)
               {
                  for (int obj = 0; obj < nObjs; ++obj) 
                     _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
                     
                  double const Psi0_k  = this->eval_level_cost(level, cached_errors);
                  double const Psi0_kk = (level > 0) ? this->eval_level_cost(level-1, cached_errors) : 0.0;

                  for (int obj = 0; obj < nObjs; ++obj) cached_errors0[obj] = cached_errors[obj];

                  //Compute cost to log
                  double const init_cost = this->eval_level_cost(0, cached_errors);
                  best_cost = std::min(best_cost, init_cost);
                  t.stop();
                  log_file << t.getTime() << "\t" << currentIteration << "\t" << best_cost << endl;
                  t.start();
                  this->fillJacobians();
                   

                  for (int obj = 0; obj < nObjs; ++obj)
                  {
                     NLSQ_Residuals& residuals = *_residuals[obj];
                     this->compute_weights(obj, level, cached_errors[obj], cached_weights[obj]);
                  }

                  this->eval_Jt_e(cached_weights, Jt_e); scaleVectorIP(-1.0, Jt_e);

                  this->fillHessian(cached_weights);
                  this->fillJtJ();

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
                  ++currentIteration;
                  if (!success_LDL) ++LDL_failures;

                  bool stopping_reached = false;

                  bool success_decrease = false;
                  this->copyToAllParameters(&x_saved[0]);

                  if (success_LDL)
                  {
                     double const deltaSqrLength = sqrNorm_L2(delta);
                     double const paramLength = this->getParameterLength();

                     if (optimizerVerbosenessLevel >= 3)
                        cout << "Fast_HOM_Optimizer: ||delta|| = " << sqrt(deltaSqrLength) << " ||paramLength|| = " << paramLength << endl;

                     for (int paramType = 0; paramType < _paramDesc.nParamTypes; ++paramType)
                     {
                        int const paramDim = _paramDesc.dimension[paramType];
                        int const count    = _paramDesc.count[paramType];
                        int const rowStart = _paramTypeRowStart[paramType];

                        VectorArrayAdapter<double> deltaParam(count, paramDim, &delta[0] + rowStart);
                        this->updateParameters(paramType, deltaParam);
                     } // end for (paramType)
                     this->finishUpdateParameters();
                  } // end if (success_LDL)
                  
                  if (success_LDL)
                  {
                     for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
                     double const Psi1_k  = this->eval_level_cost(level, cached_errors);

                     success_decrease = (Psi1_k < Psi0_k);

                     // Check if we can jump to lower level early
                     if (allow_early_level_stopping && level > 0)
                     {
                        double const Psi1_kk = (level > 0) ? this->eval_level_cost(level-1, cached_errors) : 0.0;
#if 0
                        double const rho = (Psi0_kk - Psi1_kk) / (Psi0_k - Psi1_k);
                        if (rho > rho_stopping) stopping_reached = true;
#elif 0
                        double const eta = 0.5, th_lo = 0.5*(eta-1.0)/eta, th_hi = 0.5*(eta+1.0)/eta;
                        double const rho = (Psi0_kk - Psi1_kk) / (Psi0_k - Psi1_k);
                        //cout << "rho = " << rho << " th_lo = " << th_lo << " th_hi = " << th_hi << endl;
                        if (rho < th_lo || rho > th_hi) stopping_reached = true;
#elif 1
                        double const eta = 0.1; // 0.2
                        Vector2d const deltas = this->eval_level_deltas(level, cached_errors, cached_errors0);
                        double const rho = (deltas[0] - deltas[1]) / (deltas[0] + deltas[1]);
                        if (rho < eta) stopping_reached = true;
#else
                        double const rho = fabs(Psi0_kk - Psi1_kk) - (Psi0_k - Psi1_k);
                        if (rho > 0.0) stopping_reached = true;
#endif
                     } // end if

                     //    double const radius2 = sqrNorm_L2(delta);
                     //    this->report_lower_bound_psi(radius2);

                     if (optimizerVerbosenessLevel >= 1)
                     {
                        double const current_cost = this->eval_level_cost(0, cached_errors);
                        best_cost = std::min(best_cost, current_cost);
                        // if (optimizerVerbosenessLevel == 1 && (1 || success_decrease))
                        //    cout << "Fast_HOM_Optimizer: iteration " << currentIteration << " level = " << level << " new cost = " << current_cost << " best cost = " << best_cost << " lambda = " << damping_value << endl;
                        if (optimizerVerbosenessLevel >= 1 && (0 || success_decrease))
                        {
                           cout << "Fast_HOM_Optimizer: iteration " << setw(3) << currentIteration << " level = " << level << " new cost = " << setw(12) << current_cost
                                << " best cost = " << setw(12) << best_cost << " Psi_k(prev) = " << setw(12) << Psi0_k << " Psi_k(new) = " << setw(12) << Psi1_k
                                << " success_decreas = " << int(success_decrease) << " lambda = " << damping_value << endl;

                         
                        }
                     }

                  } // end (if success_LDL)

                  if (success_decrease)
                  {
                     damping_value = std::max(damping_value_min, damping_value / 10);
                     this->copyToAllParameters(&x0[0]);
                  }
                  else
                  {
                     if (success_LDL) this->copyFromAllParameters(&x_saved[0]);
                     damping_value *= 10;
                  }

                  if (stopping_reached) break;
               } // end for (iter)
            } // end for (level)

            this->copyFromAllParameters(&x0[0]);
            log_file.close();
         } // end minimize()
   }; // end struct Fast_HOM_Optimizer

//**********************************************************************

   template <int NLevels, bool use_plain_GNC = true>
   struct HOM_Optimizer : public BT_IRLS_Optimizer<NLevels>
   {
         HOM_Optimizer(NLSQ_ParamDesc const& paramDesc,
                       std::vector<NLSQ_CostFunction *> const& costFunctions,
                       std::vector<Robust_NLSQ_CostFunction_Base *> const& robustCostFunctions)
            : BT_IRLS_Optimizer<NLevels>(paramDesc, costFunctions, robustCostFunctions)
         { }

         typedef BT_IRLS_Optimizer<NLevels> Base;

         using Base::status;
         using Base::_JtJ;
         using Base::_hessian;
         using Base::_paramDesc;
         using Base::_totalParamCount;
         using Base::_paramTypeRowStart;
         using Base::_residuals;
         using Base::currentIteration;
         using Base::maxIterations;
         using Base::_costFunctions;
         using Base::_robustCostFunctions;
         using Base::_etas;
         using Base::_etas2;

         double eval_current_cost(vector<Vector<double> > const& cached_errors) const
         {
            int const nObjs = _robustCostFunctions.size();
            double cost = 0;
            for (int obj = 0; obj < nObjs; ++obj) cost += _robustCostFunctions[obj]->eval_target_cost(1.0, cached_errors[obj]);
            return cost;
         }

         int count_inliers()
         {
            NLSQ_CostFunction &costFun = *_costFunctions[0];
            int inls = 0;
            for (int i = 0; i < costFun._nMeasurements; ++i)
            {
                  inls += costFun.isInlier(i);
            }   
            return inls;
         }           
         
         void minimize()
         {
#if 0
            int const n_inner_iterations = use_plain_GNC ? int(0.5+double(maxIterations)/NLevels) : (0.5+2.0*maxIterations/(NLevels*(NLevels+1.0)));

            status = LEVENBERG_OPTIMIZER_TIMEOUT;

            if (_totalParamCount == 0)
            {
               // No degrees of freedom, nothing to optimize.
               if (optimizerVerbosenessLevel >= 2) cout << "BT_Parallel_IRLS_Optimizer: exiting since d.o.f is zero." << endl;
               status = LEVENBERG_OPTIMIZER_CONVERGED;
               return;
            }

            int const totalParamDimension = _JtJ.num_cols();

            vector<double> x0(totalParamDimension), trial_point(totalParamDimension), x_orig(totalParamDimension);

            Vector<double> Jt_e(totalParamDimension);
            Vector<double> delta(totalParamDimension);
            Vector<double> deltaPerm(totalParamDimension);

            int const nObjs = _robustCostFunctions.size();

            double damping_value = this->tau;

            vector<Vector<double> > cached_errors(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);

            for (int obj = 0; obj < nObjs; ++obj) fillVector(1.0, _residuals[obj]->_weights);

            int LDL_failures = 0;

            this->copyToAllParameters(&x0[0]);
            this->copyToAllParameters(&x_orig[0]);

            int const start_level = use_plain_GNC ? NLevels-1 : 0;

            currentIteration = 0;

            for (int level1 = start_level; level1 < NLevels; ++level1)
            {
               //trial_point = x0;
               trial_point = x_orig;
               this->copyFromAllParameters(&trial_point[0]);

               for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
               //double const initial_cost = this->eval_current_cost(cached_errors);
               double const initial_cost = this->eval_level_cost(level1, cached_errors);

               if (optimizerVerbosenessLevel >= 1)
                  cout << "Basin_Search_IRLS_Optimizer: target level: " << level1 << ", initial |residual|^2 = " << initial_cost << " lambda = " << damping_value << " eta[level] = " << _etas[level] << endl;

               for (int level = level1; level >= 0; --level)
               {
                  bool success_LDL = true;

                  double best_cost = initial_cost;

                  for (int iter = 0; iter < n_inner_iterations; ++iter)
                  {
                     //if (optimizerVerbosenessLevel >= 2) cout << "BT_IRLS_Optimizer: inner iteration " << iter << " lambda = " << damping_value << endl;

                     for (int obj = 0; obj < nObjs; ++obj)
                     {
                        NLSQ_Residuals& residuals = *_residuals[obj];
                        _robustCostFunctions[obj]->cache_IRLS_weights(_etas2[level], cached_errors[obj], residuals._weights);

                        int const K = residuals._weights.size();
                        for (int k = 0; k < K; ++k) scaleVectorIP(residuals._weights[k], residuals._residuals[k]);
                     }

                     this->fillJacobians();

                     this->evalJt_e(Jt_e);
                     scaleVectorIP(-1.0, Jt_e);

                     this->fillHessian();
                     this->fillJtJ();

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
                     ++currentIteration;
                     if (!success_LDL) ++LDL_failures;

                     if (success_LDL)
                     {
                        double const deltaSqrLength = sqrNorm_L2(delta);
                        double const paramLength = this->getParameterLength();

                        if (optimizerVerbosenessLevel >= 3)
                           cout << "BT_IRLS_Optimizer: ||delta|| = " << sqrt(deltaSqrLength) << " ||paramLength|| = " << paramLength << endl;

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
                        double const current_cost = this->eval_level_cost(level, cached_errors);

                        if (optimizerVerbosenessLevel >= 1)
                        {
                           cout << "Basin_Search_IRLS_Optimizer: iteration " << currentIteration << " level = " << level << " new cost = " << current_cost << " existing cost (x0) = " << best_cost << endl;
                        }
                        best_cost = current_cost;

                        damping_value = std::max(damping_value_min, damping_value / 10);
                     }
                     else
                     {
                        damping_value *= 10;
                     } // end (if success_LDL)
                  } // end for (iter)
               } // end for (level)

               // Check if new cost is better than best one at current level
               //double const current_cost = this->eval_level_cost(0, cached_errors);
               //if (optimizerVerbosenessLevel >= 1) cout << "Basin_Search_IRLS_Optimizer: new cost = " << current_cost << " existing cost = " << best_cost << endl;

               // if (current_cost < best_cost)
               // {
               //    best_cost = current_cost;
               //    this->copyToAllParameters(&x0[0]);
               // }
            } // end for (level1)
            this->copyToAllParameters(&x0[0]);
#else
            int const n_inner_iterations = int(0.5+double(maxIterations)/NLevels);

            status = LEVENBERG_OPTIMIZER_TIMEOUT;

            if (_totalParamCount == 0)
            {
               // No degrees of freedom, nothing to optimize.
               if (optimizerVerbosenessLevel >= 2) cout << "HOM_Optimizer: exiting since d.o.f is zero." << endl;
               status = LEVENBERG_OPTIMIZER_CONVERGED;
               return;
            }

            int const totalParamDimension = _JtJ.num_cols();

            Vector<double> Jt_e(totalParamDimension);
            Vector<double> delta(totalParamDimension);
            Vector<double> deltaPerm(totalParamDimension);

            vector<double> x_saved(totalParamDimension);

            int const nObjs = _robustCostFunctions.size();

            double damping_value = this->tau;

            vector<Vector<double> > cached_errors(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);

            for (int obj = 0; obj < nObjs; ++obj) fillVector(1.0, _residuals[obj]->_weights);

            int LDL_failures = 0;

            currentIteration = 0;

            ofstream log_file("log_HOM.txt");
            Timer t("HOM");
            t.start();
            for (int level = NLevels-1; level >= 0; --level)
            {
               bool success_LDL = true;

               for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
               double best_cost = this->eval_level_cost(level, cached_errors);;

               for (int iter = 0; iter < n_inner_iterations; ++iter)
               {
                  int inls = count_inliers();                  
                  if (optimizerVerbosenessLevel >= 2) cout << "HOM_Optimizer: level = " << level << " inner iteration " << iter << " lambda = " << damping_value
                                                           << " eta[level] = " << _etas[level] 
                                                           << " inls = " << inls 
                                                           << endl;

                  for (int obj = 0; obj < nObjs; ++obj)
                  {
                     NLSQ_Residuals& residuals = *_residuals[obj];
                     _robustCostFunctions[obj]->cache_IRLS_weights(_etas2[level], cached_errors[obj], residuals._weights);

                     int const K = residuals._weights.size();
                     for (int k = 0; k < K; ++k) scaleVectorIP(residuals._weights[k], residuals._residuals[k]);
                  }

                  double init_cost = this->eval_level_cost(0, cached_errors);
                  best_cost = min(best_cost, init_cost);
                  t.stop();
                  log_file << t.getTime() << "\t" << currentIteration << "\t" << best_cost << endl;
                  t.start();
                  this->fillJacobians();

                  this->evalJt_e(Jt_e);
                  scaleVectorIP(-1.0, Jt_e);

                  this->fillHessian();
                  this->fillJtJ();

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
                  ++currentIteration;
                  if (!success_LDL) ++LDL_failures;

                  bool success_iter = success_LDL;

                  if (success_LDL)
                  {
                     if (optimizerVerbosenessLevel >= 3)
                        cout << "HOM_Optimizer: ||delta|| = " << sqrNorm_L2(delta) << " ||paramLength|| = " << this->getParameterLength() << endl;

                     this->copyToAllParameters(&x_saved[0]);

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
                     double const current_cost = this->eval_level_cost(level, cached_errors);

                     if (optimizerVerbosenessLevel >= 1)
                     {
                        cout << "HOM_Optimizer: iteration " << currentIteration << " level = " << level << " new cost = " << current_cost << " existing cost = " << best_cost
                             << " lambda = " << damping_value 
                             << " #inls = " << inls 
                             << endl;
                        // log_file << currentIteration << "\t" << best_cost << endl;                          
                     }

                     success_iter = (current_cost < best_cost);
                     if (success_iter)
                        best_cost = current_cost;
                     else
                        this->copyFromAllParameters(&x_saved[0]);
                  } // end if (success_LDL)

                  if (success_iter)
                  {
                     damping_value = std::max(damping_value_min, damping_value / 10);
                  }
                  else
                  {
                     damping_value *= 10;
                  }
               } // end for (iter)
            } // end for (level)
            log_file.close();
#endif
         } // end minimize()
   }; // end struct HOM_Optimizer

//**********************************************************************

//    template <int NLevels>
//    struct Fast_HOM_Optimizer : public BT_IRLS_Optimizer<NLevels>
//    {
//          Fast_HOM_Optimizer(NLSQ_ParamDesc const& paramDesc,
//                             std::vector<NLSQ_CostFunction *> const& costFunctions,
//                             std::vector<Robust_NLSQ_CostFunction_Base *> const& robustCostFunctions)
//             : BT_IRLS_Optimizer<NLevels>(paramDesc, costFunctions, robustCostFunctions)
//          { }

//          typedef BT_IRLS_Optimizer<NLevels> Base;

//          using Base::status;
//          using Base::_JtJ;
//          using Base::_hessian;
//          using Base::_paramDesc;
//          using Base::_totalParamCount;
//          using Base::_paramTypeRowStart;
//          using Base::_residuals;
//          using Base::currentIteration;
//          using Base::maxIterations;
//          using Base::_costFunctions;
//          using Base::_robustCostFunctions;
//          using Base::_etas;
//          using Base::_etas2;

//          double eval_current_cost(vector<Vector<double> > const& cached_errors) const
//          {
//             int const nObjs = _robustCostFunctions.size();
//             double cost = 0;
//             for (int obj = 0; obj < nObjs; ++obj) cost += _robustCostFunctions[obj]->eval_target_cost(1.0, cached_errors[obj]);
//             return cost;
//          }

//          Vector2d eval_level_deltas(int const level, vector<Vector<double> > const& cached_errors, vector<Vector<double> > const& cached_errors0) const
//          {
//             double const scale2 = _etas2[level];

//             int const nObjs = _robustCostFunctions.size();

//             Vector2d res(0.0, 0.0);

//             for (int obj = 0; obj < nObjs; ++obj)
//             {
//                auto costFun = _robustCostFunctions[obj];
//                auto const& errors  = cached_errors[obj];
//                auto const& errors0 = cached_errors0[obj];

//                int const K = errors.size();

//                for (int k = 0; k < K; ++k)
//                {
//                   double const r2new = errors[k], r2old = errors0[k];
//                   if (r2new <= r2old)
//                      res[0] += std::max(0.0, costFun->eval_target_fun(scale2, r2old) - costFun->eval_target_fun(scale2, r2new));
//                   else
//                      res[1] += std::max(0.0, costFun->eval_target_fun(scale2, r2new) - costFun->eval_target_fun(scale2, r2old));
//                } // end for (k)
//             } // end for (obj)

//             return res;
//          } // end eval_level_deltas()

//          void minimize()
//          {
//             double const rho_stopping = 0.95;

//             status = LEVENBERG_OPTIMIZER_TIMEOUT;

//             if (_totalParamCount == 0)
//             {
//                // No degrees of freedom, nothing to optimize.
//                if (optimizerVerbosenessLevel >= 2) cout << "BT_Parallel_IRLS_Optimizer: exiting since d.o.f is zero." << endl;
//                status = LEVENBERG_OPTIMIZER_CONVERGED;
//                return;
//             }

//             int const totalParamDimension = _JtJ.num_cols();

//             vector<double> x0(totalParamDimension), x_saved(totalParamDimension);

//             Vector<double> Jt_e(totalParamDimension);
//             Vector<double> delta(totalParamDimension);
//             Vector<double> deltaPerm(totalParamDimension);

//             int const nObjs = _robustCostFunctions.size();

//             double damping_value = this->tau;

//             vector<Vector<double> > cached_errors(nObjs), cached_errors0(nObjs);
//             for (int obj = 0; obj < nObjs; ++obj) cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
//             for (int obj = 0; obj < nObjs; ++obj) cached_errors0[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);

//             for (int obj = 0; obj < nObjs; ++obj) fillVector(1.0, _residuals[obj]->_weights);

//             for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
//             double best_cost = this->eval_current_cost(cached_errors);

//             int LDL_failures = 0;

//             this->copyToAllParameters(&x0[0]);

//             currentIteration = 0;

//             for (int level = NLevels-1; level >= 0; --level)
//             {
//                bool success_LDL = true;

//                int const remainingIterations = maxIterations - currentIteration;
//                int const n_inner_iterations = int(0.5+double(remainingIterations)/(level+1));

//                for (int iter = 0; iter < n_inner_iterations; ++iter)
//                {
//                   for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
//                   double const Psi0_k  = this->eval_level_cost(level, cached_errors);
//                   double const Psi0_kk = (level > 0) ? this->eval_level_cost(level-1, cached_errors) : 0.0;

//                   for (int obj = 0; obj < nObjs; ++obj) cached_errors0[obj] = cached_errors[obj];

//                   //if (optimizerVerbosenessLevel >= 2) cout << "BT_IRLS_Optimizer: inner iteration " << iter << " lambda = " << damping_value << endl;

//                   for (int obj = 0; obj < nObjs; ++obj)
//                   {
//                      NLSQ_Residuals& residuals = *_residuals[obj];
//                      _robustCostFunctions[obj]->cache_IRLS_weights(_etas2[level], cached_errors[obj], residuals._weights);

//                      int const K = residuals._weights.size();
//                      for (int k = 0; k < K; ++k) scaleVectorIP(residuals._weights[k], residuals._residuals[k]);
//                   }

//                   this->fillJacobians();

//                   this->evalJt_e(Jt_e);
//                   scaleVectorIP(-1.0, Jt_e);

//                   this->fillHessian();
//                   this->fillJtJ();

//                   // Augment the diagonals
//                   for (int paramType = 0; paramType < _paramDesc.nParamTypes; ++paramType)
//                   {
//                      MatrixArray<double>& Hs = *_hessian.Hs[paramType][paramType];
//                      vector<pair<int, int> > const& nzPairs = _hessian.nonzeroPairs[paramType][paramType];
//                      int const dim = Hs.num_cols(), count = Hs.count();

//                      // Only augment those with the same parameter id
//                      for (int n = 0; n < count; ++n)
//                      {
//                         if (nzPairs[n].first != nzPairs[n].second) continue;
//                         for (int l = 0; l < dim; ++l) Hs[n][l][l] += damping_value;
//                      }
//                   } // end for (paramType)
//                   this->fillJtJ();
//                   success_LDL = this->solve_JtJ(Jt_e, deltaPerm, delta);
//                   ++currentIteration;
//                   if (!success_LDL) ++LDL_failures;

//                   bool stopping_reached = false;

//                   bool success_decrease = false;
//                   this->copyToAllParameters(&x_saved[0]);

//                   if (success_LDL)
//                   {
//                      double const deltaSqrLength = sqrNorm_L2(delta);
//                      double const paramLength = this->getParameterLength();

//                      if (optimizerVerbosenessLevel >= 3)
//                         cout << "Fast_HOM_Optimizer: ||delta|| = " << sqrt(deltaSqrLength) << " ||paramLength|| = " << paramLength << endl;

//                      for (int paramType = 0; paramType < _paramDesc.nParamTypes; ++paramType)
//                      {
//                         int const paramDim = _paramDesc.dimension[paramType];
//                         int const count    = _paramDesc.count[paramType];
//                         int const rowStart = _paramTypeRowStart[paramType];

//                         VectorArrayAdapter<double> deltaParam(count, paramDim, &delta[0] + rowStart);
//                         this->updateParameters(paramType, deltaParam);
//                      } // end for (paramType)
//                      this->finishUpdateParameters();
//                   } // end if (success_LDL)
                  
//                   if (success_LDL)
//                   {
//                      for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
//                      double const Psi1_k  = this->eval_level_cost(level, cached_errors);

//                      success_decrease = (Psi1_k < Psi0_k);

//                      // Check if we can jump to lower level early
//                      if (level > 0)
//                      {
//                         double const Psi1_kk = (level > 0) ? this->eval_level_cost(level-1, cached_errors) : 0.0;
// #if 0
//                         double const rho = (Psi0_kk - Psi1_kk) / (Psi0_k - Psi1_k);
//                         if (rho > rho_stopping) stopping_reached = true;
// #elif 0
//                         double const eta = 0.5, th_lo = 0.5*(eta-1.0)/eta, th_hi = 0.5*(eta+1.0)/eta;
//                         double const rho = (Psi0_kk - Psi1_kk) / (Psi0_k - Psi1_k);
//                         //cout << "rho = " << rho << " th_lo = " << th_lo << " th_hi = " << th_hi << endl;
//                         if (rho < th_lo || rho > th_hi) stopping_reached = true;
// #elif 1
//                         double const eta = 0.2;
//                         Vector2d const deltas = this->eval_level_deltas(level, cached_errors, cached_errors0);
//                         double const rho = (deltas[0] - deltas[1]) / (deltas[0] + deltas[1]);
//                         if (rho < eta) stopping_reached = true;
// #else
//                         double const rho = fabs(Psi0_kk - Psi1_kk) - (Psi0_k - Psi1_k);
//                         if (rho > 0.0) stopping_reached = true;
// #endif
//                      } // end if

//                      if (optimizerVerbosenessLevel >= 1)
//                      {
//                         double const current_cost = this->eval_level_cost(0, cached_errors);
//                         best_cost = std::min(best_cost, current_cost);
//                         if (optimizerVerbosenessLevel == 1)
//                            cout << "Fast_HOM_Optimizer: iteration " << currentIteration << " level = " << level << " new cost = " << current_cost << " best cost = " << best_cost << " lambda = " << damping_value << endl;
//                         else
//                            cout << "Fast_HOM_Optimizer: iteration " << currentIteration << " level = " << level << " new cost = " << current_cost << " best cost = " << best_cost << " lambda = " << damping_value
//                                 << " Psi_k(prev) = " << Psi0_k << " Psi_k(new) = " << Psi1_k << endl;
//                      }
//                   } // end (if success_LDL)

//                   if (success_decrease)
//                   {
//                      damping_value = std::max(damping_value_min, damping_value / 10);
//                      this->copyToAllParameters(&x0[0]);
//                   }
//                   else
//                   {
//                      if (success_LDL) this->copyFromAllParameters(&x_saved[0]);
//                      damping_value *= 10;
//                   }

//                   if (stopping_reached) break;
//                } // end for (iter)
//             } // end for (level)

//             this->copyFromAllParameters(&x0[0]);
//          } // end minimize()
//    }; // end struct Fast_HOM_Optimizer

} // end namespace Robust_LSQ

#endif
