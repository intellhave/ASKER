// -*- C++ -*-
#ifndef ROBUST_LSQ_BT_H
#define ROBUST_LSQ_BT_H

#include "robust_lsq_common.h"

namespace Robust_LSQ
{

   //double constexpr eta_multiplier = sqrt(2.0);
   //double constexpr eta_multiplier = sqrt(sqrt(2.0));

   template <int NLevels>
   struct BT_IRLS_Optimizer : public Robust_LSQ_Optimizer_Base
   {
         static constexpr bool init_with_optimistic_weights = 0;
         static constexpr double eta_multiplier = 2;

         BT_IRLS_Optimizer(NLSQ_ParamDesc const& paramDesc,
                           std::vector<NLSQ_CostFunction *> const& costFunctions,
                           std::vector<Robust_NLSQ_CostFunction_Base *> const& robustCostFunctions)
            : Robust_LSQ_Optimizer_Base(paramDesc, costFunctions, robustCostFunctions)
         {
            this->fillEtas();
         }

         ~BT_IRLS_Optimizer() { }

         double _etas[NLevels], _etas2[NLevels];

         virtual void fillEtas()
         {
            _etas[0] = 1.0;
            for (int k = 1; k < NLevels; ++k) _etas[k]  = eta_multiplier * _etas[k-1];
            for (int k = 0; k < NLevels; ++k) _etas2[k] = sqr(_etas[k]);
         }

         virtual bool allowStoppingCriteria() const { return true; }

         double eval_level_cost(int const level, vector<Vector<double> > const& cached_errors) const
         {
            int const nObjs = _robustCostFunctions.size();
            double cost = 0;
            for (int obj = 0; obj < nObjs; ++obj) cost += _robustCostFunctions[obj]->eval_target_cost(_etas2[level], cached_errors[obj]);
            return cost;
         }

         double eval_weights_sensitivity(int const level, vector<Vector<double> > const& cached_errors) const
         {
            int const nObjs = _robustCostFunctions.size();
            double res = 0;
            for (int obj = 0; obj < nObjs; ++obj)
               res += _robustCostFunctions[obj]->eval_weights_sensitivity(_etas2[level], cached_errors[obj]);
            return sqrt(res);
         }

         double eval_robust_objective() const
         {
            int const nObjs = _robustCostFunctions.size();
            vector<Vector<double> > cached_errors(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
            for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
            return this->eval_level_cost(0, cached_errors);
         }

//**********************************************************************

         void minimize()
         {
            int const n_inner_iterations = 1;

            status = LEVENBERG_OPTIMIZER_TIMEOUT;

            if (_totalParamCount == 0)
            {
               // No degrees of freedom, nothing to optimize.
               if (optimizerVerbosenessLevel >= 2) cout << "BT_IRLS_Optimizer: exiting since d.o.f is zero." << endl;
               status = LEVENBERG_OPTIMIZER_CONVERGED;
               return;
            }

            int const totalParamDimension = _JtJ.num_cols();

            vector<double> trial_points[NLevels];
            double         trial_costs[NLevels];  std::fill_n(trial_costs, NLevels, 1e30);
            // status < 0 means no trial point, otherwise number of refinement iterations
            int            trial_status[NLevels]; std::fill_n(trial_status, NLevels, -1);
            double         trial_quality[NLevels]; std::fill_n(trial_quality, NLevels, -1.0);
            double         best_costs[NLevels]; std::fill_n(best_costs, NLevels, 1e30);
            int            source_levels[NLevels]; std::fill_n(source_levels, NLevels, -1);

            vector<double> x0(totalParamDimension);

            for (int l = 0; l < NLevels; ++l) trial_points[l].resize(totalParamDimension);

            Vector<double> Jt_e(totalParamDimension);
            Vector<double> delta(totalParamDimension);
            Vector<double> deltaPerm(totalParamDimension);

            int const nObjs = _robustCostFunctions.size();

            double damping_value = this->tau;

            vector<Vector<double> > cached_errors(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);

            // Initialize costs and enqueue starting point
            {
               this->copyToAllParameters(&x0[0]);

               for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);

               for (int level = 0; level < NLevels; ++level)
               {
                  double const current_cost = this->eval_level_cost(level, cached_errors);
                  best_costs[level] = current_cost;

                  if (level == 0) // insert only level 0
                  //if (level == NLevels-1) // insert only at coarsest level
                  {
                     trial_costs[level]   = current_cost;
                     trial_status[level]  = 0;
                     trial_quality[level] = 1e30;
                     //trial_quality[level] = this->eval_weights_sensitivity(level, cached_errors);
                     source_levels[level] = level;
                     this->copyToAllParameters(&trial_points[level][0]);
                  }
               } // end for (level)
            } // end scope

            int LDL_failures = 0;

            for (currentIteration = 0; currentIteration < maxIterations; ++currentIteration)
            {
               if (optimizerVerbosenessLevel >= 2)
               {
                  cout << "BT_IRLS_Optimizer: trial points = [";
                  for (int k = 0; k < NLevels; ++k) cout << trial_status[k] << ":" << trial_quality[k] << " ";
                  cout << "]" << endl;
               }

               int level = 0; while (level < NLevels && trial_status[level] < 0) ++level;
               //level = NLevels-1; while (level >= 0 && !trial_is_valid[level]) --level;

               if (level < 0 || level >= NLevels)
               {
                  if (optimizerVerbosenessLevel >= 2)
                     cout << "BT_IRLS_Optimizer: no trial point found, exiting." << endl;
                  break;
               }

               // Look if there is another level with higher priority
               for (int k = level+1; k < NLevels; ++k)
               {
                  //if (trial_status[k] >= 0 && trial_status[k] < trial_status[level]) level = k;
                  if (trial_status[k] >= 0 && trial_quality[k] > trial_quality[level]) level = k;
               } // end for (k)

               this->copyFromAllParameters(&trial_points[level][0]);

               for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
               double const initial_cost = this->eval_level_cost(level, cached_errors);

               if (optimizerVerbosenessLevel >= 1)
               {
                  cout << "BT_IRLS_Optimizer: iteration: " << currentIteration << ", level = " << level << ", initial |residual|^2 = " << initial_cost << ", eta = " << _etas[level]
                       << " best costs[0] = " << best_costs[0] << " lambda = " << damping_value << endl;
                  //for (int k = 0; k < NLevels; ++k) cout << int(trial_status[k]) << " "; cout << endl;
               }

               bool level_converged = false, success_LDL = true, is_sufficient_decrease = false;
               double norm_Jt_e = 0;

               for (int iter = 0; iter < n_inner_iterations; ++iter)
               {
                  for (int obj = 0; obj < nObjs; ++obj)
                  {
                     NLSQ_Residuals& residuals = *_residuals[obj];
                     if (init_with_optimistic_weights && trial_status[level] == 0)
                        fillVector(1.0, residuals._weights);
                     else
                        _robustCostFunctions[obj]->cache_IRLS_weights(_etas2[level], cached_errors[obj], residuals._weights);

                     int const K = residuals._weights.size();
                     for (int k = 0; k < K; ++k) scaleVectorIP(residuals._weights[k], residuals._residuals[k]);
                  }

                  this->fillJacobians();

                  this->evalJt_e(Jt_e);
                  scaleVectorIP(-1.0, Jt_e);
                  double const norm_Linf_Jt_e = norm_Linf(Jt_e);
                  norm_Jt_e = norm_L2(Jt_e);

                  if (this->allowStoppingCriteria() && this->applyGradientStoppingCriteria(norm_Linf_Jt_e))
                  {
                     if (optimizerVerbosenessLevel >= 2) cout << "BT_IRLS_Optimizer: converged at current level due to gradient stopping," << "norm_Linf_Jt_e = " << norm_Linf_Jt_e << endl;
                     level_converged = true; goto after_inner;
                  }

                  this->fillHessian();
                  this->fillJtJ();
                  success_LDL = 0;

                  if (1 && !success_LDL)
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

                  if (success_LDL)
                  {
                     double const deltaSqrLength = sqrNorm_L2(delta);
                     double const paramLength = this->getParameterLength();

                     if (optimizerVerbosenessLevel >= 3)
                        cout << "BT_IRLS_Optimizer: ||delta|| = " << sqrt(deltaSqrLength) << " ||paramLength|| = " << paramLength << endl;

                     if (this->allowStoppingCriteria() && this->applyUpdateStoppingCriteria(paramLength, sqrt(deltaSqrLength)))
                     {
                        if (optimizerVerbosenessLevel >= 2) cout << "BT_IRLS_Optimizer: converged at current level to small update, deltaSqrLength = " << deltaSqrLength << endl;
                        level_converged = true; goto after_inner;
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
                     ++trial_status[level];

                     damping_value = std::max(damping_value_min, damping_value / 10);
                  }
                  else
                  {
                     damping_value *= 10;
                     break;
                  } // end (if success_LDL)
               } // end for (iter)

            after_inner:
               if (success_LDL)
               {
                  // Check if new cost is better than best one at current level
                  double const current_cost = this->eval_level_cost(level, cached_errors);

                  is_sufficient_decrease = (initial_cost - current_cost > 1e-2);

                  if (optimizerVerbosenessLevel >= 2) cout << "BT_IRLS_Optimizer: success_LDL = " << int(success_LDL) << " new cost = " << current_cost << " existing cost = " << best_costs[level] << endl;

                  if (current_cost < best_costs[level])
                  {
                     best_costs[level] = current_cost;
                     // Only copying of params at level 0 required to return the solution
                     if (level == 0) this->copyToAllParameters(&x0[0]);
                  }

                  if (level_converged)
                  {
                     trial_status[level] = -1;
                     trial_quality[level] = -1;
                  }
                  else
                  {
                     this->copyToAllParameters(&trial_points[level][0]); // copy back improved solution
                     trial_quality[level] = norm_Jt_e / _etas2[level];
                     //trial_quality[level] = norm_Jt_e / _etas[level];
                     //trial_quality[level] = norm_Jt_e;
                     //trial_quality[level] = (initial_cost - current_cost) / _etas2[level];
                     //trial_quality[level] = this->eval_weights_sensitivity(level, cached_errors);
                  }
#if 0
                  // Check if we obtained a very strong solution
                  int next_level = level;
                  for (int l = level-1; l >= 0; --l)
                  {
                     double const next_level_cost = this->eval_level_cost(l, cached_errors);

                     if (optimizerVerbosenessLevel >= 2)
                        cout << "BT_IRLS_Optimizer: testing level " << l << " new cost = " << next_level_cost << " existing cost = " << best_costs[l] << endl;

                     if (next_level_cost < best_costs[l])
                     {
                        if (l == 0) this->copyToAllParameters(&x0[0]);
                        best_costs[l] = next_level_cost;
                        trial_status[l] = -1;
                        source_levels[l] = -1;
                        next_level = l;
                     }
                     else
                        break;
                  } // end for (l)

                  if (next_level < level)
                  {
                     if (optimizerVerbosenessLevel >= 2)
                        cout << "BT_IRLS_Optimizer: Very strong move, adding trial for level " << next_level << " cost = " << best_costs[next_level] << endl;

                     trial_status[level]  = -1;
                     trial_quality[level] = -1;
                     source_levels[level] = -1;

                     trial_status[next_level]  = 0;
                     trial_costs[next_level]   = best_costs[next_level];
                     trial_quality[next_level] = 1e30;
                     //trial_quality[next_level] = this->eval_weights_sensitivity(next_level, cached_errors);
                     source_levels[next_level] = level;
                     this->copyToAllParameters(&trial_points[next_level][0]);
                  }
                  else if ((1 || !is_sufficient_decrease) && level < NLevels-1)
                  {
                     next_level = level+1;
                     next_level = std::max(next_level, source_levels[level]+1);
                     next_level = std::min(next_level, NLevels-1);

                     double const current_cost_up = this->eval_level_cost(next_level, cached_errors);
                     //if (optimizerVerbosenessLevel >= 2) cout << "BT_IRLS_Optimizer: testing upward move, cost_up = " << current_cost_up << " trial_status[next_level] = " << trial_status[next_level] << endl;

                     // Insert trial point at coarser level, if it is more promising (or not existent at coarser level)
                     if ((trial_status[next_level] < 0) || (current_cost_up < trial_costs[next_level]))
                     {
                        if (optimizerVerbosenessLevel >= 2) cout << "BT_IRLS_Optimizer: going up to level " << next_level << endl;

                        trial_status[next_level]  = 0;
                        trial_costs[next_level]   = current_cost_up;
                        trial_quality[next_level] = 1e30;
                        //trial_quality[next_level] = this->eval_weights_sensitivity(next_level, cached_errors);
                        source_levels[next_level] = level;
                        this->copyToAllParameters(&trial_points[next_level][0]);
                     } // end if
                  } // end if
#else
                  // Check if we obtained a very strong solution
                  bool is_strong_step = false, is_very_strong_step = false;
                  for (int l = 0; l < level; ++l)
                  {
                     double const next_level_cost = this->eval_level_cost(l, cached_errors);

                     if (optimizerVerbosenessLevel >= 2)
                        cout << "BT_IRLS_Optimizer: testing level " << l << " new cost = " << next_level_cost << " existing cost = " << best_costs[l] << endl;

                     if (next_level_cost < best_costs[l])
                     {
                        is_very_strong_step = true;
                        if (l == 0) this->copyToAllParameters(&x0[0]);
                        best_costs[l] = next_level_cost;
                     }
                     if ((trial_status[l] < 0) || (next_level_cost < trial_costs[l]))
                     {
                        is_strong_step = true;
                        trial_status[l]  = 0;
                        trial_costs[l]   = next_level_cost;
                        trial_quality[l] = 1e30;
                        //trial_quality[l] = this->eval_weights_sensitivity(next_level, cached_errors);
                        source_levels[l] = level;
                        this->copyToAllParameters(&trial_points[l][0]);
                     }
                  } // end for (l)

                  if (!is_very_strong_step  && (1 || !is_sufficient_decrease) && level < NLevels-1)
                  {
                     int next_level = level+1;
                     next_level = std::max(next_level, source_levels[level]+1);
                     next_level = std::min(next_level, NLevels-1);

                     double const current_cost_up = this->eval_level_cost(next_level, cached_errors);
                     //if (optimizerVerbosenessLevel >= 2) cout << "BT_IRLS_Optimizer: testing upward move, cost_up = " << current_cost_up << " trial_status[next_level] = " << trial_status[next_level] << endl;

                     // Insert trial point at coarser level, if it is more promising (or not existent at coarser level)
                     if ((trial_status[next_level] < 0) || (current_cost_up < trial_costs[next_level]))
                     {
                        if (optimizerVerbosenessLevel >= 2) cout << "BT_IRLS_Optimizer: going up to level " << next_level << endl;

                        trial_status[next_level]  = 0;
                        trial_costs[next_level]   = current_cost_up;
                        trial_quality[next_level] = 1e30;
                        //trial_quality[next_level] = this->eval_weights_sensitivity(next_level, cached_errors);
                        source_levels[next_level] = level;
                        this->copyToAllParameters(&trial_points[next_level][0]);
                     } // end if
                  } // end if
#endif
               } // end if (success_LDL)
            } // end for (currentIteration)

            if (optimizerVerbosenessLevel >= 1 && currentIteration+1 >= maxIterations)
            {
               cout << "BT_IRLS_Optimizer: reached maximum number of iterations, exiting." << endl;
            }

            if (optimizerVerbosenessLevel >= 1)
            {
               //cout << "Leaving BT_IRLS_Optimizer::minimize()." << endl;
               cout << "LDL_failures = " << LDL_failures << " trial_status = "; for (int k = 0; k < NLevels; ++k) cout << int(trial_status[k]) << " "; cout << endl;
            }

            this->copyFromAllParameters(&x0[0]);
         } // end minimize()
   }; // end struct BT_IRLS_Optimizer

//**********************************************************************

   // template <int NLevels>
   // struct BT_SqrtPsi_Optimizer : public BT_IRLS_Optimizer<NLevels>
   // {
   //       BT_SqrtPsi_Optimizer(NLSQ_ParamDesc const& paramDesc,
   //                            std::vector<NLSQ_CostFunction *> const& costFunctions,
   //                            std::vector<Robust_NLSQ_CostFunction_Base *> const& robustCostFunctions)
   //          : BT_IRLS_Optimizer<NLevels>(paramDesc, costFunctions, robustCostFunctions)
   //       { }

   //       typedef BT_IRLS_Optimizer<NLevels> Base;

   //       virtual bool allowStoppingCriteria() const { return true; }

   //       void minimize()
   //       {
   //          int const n_inner_iterations = 1;

   //          Base::status = LEVENBERG_OPTIMIZER_TIMEOUT;

   //          if (Base::_totalParamCount == 0)
   //          {
   //             // No degrees of freedom, nothing to optimize.
   //             if (optimizerVerbosenessLevel >= 2) cout << "BT_SqrtPsi_Optimizer: exiting since d.o.f is zero." << endl;
   //             Base::status = LEVENBERG_OPTIMIZER_CONVERGED;
   //             return;
   //          }

   //          int const totalParamDimension = Base::_JtJ.num_cols();

   //          vector<double> trial_points[NLevels];
   //          double         trial_costs[NLevels];  std::fill_n(trial_costs, NLevels, 1e30);
   //          // status < 0 means no trial point, otherwise number of refinement iterations
   //          int            trial_status[NLevels]; std::fill_n(trial_status, NLevels, -1);
   //          double         trial_quality[NLevels]; std::fill_n(trial_quality, NLevels, -1.0);
   //          double         best_costs[NLevels]; std::fill_n(best_costs, NLevels, 1e30);
   //          vector<double> x0(totalParamDimension);

   //          for (int l = 0; l < NLevels; ++l) trial_points[l].resize(totalParamDimension);

   //          Vector<double> Jt_e(totalParamDimension);
   //          Vector<double> delta(totalParamDimension);
   //          Vector<double> deltaPerm(totalParamDimension);

   //          int const nObjs = Base::_robustCostFunctions.size();

   //          double damping_value = this->tau;

   //          vector<Vector<double> > cached_errors(nObjs);
   //          for (int obj = 0; obj < nObjs; ++obj) cached_errors[obj].newsize(Base::_robustCostFunctions[obj]->_nMeasurements);

   //          // Initialize costs and enqueue starting point
   //          {
   //             this->copyToAllParameters(&x0[0]);

   //             for (int obj = 0; obj < nObjs; ++obj) Base::_robustCostFunctions[obj]->cache_residuals(*Base::_residuals[obj], cached_errors[obj]);

   //             for (int level = 0; level < NLevels; ++level)
   //             {
   //                double const current_cost = this->eval_level_cost(level, cached_errors);
   //                best_costs[level] = current_cost;

   //                //if (level == 0) // insert only level 0
   //                if (level == NLevels-1) // insert only at coarsest level
   //                {
   //                   trial_costs[level]   = current_cost;
   //                   trial_status[level]  = 0;
   //                   trial_quality[level] = 1e30;
   //                   //trial_quality[level] = this->eval_weights_sensitivity(level, cached_errors);
   //                   this->copyToAllParameters(&trial_points[level][0]);
   //                }
   //             } // end for (level)
   //          } // end scope

   //          int LDL_failures = 0;

   //          for (Base::currentIteration = 0; Base::currentIteration < Base::maxIterations; ++Base::currentIteration)
   //          {
   //             if (optimizerVerbosenessLevel >= 2)
   //             {
   //                cout << "BT_SqrtPsi_Optimizer: trial points = [";
   //                for (int k = 0; k < NLevels; ++k) cout << trial_status[k] << ":" << trial_quality[k] << " ";
   //                cout << "]" << endl;
   //             }

   //             int level = 0; while (level < NLevels && trial_status[level] < 0) ++level;
   //             //level = NLevels-1; while (level >= 0 && !trial_is_valid[level]) --level;

   //             if (level < 0 || level >= NLevels)
   //             {
   //                if (optimizerVerbosenessLevel >= 2)
   //                   cout << "BT_SqrtPsi_Optimizer: no trial point found, exiting." << endl;
   //                break;
   //             }

   //             // Look if there is another level with higher priority
   //             for (int k = level+1; k < NLevels; ++k)
   //             {
   //                //if (trial_status[k] >= 0 && trial_status[k] < trial_status[level]) level = k;
   //                if (trial_status[k] >= 0 && trial_quality[k] > trial_quality[level]) level = k;
   //             } // end for (k)

   //             this->copyFromAllParameters(&trial_points[level][0]);

   //             for (int obj = 0; obj < nObjs; ++obj) Base::_robustCostFunctions[obj]->cache_residuals(*Base::_residuals[obj], cached_errors[obj]);
   //             double const initial_cost = this->eval_level_cost(level, cached_errors);

   //             if (optimizerVerbosenessLevel >= 1)
   //             {
   //                cout << "BT_SqrtPsi_Optimizer: iteration: " << Base::currentIteration << ", level = " << level << ", initial |residual|^2 = " << initial_cost << ", eta = " << Base::_etas[level]
   //                     << " best costs[0] = " << best_costs[0] << " lambda = " << damping_value << endl;
   //                //for (int k = 0; k < NLevels; ++k) cout << int(trial_status[k]) << " "; cout << endl;
   //             }

   //             bool level_converged = false, success_LDL = true;
   //             double norm_Jt_e = 0;

   //             //for (int iter = 0; iter < n_inner_iterations; ++iter)
   //             {
   //                // Here we use the sqrt(psi) method, which means we have to modify the residuals and Jacobians
   //                for (int obj = 0; obj < nObjs; ++obj)
   //                {
   //                   NLSQ_CostFunction&             costFun        = *Base::_costFunctions[obj];
   //                   Robust_NLSQ_CostFunction_Base& robustCostFun  = *Base::_robustCostFunctions[obj];
   //                   NLSQ_Residuals&                residuals      = *Base::_residuals[obj];
   //                   Vector<double> const&          errors         = cached_errors[obj];
   //                   int const K = costFun._nMeasurements;

   //                   Vector<double> sqrt_psi_values(K), weights(K), gamma_values(K);
   //                   fillVector(1.0, weights);
   //                   fillVector(1.0, residuals._weights);

   //                   costFun.fillAllJacobians(weights, residuals._Js);

   //                   robustCostFun.cache_weight_fun(Base::_etas2[level], errors, weights);
   //                   robustCostFun.cache_sqrt_psi_values(Base::_etas2[level], errors, sqrt_psi_values);
   //                   robustCostFun.cache_gamma_values(Base::_etas2[level], errors, gamma_values);

   //                   // Rescale residuals
   //                   for (int k = 0; k < K; ++k)
   //                   {
   //                      double const r2 = errors[k];
   //                      if (r2 > 1e-6)
   //                      {
   //                         double const sqrt_psi = sqrt_psi_values[k], norm_r = sqrt(r2);
   //                         scaleVectorIP(sqrt_psi / norm_r, residuals._residuals[k]);
   //                      }
   //                      else
   //                      {
   //                         scaleVectorIP(0.70710678118, residuals._residuals[k]);
   //                      }
   //                   } // end for (k)

   //                   // Now adjust the Jacobians
   //                   Matrix<double> outer(costFun._measurementDimension, costFun._measurementDimension, 0.0);
   //                   for (int i = 0; i < costFun._usedParamTypes.size(); ++i)
   //                   {
   //                      MatrixArray<double>& J = *residuals._Js[i];
   //                      Matrix<double> J0(J.num_rows(), J.num_cols());

   //                      for (int k = 0; k < K; ++k)
   //                      {
   //                         double const r2 = errors[k];
   //                         if (r2 > 1e-6)
   //                         {
   //                            double const norm_r = sqrt(r2), r_norm3 = 1.0 / (r2 * norm_r);
   //                            makeOuterProductMatrix(residuals._residuals[k], outer);
   //                            scaleMatrixIP(-gamma_values[k] * r_norm3 / sqrt_psi_values[k], outer);
   //                            for (int j = 0; j < costFun._measurementDimension; ++j) outer[j][j] += sqrt_psi_values[k] / norm_r;

   //                            copyMatrix(J[k], J0); multiply_A_B(outer, J0, J[k]);
   //                         }
   //                         else
   //                            scaleMatrixIP(0.70710678118, J[k]);
   //                      } // end for (k)
   //                   } // end for (i)
   //                } // end for (obj)

   //                this->evalJt_e(Jt_e);
   //                scaleVectorIP(-1.0, Jt_e);
   //                double const norm_Linf_Jt_e = norm_Linf(Jt_e);
   //                norm_Jt_e = norm_L2(Jt_e);

   //                if (this->allowStoppingCriteria() && this->applyGradientStoppingCriteria(norm_Linf_Jt_e))
   //                {
   //                   if (optimizerVerbosenessLevel >= 2) cout << "BT_SqrtPsi_Optimizer: converged at current level due to gradient stopping," << "norm_Linf_Jt_e = " << norm_Linf_Jt_e << endl;
   //                   level_converged = true; goto after_inner;
   //                }

   //                this->fillHessian();
   //                this->fillJtJ();
   //                success_LDL = 0;

   //                // Augment the diagonals
   //                for (int paramType = 0; paramType < Base::_paramDesc.nParamTypes; ++paramType)
   //                {
   //                   MatrixArray<double>& Hs = *Base::_hessian.Hs[paramType][paramType];
   //                   vector<pair<int, int> > const& nzPairs = Base::_hessian.nonzeroPairs[paramType][paramType];
   //                   int const dim = Hs.num_cols(), count = Hs.count();

   //                   // Only augment those with the same parameter id
   //                   for (int n = 0; n < count; ++n)
   //                   {
   //                      if (nzPairs[n].first != nzPairs[n].second) continue;
   //                      for (int l = 0; l < dim; ++l) Hs[n][l][l] += damping_value;
   //                   }
   //                } // end for (paramType)

   //                this->fillJtJ();
   //                success_LDL = this->solve_JtJ(Jt_e, deltaPerm, delta);
   //                if (!success_LDL) ++LDL_failures;

   //                if (success_LDL)
   //                {
   //                   double const deltaSqrLength = sqrNorm_L2(delta);
   //                   double const paramLength = this->getParameterLength();

   //                   if (optimizerVerbosenessLevel >= 3)
   //                      cout << "BT_SqrtPsi_Optimizer: ||delta|| = " << sqrt(deltaSqrLength) << " ||paramLength|| = " << paramLength << endl;

   //                   if (this->allowStoppingCriteria() && this->applyUpdateStoppingCriteria(paramLength, sqrt(deltaSqrLength)))
   //                   {
   //                      if (optimizerVerbosenessLevel >= 2) cout << "BT_SqrtPsi_Optimizer: converged at current level to small update, deltaSqrLength = " << deltaSqrLength << endl;
   //                      level_converged = true; goto after_inner;
   //                   }

   //                   for (int paramType = 0; paramType < Base::_paramDesc.nParamTypes; ++paramType)
   //                   {
   //                      int const paramDim = Base::_paramDesc.dimension[paramType];
   //                      int const count    = Base::_paramDesc.count[paramType];
   //                      int const rowStart = Base::_paramTypeRowStart[paramType];

   //                      VectorArrayAdapter<double> deltaParam(count, paramDim, &delta[0] + rowStart);
   //                      this->updateParameters(paramType, deltaParam);
   //                   } // end for (paramType)
   //                   this->finishUpdateParameters();

   //                   for (int obj = 0; obj < nObjs; ++obj) Base::_robustCostFunctions[obj]->cache_residuals(*Base::_residuals[obj], cached_errors[obj]);
   //                   ++trial_status[level];

   //                   damping_value = std::max(damping_value_min, damping_value / 10);
   //                }
   //                else
   //                {
   //                   damping_value *= 10;
   //                   break;
   //                } // end (if success_LDL)
   //             } // end for (iter)

   //          after_inner:
   //             if (success_LDL)
   //             {
   //                // Check if new cost is better than best one at current level
   //                double const current_cost = this->eval_level_cost(level, cached_errors);

   //                if (optimizerVerbosenessLevel >= 2) cout << "BT_IRLS_Optimizer: success_LDL = " << int(success_LDL) << " new cost = " << current_cost << " existing cost = " << best_costs[level] << endl;

   //                if (current_cost < best_costs[level])
   //                {
   //                   best_costs[level] = current_cost;
   //                   // Only copying of params at level 0 required to return the solution
   //                   if (level == 0) this->copyToAllParameters(&x0[0]);
   //                }

   //                if (level_converged)
   //                {
   //                   trial_status[level] = -1;
   //                   trial_quality[level] = -1;
   //                }
   //                else
   //                {
   //                   this->copyToAllParameters(&trial_points[level][0]); // copy back improved solution
   //                   trial_quality[level] = norm_Jt_e / Base::_etas2[level];
   //                   //trial_quality[level] = norm_Jt_e;
   //                   //trial_quality[level] = this->eval_weights_sensitivity(level, cached_errors);
   //                }

   //                // Check if we obtained a very strong solution
   //                int next_level = level;
   //                for (int l = level-1; l >= 0; --l)
   //                {
   //                   double const next_level_cost = this->eval_level_cost(l, cached_errors);

   //                   if (optimizerVerbosenessLevel >= 2)
   //                      cout << "BT_SqrtPsi_Optimizer: testing level " << l << " new cost = " << next_level_cost << " existing cost = " << best_costs[l] << endl;

   //                   if (next_level_cost < best_costs[l])
   //                   {
   //                      if (l == 0) this->copyToAllParameters(&x0[0]);
   //                      best_costs[l] = next_level_cost;
   //                      trial_status[l] = -1;
   //                      next_level = l;
   //                   }
   //                   else
   //                      break;
   //                } // end for (l)

   //                if (next_level < level)
   //                {
   //                   if (optimizerVerbosenessLevel >= 2)
   //                      cout << "BT_SqrtPsi_Optimizer: Very strong move, adding trial for level " << next_level << " cost = " << best_costs[next_level] << endl;

   //                   trial_status[level]       = -1;
   //                   trial_quality[level]      = -1;
   //                   trial_status[next_level]  = 0;
   //                   trial_costs[next_level]   = best_costs[next_level];
   //                   trial_quality[next_level] = 1e30;
   //                   //trial_quality[next_level] = this->eval_weights_sensitivity(next_level, cached_errors);
   //                   this->copyToAllParameters(&trial_points[next_level][0]);
   //                }
   //                else if (level < NLevels-1)
   //                {
   //                   double const current_cost_up = this->eval_level_cost(level+1, cached_errors);

   //                   //if (optimizerVerbosenessLevel >= 2) cout << "BT_IRLS_Optimizer: testing upward move, cost_up = " << current_cost_up << " trial_status[level+1] = " << trial_status[level+1] << endl;

   //                   // Insert trial point at coarser level, if it is more promising (or not existent at coarser level)
   //                   if ((trial_status[level+1] < 0) || (current_cost_up < trial_costs[level+1]))
   //                   {
   //                      if (optimizerVerbosenessLevel >= 2) cout << "BT_SqrtPsi_Optimizer: going one level up." << endl;

   //                      trial_status[level+1]  = 0;
   //                      trial_costs[level+1]   = current_cost_up;
   //                      trial_quality[level+1] = 1e30;
   //                      //trial_quality[level+1] = this->eval_weights_sensitivity(level+1, cached_errors);
   //                      this->copyToAllParameters(&trial_points[level+1][0]);
   //                   }
   //                } // end if
   //             } // end if (success_LDL)
   //          } // end for (currentIteration)

   //          if (optimizerVerbosenessLevel >= 1 && Base::currentIteration+1 >= Base::maxIterations)
   //          {
   //             cout << "BT_SqrtPsi_Optimizer: reached maximum number of iterations, exiting." << endl;
   //          }

   //          if (optimizerVerbosenessLevel >= 2)
   //          {
   //             cout << "Leaving BT_SqrtPsi_Optimizer::minimize()." << endl;
   //             cout << "LDL_failures = " << LDL_failures << " trial_status = ";
   //             for (int k = 0; k < NLevels; ++k) cout << int(trial_status[k]) << " "; cout << endl;
   //          }

   //          this->copyFromAllParameters(&x0[0]);
   //       } // end minimize()
   // }; // end struct BT_SqrtPsi_Optimizer

} // end namespace Robust_LSQ

#endif
