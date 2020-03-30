// -*- C++ -*-
#ifndef ROBUST_LSQ_GNC_MOO_H
#define ROBUST_LSQ_GNC_MOO_H

#include "robust_lsq_bt.h"
#include "Base/v3d_timer.h"
#include <iomanip>
#include <random>
#include <fstream>

namespace Robust_LSQ
{

   namespace MOO_HOM_Optimizer_Params
   {

      constexpr double cosine_threshold = -0.95; //-0.95;
      constexpr double g_ratio_threshold = 0.1;

      constexpr bool precondition_g1 = 1;

   }  // end namespace MOO_HOM_Optimizer_Params

   template <int NLevels>
   struct MOO_HOM_Optimizer : public HOM_Optimizer_Base<NLevels>
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

         static double constexpr beta = 0.5, beta1 = 1.0 - beta;

         static bool constexpr only_distort_gradient = 0;
         static bool constexpr use_random_distortion = 0;

         MOO_HOM_Optimizer(NLSQ_ParamDesc const& paramDesc,
                           std::vector<NLSQ_CostFunction *> const& costFunctions,
                           std::vector<Robust_NLSQ_CostFunction_Base *> const& robustCostFunctions)
            : Base(paramDesc, costFunctions, robustCostFunctions)
         { }

         static double get_scale1(double const beta, double const sqrNorm_g0, double const sqrNorm_g1, double const dot_g0_g1)
         {
            if (sqrNorm_g1 < 1e-12) return 0.0;
            if (beta == 0.5) return sqrt(sqrNorm_g0/sqrNorm_g1);
            double const a = beta*sqrNorm_g1, b = (1.0-2.0*beta)*dot_g0_g1, c = (beta-1.0)*sqrNorm_g0;
            double const D = sqrt(std::max(0.0, b*b - 4.0*a*c));
            return 0.5*(D - b)/a;
         }

         void minimize()
         {
            std::default_random_engine rng;
            std::normal_distribution<double> dist_normal(0.0, 1.0);
            std::uniform_real_distribution<double> dist_unif(-0.1, 0.1);

            status = LEVENBERG_OPTIMIZER_TIMEOUT;

            if (_totalParamCount == 0)
            {
               // No degrees of freedom, nothing to optimize.
               if (optimizerVerbosenessLevel >= 2) cout << "MOO_HOM_Optimizer: exiting since d.o.f is zero." << endl;
               status = LEVENBERG_OPTIMIZER_CONVERGED;
               return;
            }

            int const totalParamDimension = _JtJ.num_cols();

            vector<double> x_saved(totalParamDimension);

            Vector<double> Jt_e(totalParamDimension), g0(totalParamDimension), g1(totalParamDimension);
            Vector<double> delta(totalParamDimension);
            Vector<double> deltaPerm(totalParamDimension);

            int const nObjs = _robustCostFunctions.size();

            double damping_value = this->tau;

            vector<Vector<double> > cached_errors(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);

            vector<Vector<double> > cached_weights1(nObjs), cached_weights0(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) cached_weights1[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
            for (int obj = 0; obj < nObjs; ++obj) cached_weights0[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);

            for (int obj = 0; obj < nObjs; ++obj) fillVector(1.0, _residuals[obj]->_weights);

            double best_cost = -1;

            int LDL_failures = 0;

            currentIteration = 0;

            ofstream log_file("log_MOO.txt");
            Timer t("MOO");
            t.start();
            for (int level = NLevels-1; level >= 0; --level)
            {
               bool success_LDL = true;

               int const remainingIterations = maxIterations - currentIteration;
               int const n_inner_iterations = force_level_stopping ? int(0.5+double(remainingIterations)/(level+1)) : remainingIterations;

               //int const level0 = std::max(0, level-1);
               int const level0 = 0;

               for (int iter = 0; iter < n_inner_iterations; ++iter)
               {
                  
                  for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
                  double const Psi0_k  = this->eval_level_cost(level, cached_errors);
                  double const Psi0_k0 = (level > 0) ? this->eval_level_cost(level0, cached_errors) : Psi0_k;

                  if (currentIteration == 0) best_cost = Psi0_k0;

                  t.stop();
                  log_file << t.getTime() << "\t" << iter << "\t" << best_cost << endl;
                  t.start();

                  this->fillJacobians();

                  for (int obj = 0; obj < nObjs; ++obj)
                  {
                     NLSQ_Residuals& residuals = *_residuals[obj];
                     this->compute_weights(obj, level0, cached_errors[obj], cached_weights0[obj]);
                     if (level > 0) this->compute_weights(obj, level, cached_errors[obj], cached_weights1[obj]);
                  } // end for (obj)

                  double scale1 = 0;

                  if (level > 0 && !use_random_distortion)
                  {
                     this->eval_Jt_e(cached_weights0, g0);
                     this->eval_Jt_e(cached_weights1, g1);

                     double const sqrNorm_g0 = sqrNorm_L2(g0), sqrNorm_g1 = sqrNorm_L2(g1), dot_g0_g1 = innerProduct(g0, g1);
                     double const cos_01 = innerProduct(g0, g1)/std::max(1e-8, sqrt(sqrNorm_g0*sqrNorm_g1));

                     if (optimizerVerbosenessLevel >= 2) cout << "MOO_HOM_Optimizer: |g0| = " << sqrt(sqrNorm_g0) << " |g1| = " << sqrt(sqrNorm_g1) << " cos(g0, g1) = " << cos_01 << endl;

                     if (cos_01 < MOO_HOM_Optimizer_Params::cosine_threshold)
                     {
                        if (optimizerVerbosenessLevel >= 2) cout << "MOO_HOM_Optimizer: leaving level due to small cosine(g0,g1) = " << cos_01 << endl;
                        break;
                     } // end if

                     if (sqrNorm_g1 > 1e-12)
                     {
                        scale1 = this->get_scale1(beta, sqrNorm_g0, sqrNorm_g1, dot_g0_g1);
                        double const s1 = beta*scale1;

                        for (int k = 0; k < g0.size(); ++k) Jt_e[k] = -(beta1*g0[k] + s1*g1[k]);

                        if (!only_distort_gradient)
                        {
                           for (int obj = 0; obj < nObjs; ++obj)
                           {
                              Vector<double> const& weights0 = cached_weights0[obj];
                              Vector<double>      & weights1 = cached_weights1[obj];

                              int const K = weights0.size();
                              for (int k = 0; k < K; ++k) weights1[k] = beta1*weights0[k] + s1*weights1[k];
                           } // end for (obj)
                        }
                     }
                     else
                     {
                        if (optimizerVerbosenessLevel >= 2) cout << "MOO_HOM_Optimizer: leaving level due to small |g1| = " << sqrt(sqrNorm_g1) << endl;
                        break;
                     }
                  }
                  else
                  {
                     this->eval_Jt_e(cached_weights0, Jt_e);
                     scaleVectorIP(-1.0, Jt_e);
                  } // end if

                  double const norm_Jt_e = norm_L2(Jt_e);
                  if (optimizerVerbosenessLevel >= 2) cout << "MOO_HOM_Optimizer: |Jt_e| = " << norm_Jt_e << " cos(J_te, g0) = " << -innerProduct(g0, Jt_e) / norm_L2(g0) / norm_Jt_e << endl;

                  if (use_random_distortion)
                  {
                     Vector<double> v(totalParamDimension);
                     for (int j = 0; j < totalParamDimension; ++j) v[j] = dist_normal(rng);
                     double const s = 0.5 * norm_Jt_e / norm_L2(v);
                     scaleVectorIP(s, v);
                     addVectorsIP(v, Jt_e);
                     scaleVectorIP(norm_Jt_e / norm_L2(Jt_e), Jt_e);
                  } // end if

                  if (only_distort_gradient)
                     this->fillHessian(cached_weights0);
                  else
                     this->fillHessian(cached_weights1);
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

                  bool weak_decrease = false, strong_decrease = false;
                  this->copyToAllParameters(&x_saved[0]);

                  if (success_LDL)
                  {
                     double const deltaSqrLength = sqrNorm_L2(delta);
                     double const paramLength = this->getParameterLength();

                     if (optimizerVerbosenessLevel >= 2)
                        cout << "MOO_HOM_Optimizer: ||delta|| = " << sqrt(deltaSqrLength) << " ||paramLength|| = " << paramLength << endl;

                     if (level > 0)
                     {
                        double const updateThreshold = 1e-12;
                        double const updateLength = sqrt(deltaSqrLength);
                        if (updateLength < updateThreshold * (paramLength + updateThreshold))
                        {
                           if (optimizerVerbosenessLevel >= 2) cout << "MOO_HOM_Optimizer: leaving level due to small |delta| = " << updateLength << endl;
                           break;
                        }
                     } // end if

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

                  double rho = -1;
                  
                  if (success_LDL)
                  {
                     for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);

                     // Actutal costs
                     double const Psi1_k  = this->eval_level_cost(level, cached_errors);
                     double const Psi1_k0 = (level > 0) ? this->eval_level_cost(level0, cached_errors) : Psi1_k;

                     double const model_gain = 0.5*(damping_value*sqrNorm_L2(delta) + innerProduct(delta, Jt_e));
                     double const true_gain  = beta1*(Psi0_k0 - Psi1_k0) + beta*scale1*(Psi0_k - Psi1_k);
                     rho = true_gain / std::max(1e-6, model_gain);

                     weak_decrease   = (Psi1_k0 < Psi0_k0);
                     strong_decrease = (Psi1_k < Psi0_k) && weak_decrease;

                     if (weak_decrease && level > 0)
                     {
                        double const improvementThreshold = 1e-6;
                        double const relImprovement = fabs((Psi1_k0 < Psi0_k0) / Psi0_k0);
                        if (relImprovement < improvementThreshold)
                        {
                           if (optimizerVerbosenessLevel >= 2) cout << "MOO_HOM_Optimizer: leaving level due to rel. improvement = " << relImprovement << endl;
                           break;
                        }
                     } // end if

                     if (optimizerVerbosenessLevel >= 1)
                     {
                        double const current_cost = this->eval_level_cost(0, cached_errors);
                        // if (optimizerVerbosenessLevel == 1)
                        //    cout << "MOO_HOM_Optimizer: iteration " << currentIteration << " level = " << level << " new cost = " << current_cost << " best cost = " << best_cost
                        //         << " success_decrease = " << int(success_decrease) << " lambda = " << damping_value << endl;
                        // else
                           cout << "MOO_HOM_Optimizer: iteration " << currentIteration << " level = " << level << " new cost = " << current_cost << " best cost = " << best_cost
                                << " Psi(prev) = (" << Psi0_k0 << ", " << Psi0_k << ") Psi(new) = (" << Psi1_k0 << ", " << Psi1_k << ") rho = " << rho
                                << " beta = " << beta << " scale1 = " << scale1 << " lambda = " << damping_value << endl;
                        best_cost = std::min(best_cost, current_cost);
                        
                     }
                  } // end (if success_LDL)

                  //cout << "success_LDL = " << success_LDL << " strong_decrease = " << strong_decrease << " weak_decrease = " << weak_decrease << " lambda = " << damping_value << endl;

                  //if (rho > 0 && weak_decrease)
                  if (weak_decrease)
                     damping_value = std::max(damping_value_min, damping_value / 10);
                  else
                  {
                     damping_value *= 10;
                     this->copyFromAllParameters(&x_saved[0]);
                  }

                  //if (success_LDL && !weak_decrease) this->copyFromAllParameters(&x_saved[0]);
                  //if (rho <= 0 || !weak_decrease) this->copyFromAllParameters(&x_saved[0]);
               } // end for (iter)
            } // end for (level)
            log_file.close();
         } // end minimize()

   }; // end struct MOO_HOM_Optimizer

//**********************************************************************

   template <int NLevels>
   struct GN_MOO_HOM_Optimizer : public HOM_Optimizer_Base<NLevels>
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

         static double constexpr eta = 0.5, eta1 = 1.0 - eta;

         GN_MOO_HOM_Optimizer(NLSQ_ParamDesc const& paramDesc,
                              std::vector<NLSQ_CostFunction *> const& costFunctions,
                              std::vector<Robust_NLSQ_CostFunction_Base *> const& robustCostFunctions)
            : Base(paramDesc, costFunctions, robustCostFunctions)
         { }

         void minimize()
         {
            status = LEVENBERG_OPTIMIZER_TIMEOUT;

            if (_totalParamCount == 0)
            {
               // No degrees of freedom, nothing to optimize.
               if (optimizerVerbosenessLevel >= 2) cout << "GN_MOO_HOM_Optimizer: exiting since d.o.f is zero." << endl;
               status = LEVENBERG_OPTIMIZER_CONVERGED;
               return;
            }

            int const totalParamDimension = _JtJ.num_cols();

            vector<double> x_saved(totalParamDimension);

            Vector<double> Jt_e(totalParamDimension), g0(totalParamDimension), g1(totalParamDimension);
            Vector<double> delta(totalParamDimension);
            Vector<double> deltaPerm(totalParamDimension);

            int const nObjs = _robustCostFunctions.size();

            double damping_value = this->tau;

            vector<Vector<double> > cached_errors(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) cached_errors[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);

            vector<Vector<double> > cached_weights1(nObjs), cached_weights0(nObjs), combined_weights(nObjs);
            for (int obj = 0; obj < nObjs; ++obj) cached_weights0[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
            for (int obj = 0; obj < nObjs; ++obj) cached_weights1[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);
            for (int obj = 0; obj < nObjs; ++obj) combined_weights[obj].newsize(_robustCostFunctions[obj]->_nMeasurements);

            for (int obj = 0; obj < nObjs; ++obj) fillVector(1.0, _residuals[obj]->_weights);

            double best_cost = -1;

            int LDL_failures = 0;

            currentIteration = 0;

            for (int level = NLevels-1; level >= 0; --level)
            {
               int const remainingIterations = maxIterations - currentIteration;
               int const n_inner_iterations = force_level_stopping ? int(0.5+double(remainingIterations)/(level+1)) : remainingIterations;

               for (int iter = 0; iter < n_inner_iterations; ++iter)
               {
                  bool success_LDL = true;

                  for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);
                  double const Psi0_k = this->eval_level_cost(level, cached_errors);
                  double const Psi0_0 = (level > 0) ? this->eval_level_cost(0, cached_errors) : Psi0_k;

                  if (currentIteration == 0) best_cost = Psi0_0;

                  this->fillJacobians();

                  for (int obj = 0; obj < nObjs; ++obj)
                  {
                     NLSQ_Residuals& residuals = *_residuals[obj];
                     //_robustCostFunctions[obj]->cache_weight_fun(_etas2[level0], cached_errors[obj], cached_weights0[obj]);
                     //if (level > 0) _robustCostFunctions[obj]->cache_weight_fun(_etas2[level], cached_errors[obj], cached_weights1[obj]);
                     this->compute_weights(obj, 0, cached_errors[obj], cached_weights0[obj]);
                     if (level > 0) this->compute_weights(obj, level, cached_errors[obj], cached_weights1[obj]);
                  } // end for (obj)

                  if (level > 0)
                  {
                     this->eval_Jt_e(cached_weights0, g0);
                     this->eval_Jt_e(cached_weights1, g1);

                     double const norm_g0 = norm_L2(g0), norm_g1 = norm_L2(g1), dot_g0_g1 = innerProduct(g0, g1);
                     double const cos_01 = innerProduct(g0, g1)/std::max(1e-8, norm_g0*norm_g1);

                     if (optimizerVerbosenessLevel >= 2) cout << "GN_MOO_HOM_Optimizer: |g0| = " << norm_g0 << " |g1| = " << norm_g1 << " cos(g0, g1) = " << cos_01 << endl;

                     if (cos_01 < MOO_HOM_Optimizer_Params::cosine_threshold)
                     {
                        if (optimizerVerbosenessLevel >= 2) cout << "GN_MOO_HOM_Optimizer: leaving level due to small cosine(g0,g1) = " << cos_01 << endl;
                        break;
                     } // end if

                     if (MOO_HOM_Optimizer_Params::precondition_g1)
                     {
                        double const scale = norm_g0 / std::max(1e-8, norm_g1);
                        scaleVectorIP(scale, g1);
                        for (int obj = 0; obj < nObjs; ++obj) scaleVectorIP(scale, cached_weights1[obj]);

                        if (optimizerVerbosenessLevel >= 2) cout << "GN_MOO_HOM_Optimizer: |g0| = " << norm_g0 << " |g1| = " << norm_g1 << " scale = " << scale << endl;
                     }

                     for (int obj = 0; obj < nObjs; ++obj)
                     {
                        Vector<double> const& weights0 = cached_weights0[obj];
                        Vector<double> const& weights1 = cached_weights1[obj];
                        Vector<double>      & weights  = combined_weights[obj];

                        int const K = weights0.size();
                        for (int k = 0; k < K; ++k) weights[k] = eta1*weights0[k] + eta*weights1[k];
                     } // end for (obj)
                  }
                  else
                  {
                     for (int obj = 0; obj < nObjs; ++obj) copyVector(cached_weights0[obj], combined_weights[obj]);
                  }

                  this->eval_Jt_e(combined_weights, Jt_e); scaleVectorIP(-1.0, Jt_e);

                  // double const norm_Jt_e = norm_L2(Jt_e), norm_g0 = norm_L2(g0), g_ratio = norm_Jt_e / std::max(1e-8, norm_g0);
                  // if (optimizerVerbosenessLevel >= 2) cout << "GN_MOO_HOM_Optimizer: |J_te| = " << norm_Jt_e << " |g0| = " << norm_g0 << " g_ratio = " << g_ratio << endl;
                  // if (level > 0 && g_ratio < MOO_HOM_Optimizer_Params::g_ratio_threshold)
                  // {
                  //    if (optimizerVerbosenessLevel >= 2) cout << "GN_MOO_HOM_Optimizer: leaving level due to small g_ratio = " << g_ratio << endl;
                  //    break;
                  // } // end if

                  this->fillHessian(combined_weights);
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

                  bool weak_decrease = false, strong_decrease = false;
                  this->copyToAllParameters(&x_saved[0]);

                  if (success_LDL)
                  {
                     double const deltaSqrLength = sqrNorm_L2(delta);
                     double const paramLength = this->getParameterLength();

                     if (optimizerVerbosenessLevel >= 2)
                        cout << "GN_MOO_HOM_Optimizer: ||delta|| = " << sqrt(deltaSqrLength) << " ||paramLength|| = " << paramLength << endl;

                     if (level > 0)
                     {
                        double const updateThreshold = 1e-12;
                        double const updateLength = sqrt(deltaSqrLength);
                        if (updateLength < updateThreshold * (paramLength + updateThreshold))
                        {
                           if (optimizerVerbosenessLevel >= 2) cout << "GN_MOO_HOM_Optimizer: leaving level due to small |delta| = " << updateLength << endl;
                           break;
                        }
                     } // end if

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

                  double rho = -1;
                  
                  if (success_LDL)
                  {
                     for (int obj = 0; obj < nObjs; ++obj) _robustCostFunctions[obj]->cache_residuals(*_residuals[obj], cached_errors[obj]);

                     // Actutal costs
                     double const Psi1_k = this->eval_level_cost(level, cached_errors);
                     double const Psi1_0 = (level > 0) ? this->eval_level_cost(0, cached_errors) : Psi1_k;

                     double const model_gain = 0.5*(damping_value*sqrNorm_L2(delta) + innerProduct(delta, Jt_e));
                     double true_gain  = Psi0_0 - Psi1_0;
                     if (level > 0)
                     {
                        double const scale = MOO_HOM_Optimizer_Params::precondition_g1 ? norm_L2(g0) / std::max(1e-8, norm_L2(g1)) : 1.0;
                        true_gain = eta1*(Psi0_0 - Psi1_0) + scale*eta*(Psi0_k - Psi1_k);
                     }

                     rho = true_gain / std::max(1e-8, model_gain);

                     if (optimizerVerbosenessLevel >= 2) cout << "GN_MOO_HOM_Optimizer: model gain = " << model_gain << " true gain = " << true_gain << " rho = " << rho << endl;

                     weak_decrease   = (Psi1_0 < Psi0_0);
                     strong_decrease = (Psi1_k < Psi0_k) && weak_decrease;
                     //strong_decrease = (rho > 0);

                     if (weak_decrease && level > 0)
                     {
                        double const improvementThreshold = 1e-6;
                        double const relImprovement = fabs((Psi1_0 < Psi0_0) / Psi0_0);
                        if (relImprovement < improvementThreshold)
                        {
                           if (optimizerVerbosenessLevel >= 2) cout << "GN_MOO_HOM_Optimizer: leaving level due to rel. improvement = " << relImprovement << endl;
                           break;
                        }
                     } // end if

                     if (optimizerVerbosenessLevel >= 1)
                     {
                        double const current_cost = Psi1_0; //this->eval_level_cost(0, cached_errors);
                        // if (optimizerVerbosenessLevel == 1)
                        //    cout << "MOO_HOM_Optimizer: iteration " << currentIteration << " level = " << level << " new cost = " << current_cost << " best cost = " << best_cost
                        //         << " success_decrease = " << int(success_decrease) << " lambda = " << damping_value << endl;
                        // else
                        if (0 || strong_decrease)
                           cout << "GN_MOO_HOM_Optimizer: iteration " << setw(3) << currentIteration << " level = " << level
                                << " best cost = " << setw(12) << best_cost
                                << " Psi(prev,new) = (" << setw(12) << Psi0_0 << "  " << setw(12) << Psi1_0
                                << ") Psi_k(prev,new) = (" << setw(12) << Psi0_k << "  " << setw(12) << Psi1_k << ")  strong = " << strong_decrease
                                << " rho = " << rho << " eta = " << eta << " lambda = " << damping_value << endl;
                        best_cost = std::min(best_cost, current_cost);
                     }

                  } // end (if success_LDL)

                  //cout << "success_LDL = " << success_LDL << " strong_decrease = " << strong_decrease << " weak_decrease = " << weak_decrease << " lambda = " << damping_value << endl;

#if 0
                  if (strong_decrease)
                     damping_value = std::max(damping_value_min, damping_value / 10);
                  else
                  {
                     damping_value *= 10;
                     this->copyFromAllParameters(&x_saved[0]);
                  }
#else
                  if (rho > 0)
                     damping_value = std::max(damping_value_min, damping_value / 10);
                  else
                  {
                     damping_value *= 10;
                  }

                  if (!weak_decrease) this->copyFromAllParameters(&x_saved[0]);
                  if (rho > 0 && !strong_decrease)
                  {
                     if (optimizerVerbosenessLevel >= 2) cout << "GN_MOO_HOM_Optimizer: leaving level since rho>0 and no strong decrease" << endl;
                     break;
                  }
#endif

                  //if (success_LDL && !weak_decrease) this->copyFromAllParameters(&x_saved[0]);
                  //if (rho <= 0 || !weak_decrease) this->copyFromAllParameters(&x_saved[0]);
               } // end for (iter)
            } // end for (level)
         } // end minimize()

   }; // end struct GN_MOO_HOM_Optimizer

//**********************************************************************
} // end namespace Robust_LSQ

#endif
