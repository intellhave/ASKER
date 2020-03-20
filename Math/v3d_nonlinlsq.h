// -*- C++ -*-
#ifndef V3D_NONLINEAR_LSQR_H
#define V3D_NONLINEAR_LSQR_H

#include "Math/v3d_linear.h"
#include "Math/v3d_linear_tnt.h"
#include "Math/v3d_mathutilities.h"
#include "Math/v3d_optimization.h"

#include <vector>
#include <iostream>

#define V3DLIB_ENABLE_SUITESPARSE

namespace V3D
{

#if defined(V3DLIB_ENABLE_SUITESPARSE)

#define NLSQ_MAX_PARAM_TYPES 32

   struct NLSQ_ParamDesc
   {
         int nParamTypes;                     //!< How many different kinds of parameters exist (2 for std. BA, cameras and 3D points)
         int dimension[NLSQ_MAX_PARAM_TYPES]; //!< What is the dimension of each parameter kind
         int count[NLSQ_MAX_PARAM_TYPES];     //!< How many unknowns are there per parameter type
   }; // end struct NLSQ_ParamDesc
     
   //! This structure holds the residuals, weights and Jacobian for a filter method with two cost functions
   struct Filter_Residuals
   {
         Filter_Residuals(vector<int> const& usedParamTypes, int const nMeasurements,
                        int const measurementDimension, NLSQ_ParamDesc const& paramDesc)
            : _residualsF(nMeasurements, measurementDimension),
              _residualsH(nMeasurements, measurementDimension),
              _weights(nMeasurements), _h_weights(nMeasurements),
              _JFs(usedParamTypes.size()),
              _JHs(usedParamTypes.size())
         {
            for (int k = 0; k < usedParamTypes.size(); ++k)
            {
               int const paramType = usedParamTypes[k];
               int const paramDimension = paramDesc.dimension[paramType];
               _JFs[k] = new MatrixArray<double>(nMeasurements, measurementDimension, paramDimension);
               _JHs[k] = new MatrixArray<double>(nMeasurements, measurementDimension, paramDimension);
            } // end for (k)
         } // end NLSQ_Residuals()

         Filter_Residuals(vector<int> const& usedParamTypes, int const nMeasurements,
                        int const fResidualDimension, int const hResidualDimension, NLSQ_ParamDesc const& paramDesc)
            : _residualsF(nMeasurements, fResidualDimension),
              _residualsH(nMeasurements, hResidualDimension),
              _weights(nMeasurements),
              _JFs(usedParamTypes.size()),
              _JHs(usedParamTypes.size())
         {
            for (int k = 0; k < usedParamTypes.size(); ++k)
            {
               int const paramType = usedParamTypes[k];
               int const paramDimension = paramDesc.dimension[paramType];
               _JFs[k] = new MatrixArray<double>(nMeasurements, fResidualDimension, paramDimension);
               _JHs[k] = new MatrixArray<double>(nMeasurements, hResidualDimension, paramDimension);
            } // end for (k)
         } // end NLSQ_Residuals()

         ~Filter_Residuals()
         {
            for (int i = 0; i < _JFs.size(); ++i) delete _JFs[i];
            for (int i = 0; i < _JHs.size(); ++i) delete _JHs[i];
         }

         VectorArray<double> _residualsF;
         VectorArray<double> _residualsH;

         Vector<double>      _weights;
         Vector<double>      _h_weights;

         vector<MatrixArray<double> * > _JHs;
         vector<MatrixArray<double> * > _JFs;
   }; // end struct NLSQ_Residuals



   //! This structure holds the residuals, weights and Jacobian for a particular cost function.
   struct NLSQ_Residuals
   {
         NLSQ_Residuals(vector<int> const& usedParamTypes, int const nMeasurements,
                        int const measurementDimension, NLSQ_ParamDesc const& paramDesc)
            : _residuals(nMeasurements, measurementDimension),
              _weights(nMeasurements),
              _Js(usedParamTypes.size())
         {
            for (int k = 0; k < usedParamTypes.size(); ++k)
            {
               int const paramType = usedParamTypes[k];
               int const paramDimension = paramDesc.dimension[paramType];
               _Js[k] = new MatrixArray<double>(nMeasurements, measurementDimension, paramDimension);
            } // end for (k)
         } // end NLSQ_Residuals()

         ~NLSQ_Residuals()
         {
            for (int i = 0; i < _Js.size(); ++i) delete _Js[i];
         }

         VectorArray<double> _residuals;
         Vector<double>      _weights;

         vector<MatrixArray<double> * > _Js;
   }; // end struct NLSQ_Residuals

   struct NLSQ_LM_BlockHessian
   {
         NLSQ_LM_BlockHessian()
         {
            for (int t1 = 0; t1 < NLSQ_MAX_PARAM_TYPES; ++t1)
               for (int t2 = 0; t2 < NLSQ_MAX_PARAM_TYPES; ++t2)
                  Hs[t1][t2] = 0;
         }

         void allocateMatrices(NLSQ_ParamDesc const& paramDesc)
         {
            int const nParamTypes = paramDesc.nParamTypes;

            for (int t1 = 0; t1 < nParamTypes; ++t1)
               for (int t2 = 0; t2 < nParamTypes; ++t2)
               {
                  int const nz = nonzeroPairs[t1][t2].size();
                  if (nz > 0)
                  {
                     int const rows = paramDesc.dimension[t1];
                     int const cols = paramDesc.dimension[t2];
                     Hs[t1][t2] = new MatrixArray<double>(nz, rows, cols);
                  }
                  else
                     Hs[t1][t2] = 0;
               } // end for (t2)
         } // end allocateMatrices()

         void deallocateMatrices()
         {
            for (int t1 = 0; t1 < NLSQ_MAX_PARAM_TYPES; ++t1)
               for (int t2 = 0; t2 < NLSQ_MAX_PARAM_TYPES; ++t2)
                  if (Hs[t1][t2]) delete Hs[t1][t2];
         }

         void setZero()
         {
            for (int t1 = 0; t1 < NLSQ_MAX_PARAM_TYPES; ++t1)
               for (int t2 = 0; t2 < NLSQ_MAX_PARAM_TYPES; ++t2)
                  if (Hs[t1][t2])
                  {
                     MatrixArray<double>& H = *Hs[t1][t2];
                     for (int n = 0; n < H.count(); ++n) makeZeroMatrix(H[n]);
                  }
         } // end setZero()

         vector<pair<int, int> > nonzeroPairs[NLSQ_MAX_PARAM_TYPES][NLSQ_MAX_PARAM_TYPES];
         MatrixArray<double> *   Hs[NLSQ_MAX_PARAM_TYPES][NLSQ_MAX_PARAM_TYPES];
   }; // end struct NLSQ_LM_BlockHessian

   struct NLSQ_CostFunction
   {
         NLSQ_CostFunction(std::vector<int> const& usedParamTypes,
                           Matrix<int> const& correspondingParams,
                           int const measurementDimension)
            : _usedParamTypes(usedParamTypes),
              _nMeasurements(correspondingParams.num_rows()),
              _measurementDimension(measurementDimension),
              _correspondingParams(correspondingParams)
         {
            assert(usedParamTypes.size() == correspondingParams.num_cols());
         }

         virtual void  preIterationCallback() { }
         virtual void  initializeResiduals()  { }
         virtual void  evalResidual(int const k, Vector<double>& residual) const = 0;
         virtual void  evalLinearizedResidual(int const k, Vector2d& residual) {}
         virtual void  evalLinearizedResidual(int const k, Matrix<double>& J, Vector2d& residual) {}

         //check for inlier
         virtual bool isInlier(int const i) {}

         //This is added for auxiliary variables
         virtual double evalFiCost(int const i) const {return 0.0;}
         virtual double evalFiWeight(int const i) const {return 1.0;}
         virtual void evalFiResidual(int const i, Vector<double>& e) {}
         
         virtual double evalHiCost(int const i) const {return 0.0;}
         virtual void evalHiResidual(int const i, Vector<double>& e) {}
         virtual double evalHiWeight(int const i) const {return 1.0;}
         virtual double evalRiCost (int const i) const {return 0.0;}
         virtual void setPenaltyParam(double const param) {};

         //Initialize auxiliary variables
         virtual void initAuxVars() {}

         virtual void evalOrgResidual(int const k, Vector<double>& e) const {}
         
         virtual void fillLinearConstraints(int const k, 
                                           Vector<double>& constr1, 
                                           Vector<double>& constr2,
                                           double& b1, 
                                           double& b2) {}
         virtual void   initializeWeights(VectorArray<double> const& residuals) { }
         virtual double getWeight(Vector<double> const& residual, int const k) const { return 1.0; }


         virtual double evalCost(VectorArray<double> const& residuals, Vector<double> const& weights) const
         {
            double res = 0.0;
            for (int k = 0; k < _nMeasurements; ++k) res += sqr(weights[k]) * sqrNorm_L2(residuals[k]);
            return res;
         } // end evalCost()


         virtual void initializeJacobian() { }
         virtual void fillJacobian(int const whichParam, int const paramIx, int const k, Matrix<double>& J) const = 0;

         // //For filter methods
         virtual void fillJacobian_F(int const whichParam, int const paramIx, int const k, Matrix<double>& J) {}
         virtual void fillJacobian_H(int const whichParam, int const paramIx, int const k, Matrix<double>& J) {}

         virtual void combineJacobian(Matrix<double>const& J1, Matrix<double>const& J2, 
                                      double const w1, double const w2,
                                      Matrix<double>& J) {}

         //Cache residuals for filter methods
         virtual void cache_filter_residuals(Filter_Residuals &residuals) {}


         virtual void multiply_JtJ(double const lambda, int const k, Vector<double> const& residual, Matrix<double> const& J1, Matrix<double> const& J2,
                                   Matrix<double>& J1tJ2)
         {
            multiply_At_B(J1, J2, J1tJ2);
         }

         virtual void multiply_Jt_e(double const lambda, int const paramType, int const k, Matrix<double> const& Jk, Vector<double> const& residual,
                                    Vector<double>& Jkt_e)
         {
            multiply_At_v(Jk, residual, Jkt_e);
         }

         Matrix<int> const& correspondingParams() const { return _correspondingParams; }

         // Return true if multiply_JtJ() or multiply_Jt_e depend on the LM parameter lambda.
         // If this is the case, solely modifying the main diagonal of JtJ is not enough.
         virtual bool forbid_derivative_caching() const { return false; }

         //protected:
         void evalAllResiduals(VectorArray<double>& residuals) const
         {
            for (int k = 0; k < _nMeasurements; ++k) 
               this->evalResidual(k, residuals[k]);
         } // end evalAllResiduals()

         void fillAllWeights(VectorArray<double> const& residuals, Vector<double>& w) const
         {
            for (int k = 0; k < _nMeasurements; ++k) w[k] = this->getWeight(residuals[k], k);
         }

         void fillAllJacobians(Vector<double> const& weights, vector<MatrixArray<double> * >& Js) const;

         //This is for Filter
         virtual void fillAllJacobians_F(Vector<double> const& weights, vector<MatrixArray<double> * >& Js) {};
         virtual void fillAllJacobians_F_smooth(Vector<double> const& weights, vector<MatrixArray<double> * >& Js) {};
         virtual void fillAllJacobians_H(Vector<double> const& weights, vector<MatrixArray<double> * >& Js) {};
         virtual void combineAllJacobians(vector<MatrixArray<double> * >const & Js1,
                                          vector<MatrixArray<double> * >const & Js2,
                                          double const w1, double const w2,
                                          vector<MatrixArray<double> * >& Js ) {};




         std::vector<int> const& _usedParamTypes;
         int              const  _nMeasurements;
         int              const  _measurementDimension;
         Matrix<int>      const& _correspondingParams;

         friend struct NLSQ_LM_Optimizer;
   }; // end struct NLSQ_CostFunction

   struct NLSQ_LM_Optimizer : public LevenbergOptimizerCommon
   {
         NLSQ_LM_Optimizer(NLSQ_ParamDesc const& paramDesc,
                           std::vector<NLSQ_CostFunction *> const& costFunctions,
                           ostream &log_ostream = std::cout,
                           bool const runSymamd = true);

         ~NLSQ_LM_Optimizer();

         void minimize();

         virtual bool allowStoppingCriteria() const { return true; }

         virtual void getLambda(int const paramType, int const paramIx, int const paramDim, double * lambda_dst) const { std::fill(lambda_dst, lambda_dst + paramDim, lambda); }

         virtual double getParameterLength() const = 0;

         virtual void preIterationCallback() { }
         virtual void updateParameters(int const paramType, VectorArrayAdapter<double> const& delta) = 0;
         virtual void updateStepParams(Vector<double> const& tk) {} 
         virtual void finishUpdateParameters() { }
         virtual void saveAllParameters() = 0;
         virtual void restoreAllParameters() = 0;

         double evalCurrentObjective()
         {
            int const nObjs = _costFunctions.size();
            double err = 0.0;
            for (int obj = 0; obj < nObjs; ++obj)
            {
               NLSQ_CostFunction& costFun = *_costFunctions[obj];
               NLSQ_Residuals& residuals = *_residuals[obj];

               costFun.preIterationCallback();
               costFun.initializeResiduals();
               costFun.evalAllResiduals(residuals._residuals);
               costFun.initializeWeights(residuals._residuals);
               costFun.fillAllWeights(residuals._residuals, residuals._weights);

               err += costFun.evalCost(residuals._residuals, residuals._weights);
            } // end for (obj)
            return err;
         } // end evalCurrentObjective()

         CCS_Matrix<double> const& JtJ() const { return _JtJ; }

         vector<NLSQ_Residuals *> const& residuals() const { return _residuals; }

      protected:
         void setupSparseJtJ(bool const runSymamd);
         void fillJtJ();
         void fillJacobians();
         void fillHessian();
         void evalJt_e(Vector<double>& Jt_e);

         //! Map (paramType,ix) pairs to a continuous range of global ids.
         int getParamId(int const paramType, int const paramIx) const { return _paramTypeStartID[paramType] + paramIx; }

         NLSQ_ParamDesc                   const& _paramDesc;
         std::vector<NLSQ_CostFunction *> const& _costFunctions;

         //! Mapping of parameter types to their first id in the continuous range.
         int _paramTypeStartID[NLSQ_MAX_PARAM_TYPES + 1];
         int _totalParamCount;
         vector<pair<int, int> > _paramIdInverseMap; //!< the reverse mapping from global ids to (paramType,id) pairs.

         vector<NLSQ_Residuals *>   _residuals;      // one per cost function

         //!< _hessianIndices establishes the link between the local hessian in a cost function to the global non-zero block.
         vector<MatrixArray<int> *> _hessianIndices; // one per cost function
         NLSQ_LM_BlockHessian       _hessian;

         // Used for sparse Cholesky
         std::vector<int> _JtJ_Lp, _JtJ_Parent, _JtJ_Lnz;
         std::vector<int> _perm_JtJ, _invPerm_JtJ;

         int _paramTypeRowStart[NLSQ_MAX_PARAM_TYPES + 1];
         CCS_Matrix<double> _JtJ;

         ostream &_log_ostream;
   }; // end struct NLSQ_LM_Optimizer

#endif // defined(V3DLIB_ENABLE_SUITESPARSE)

} // end namespace V3D

#endif
