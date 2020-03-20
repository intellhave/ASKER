#ifndef ROBUST_FILTER_H
#define ROBUST_FILTER_H

#include "robust_lsq_common.h"
#include "Base/v3d_timer.h"
#include <fstream>

namespace Robust_LSQ
{
    struct Robust_Filter_MOO_Optimizer : public Robust_LSQ_Customized_Weights_Optimizer_Base
    {
        static double constexpr filter_alpha = 0.0001;
        static double constexpr penalty_damping = 1e-1;    
        static double constexpr h_damping = 2.0;

        //This is for line search during exploration (Optional)
        static double constexpr p_step_size_min = 0.99;
        static double constexpr p_step_size_max = 1.0;
        static double constexpr p_step_size_scale = 1.1;

        //Line Search in Restoration Step
        static double constexpr h_step_size_min = -0.5;
        static double constexpr h_step_size_max = 0.5;
        static double constexpr h_step_size_scale = 0.05;

        //Threshold for early stopping when h is small (optional)
        static double constexpr h_epsilon = 1e-20;

        typedef Robust_LSQ_Customized_Weights_Optimizer_Base Base;

        struct Filter
        {
        public:
            Filter(double alpha = filter_alpha) : _alpha(alpha)
            {
            }

            //Check if the pair (f, h) belongs to the forbidden region
            bool isDominated(double f, double h)
            {
                // return false;
                for (int i = 0; i < _elements.size(); ++i)                        
                {
                    std::pair<double, double> e = _elements[i];
                    if (f >= e.first && h >= e.second)
                        return true;
                }
                return false;
            }

            //Add one more element to the filter
            void addElement(double f, double h, double cost)
            {
                double alpha = _alpha;

                double fn = f - alpha * h;
                double hn = h - alpha * h;
               _elements.push_back(make_pair(fn, hn));
            }

           //Add one more element to the filter
            void addElement(double f, double h)
            {
                _elements.push_back(make_pair(f - _alpha * h, (1 - _alpha) * h));
            }

            //Remove last element
            void removeLastElement()
            {
                _elements.erase(_elements.end() - 1);
            }

            //Return the current filter size
            int getSize() { return _elements.size(); }

        private:
            double _alpha;
            vector<pair<double, double>> _elements;
        }; //end struct filter

        Robust_Filter_MOO_Optimizer(NLSQ_ParamDesc const &paramDesc,
                                    std::vector<NLSQ_CostFunction *> const &costFunctions,
                                    std::vector<Robust_NLSQ_CostFunction_Base *> const &robustCostFunctions,
                                    vector<Vector2d> &Sigma,
                                    std::ofstream &log_file)
            : Base(paramDesc, costFunctions, robustCostFunctions),
            _log_file(log_file),  _Sigma(Sigma)
        {            
            //Allocate the filter
            _filter = new Filter();
        }

        ~Robust_Filter_MOO_Optimizer()
        {
            delete _filter;
        }
        
        double _penaltyLambda = penalty_damping;
        double _hLambda = h_damping;
        double _lambda;

        double _pMinStepSize = p_step_size_min;
        double _pMaxStepSize = p_step_size_max;       
        double _pStepSizeScale = p_step_size_scale;

        double _hMinStepSize = h_step_size_min;
        double _hMaxStepSize = h_step_size_max;       
        double _hStepSizeScale = h_step_size_scale;

        double _mu, _mu1;

        enum FilterStep
        {
            PENALTY,
            EXPLORATION,
            IRLS
        };

        double eval_current_cost() const
        {
            int const nObjs = _robustCostFunctions.size();
            double cost = 0;
            for (int obj = 0; obj < nObjs-1; ++obj)
            {
                NLSQ_CostFunction &costFun = *_costFunctions[obj];
                Robust_NLSQ_CostFunction_Base &robustCostFun = *_robustCostFunctions[obj];

                Vector<double> org_errors(costFun._nMeasurements);
                for (int k = 0; k < costFun._nMeasurements; ++k)
                {
                    Vector<double> e(2);
                    costFun.evalOrgResidual(k, e);
                    org_errors[k] = sqrNorm_L2(e);                                        
                }
                cost += robustCostFun.eval_target_cost(1.0, org_errors);
            }
            return cost;
        }

        void evalFH(vector<Vector<double>> const& cached_errors, double &f, double &h)
        {
            int const nObjs = _robustCostFunctions.size();
            f = 0; h = 0;            
            for (int obj = 0; obj < nObjs - 1; ++obj)
            {                                
                NLSQ_CostFunction &costFun = *_costFunctions[obj];    
                Robust_NLSQ_CostFunction_Base &robustCost = *_robustCostFunctions[obj];
                f += robustCost.eval_target_cost(1.0, cached_errors[obj]);                
            }            

            for (int k = 0; k < _Sigma.size(); ++k)
                h += _Sigma[k][0]*_Sigma[k][0];
        }

        //Evaluation of obj gradient only involves hi
        void evalH_Jt_e(vector< Vector<double> > const& cached_weights, Vector<double> &g)
        {
            makeZeroVector(g);
            int const nObjs = _robustCostFunctions.size();
            int obj = nObjs - 1;
            {
                NLSQ_CostFunction& costFun = *_costFunctions[obj];
                NLSQ_Residuals& residuals = *_residuals[obj];

                // Vector<double> const& weights = given_weights[obj];

                int const nParamTypes = costFun._usedParamTypes.size();
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
                        
                        for (int l = 0; l < paramDim; ++l) 
                            g[dstRow + l] += Jkt_e[l];
                    } // end for k 
                } //end for i
            }       
        }

        void evalF_Jt_e(vector<Vector<double> > const& given_weights, Vector<double> &g)
        {
            makeZeroVector(g);
            int const nObjs = _costFunctions.size();

            //Note here only compute for the first (n-1) functions, since the last one is for constraints
            for (int obj = 0; obj < nObjs-1; ++obj) 
            {
                NLSQ_CostFunction& costFun = *_costFunctions[obj];
                NLSQ_Residuals& residuals = *_residuals[obj];
                Robust_NLSQ_CostFunction_Base& robustCostFun = *_robustCostFunctions[obj];
                
                Vector<double> const& weights = given_weights[obj];

                int const nParamTypes = costFun._usedParamTypes.size();
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

                        for (int l = 0; l < paramDim; ++l) 
                            g[dstRow + l] -= _mu*Jkt_e[l];
                        
                        //Add gradient for constraint violation
                        if (paramType==2)
                        {
                            NLSQ_Residuals& h_residuals = *_residuals[nObjs - 1];
                            g[dstRow] -= _mu1 * h_residuals._residuals[k][0];
                        }
                    } // end for k 
                } //end for i
            }
        }
       
        void prepareH(vector< Vector<double> > const& cached_weights, Vector<double> &Jt_e)
        {

            makeZeroVector(Jt_e);
            int const nObjs = _costFunctions.size();            
            int const totalParamDimension = _JtJ.num_cols();
            Vector<double> gH(totalParamDimension);
            evalH_Jt_e(cached_weights, gH);

            double norm_gH = norm_L2(gH);
            // cout << " Norm GH = " << norm_gH << endl;

            double scale = 1.0;
            for (int i = 0; i < gH.size(); ++i)
            {
                Jt_e[i] = -scale*gH[i];          
            }        
        }

        
        void preparePenalty(vector< Vector<double> > const& cached_weights, Vector<double> &Jt_e)
        {            
            int const totalParamDimension = _JtJ.num_cols();
            Vector<double> gF(totalParamDimension), gH(totalParamDimension);                       
            _mu = 0.7; _mu1 = 0.3;            

            //Compute J'r
            evalF_Jt_e(cached_weights, Jt_e);

       }//end preparePenalty

        void fillHessian(FilterStep &stepType, vector<Vector<double>> const& cached_weights)
        {
                // Set Hessian to zero
                _hessian.setZero();

                int const nObjs = _costFunctions.size();
                for (int obj = 0; obj < nObjs-1; ++obj)
                {
                    NLSQ_CostFunction& costFun = *_costFunctions[obj];
                    
                    int const nParamTypes = costFun._usedParamTypes.size();
                    int const nMeasurements = costFun._nMeasurements, residualDim = costFun._measurementDimension;

                    Matrix<double> H(residualDim, residualDim);
                    MatrixArray<int> const& index = *_hessianIndices[obj];
                    NLSQ_Residuals   const& residuals = *_residuals[obj];                    
                    //    Vector<double> const& weights = given_weights[obj];
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
                                
                                if (stepType==PENALTY)
                                {                                
                                    multiply_At_B(Js1[k], Js2[k], J1tJ2); 
                                    double w =  cached_weights[obj][k];
                                    w *= (obj < nObjs-1)?_mu:_mu1; 

                                    scaleMatrixIP(w, J1tJ2);
                                    addMatricesIP(J1tJ2, Hs[n]);

                                    if ((t1 == 2) && (t2==2))
                                    {
                                        Hs[n][0][0] += _mu1 + _hLambda;
                                    }
                                    
                                }
                                else if (stepType == EXPLORATION)
                                {                                    
                                    if (obj == nObjs-1)
                                    {
                                        multiply_At_B(Js1[k], Js2[k], J1tJ2);                             
                                        addMatricesIP(J1tJ2, Hs[n]);
                                    }
                                }
                                else
                                {
                                    assert(false);
                                }
                            } // end for (k)
                        } // end for (i2)
                    } // end for (i1)
                } // end for (obj)
            } // end fillHessian()

        void minimize()
        {
            status = LEVENBERG_OPTIMIZER_TIMEOUT;

            assert(_totalParamCount > 0);
            int const totalParamDimension = _JtJ.num_cols();
            vector<double> x_saved(totalParamDimension);
            vector<double> best_x(totalParamDimension);

            Vector<double> Jt_e(totalParamDimension);
            Vector<double> gF(totalParamDimension), gH(totalParamDimension);
            Vector<double> delta(totalParamDimension);
            Vector<double> deltaPerm(totalParamDimension);
            double bestCost = 1e20;

            int const nObjs = _costFunctions.size();
            vector<Vector<double>> cached_errors(nObjs);
            vector< Vector<double>> cached_weights(nObjs);

            for (int obj = 0; obj < nObjs; ++obj)
            {
                cached_errors[obj].newsize(_costFunctions[obj]->_nMeasurements);
                cached_weights[obj].newsize(_costFunctions[obj]->_nMeasurements);
            }
            
            for (int obj = 0; obj < nObjs; ++obj) fillVector(1.0, _residuals[obj]->_weights);
            
            double f, h, f1, h1;    

            FilterStep stepType = PENALTY;
            int acceptCount = 0;
            this->copyToAllParameters(&x_saved[0]);
            double best_cost = 1e40;
            Timer t("Bundle");
            t.start();
            int iter = 0;   
            double totalTime = 0;
            while (iter < maxIterations)
            {
                this->copyToAllParameters(&x_saved[0]);
                // cost = this->eval_current_cost(cached_errors);
                if (iter == 0)
                    for (int obj = 0; obj < nObjs-1; ++obj)
                    {
                        NLSQ_Residuals &residuals = * _residuals[obj];
                        NLSQ_CostFunction &costFun = *_costFunctions[obj];                    
                        Robust_NLSQ_CostFunction_Base &robustCost = *_robustCostFunctions[obj];                    

                        //Cache Residuals and Weight Functions                    
                        robustCost.cache_residuals(residuals, cached_errors[obj]);
                        robustCost.cache_weight_fun(1.0, cached_errors[obj], cached_weights[obj]);       
                    }
            
                //Eval cost and inliers, then save the best results
                double initial_cost = eval_current_cost();  
                if (initial_cost < best_cost)
                {
                    best_cost = initial_cost;
                    copyToAllParameters(&best_x[0]);
                }

                //Evaluate objective and constraint violations
                this->evalFH(cached_errors, f, h);

                if (!_filter->isDominated(f,h) && stepType == PENALTY)
                    _filter->addElement(f, h);

                t.stop(); 
                cout << " Time = " << t.getTime() << "  iter = " << iter << " cost = " << initial_cost<< endl;

                if (stepType == PENALTY)    
                    _log_file << t.getTime() << "\t" << iter << "\t" << best_cost << "\t" << f <<"\t" << h << "\t" << endl;

                t.start();
                bool success_LDL = true;

                //Fill Jacobians
                this->fillJacobians();
                
                if (stepType == PENALTY)
                {
                    this->preparePenalty(cached_weights, Jt_e);
                    _lambda = _penaltyLambda;
                }
                else if (stepType == EXPLORATION)
                {
                    this->prepareH(cached_weights, Jt_e);
                    _lambda = h_damping;
                    // this->prepareGradDotProd(gFH);
                }
                else {assert(false);}

                //Fill Hessian
                this->fillHessian(stepType, cached_weights);

                // Augment the diagonals
                for (int paramType = 0; paramType < _paramDesc.nParamTypes; ++paramType)
                {
                    MatrixArray<double> &Hs = *_hessian.Hs[paramType][paramType];
                    vector<pair<int, int>> const &nzPairs = _hessian.nonzeroPairs[paramType][paramType];
                    int const dim = Hs.num_cols(), count = Hs.count();

                    // Only augment those with the same parameter id
                    for (int n = 0; n < count; ++n)
                    {
                        if (nzPairs[n].first != nzPairs[n].second)
                            continue;
                        for (int l = 0; l < dim; ++l)
                            Hs[n][l][l] += _lambda;
                    }
                } // end for (paramType)
                this->fillJtJ();

                //Execute Newton step
                success_LDL = this->solve_JtJ(Jt_e, deltaPerm, delta);
                ++currentIteration;
    
                if (success_LDL)
                {
                    if (stepType == PENALTY)
                    {
                        // Conduct line search to search for a non-dominated point. This is just optinal!                    
                        double alpha = _pMaxStepSize;                    
                        bool dominated = true;
                        this->copyToAllParameters(&x_saved[0]);
                        while (alpha > _pMinStepSize)
                        {
                            scaleVectorIP(alpha, delta);                        
                            for (int paramType = 0; paramType < _paramDesc.nParamTypes; ++paramType)
                            {
                                int const paramDim = _paramDesc.dimension[paramType];
                                int const count = _paramDesc.count[paramType];
                                int const rowStart = _paramTypeRowStart[paramType];

                                VectorArrayAdapter<double> deltaParam(count, paramDim, &delta[0] + rowStart);
                                this->updateParameters(paramType, deltaParam);
                            }

                            this->finishUpdateParameters();
                            //Compute cost
                            for (int obj = 0; obj < nObjs; ++obj)
                            {
                                NLSQ_Residuals &residuals = * _residuals[obj];
                                NLSQ_CostFunction &costFun = *_costFunctions[obj];                    
                                Robust_NLSQ_CostFunction_Base &robustCost = *_robustCostFunctions[obj];                                
                                robustCost.cache_residuals(*_residuals[obj], cached_errors[obj]);
                                // robustCost.cache_weight_fun(1.0, cached_errors[obj], cached_weights[obj]);
                                robustCost.cache_weight_fun(1.0, cached_errors[obj], cached_weights[obj]);       
                            }
         
                            //Compute current cost
                            double e_cost = eval_current_cost();
                            this->evalFH(cached_errors,f1, h1);

                            dominated = _filter->isDominated(f1, h1);                    
                            alpha /= _pStepSizeScale;
                            if (!dominated)
                                break;
                            this->copyFromAllParameters(&x_saved[0]);                            
                        }

                        if (!dominated)
                        {
                            if (f1 < f)
                                _filter->removeLastElement();   
                            f = f1; h = h1;

                            if (h < h_epsilon)  break;

                            _penaltyLambda = max(1e-5, _penaltyLambda*0.1);
                            _hLambda *= 0.9;
                        }
                        else
                        {                        
                            cout << "==============ENTER RESTORATION STEP========================\n";
                            this->copyFromAllParameters(&x_saved[0]);                           
                            stepType = EXPLORATION;
                            iter--;
                            //Reset damping parameters
                            _hLambda = h_damping;
                            _penaltyLambda = penalty_damping;
                        }
                        
                    }
                    else //Restoration --> Line Search
                    {
                        double alpha = _hMinStepSize;                    
                        bool dominated = true;
                    
                        double sumSigma = 0;
                        for (int i = 0; i < _Sigma.size(); ++i)
                            sumSigma += 0.5*_Sigma[i][0]*_Sigma[i][0];
                        
                        // this->copyToAllParameters(&x_saved[0]);
                        double maxAngle = -1e20;
                        vector<double> best_x(totalParamDimension);
                        Vector<double> exp_direction(totalParamDimension);

                        while (alpha < _hMaxStepSize )
                        {
                            scaleVector(alpha, delta, exp_direction);                        
                            for (int paramType = 0; paramType < _paramDesc.nParamTypes; ++paramType)
                            {
                                int const paramDim = _paramDesc.dimension[paramType];
                                int const count = _paramDesc.count[paramType];
                                int const rowStart = _paramTypeRowStart[paramType];

                                VectorArrayAdapter<double> deltaParam(count, paramDim, &exp_direction[0] + rowStart);
                                this->updateParameters(paramType, deltaParam);
                            }
                            this->finishUpdateParameters();
                            // Re-compute and cache residuals
                            for (int obj = 0; obj < nObjs; ++obj)
                            {
                                NLSQ_Residuals &residuals = * _residuals[obj];
                                NLSQ_CostFunction &costFun = *_costFunctions[obj];                    
                                Robust_NLSQ_CostFunction_Base &robustCost = *_robustCostFunctions[obj];                                
                                robustCost.cache_residuals(*_residuals[obj], cached_errors[obj]);
                                robustCost.cache_IRLS_weights(1.0, cached_errors[obj], cached_weights[obj]);
                            }

                            //Compute current cost
                            double e_cost = eval_current_cost();
                            this->evalFH(cached_errors, f1, h1);
                            dominated = _filter->isDominated(f1, h1);                    

                            //compute Angle between the two gradient vectors
                            evalH_Jt_e(cached_weights, gH);
                            evalF_Jt_e(cached_weights, gF);
                            double angle = innerProduct(gH, gF);
                            angle /= (norm_L2(gH) + norm_L2(gF));
                            // cout << alpha  << "   " << angle << " " << f1 << " " << h1 << endl;
                                                         
                            if (!dominated && angle > maxAngle)
                            {
                                maxAngle = angle;
                                this->copyToAllParameters(&best_x[0]);
                            }
                            alpha += _hStepSizeScale;                        
                            this->copyFromAllParameters(&x_saved[0]);                            
                        }
                        this->copyFromAllParameters(&best_x[0]);
                        stepType = PENALTY;                                                            
                    }
                } //if success LDL
                else
                {
                    _penaltyLambda = _penaltyLambda * 10;
                    cout << " LDL Failed , lambda = " << _penaltyLambda << endl;
                }
                iter++;
            } //while iter

   
            // this->copyFromAllParameters(&best_x[0]);
            // _log_file.close();
        }

    protected:
        Filter *_filter;        
        vector<Vector2d> & _Sigma;
        std::ofstream &_log_file;
    }; // end struct Robust_Filter_LSQ_Optimizer

    
    struct ConstraintCostFunction : public NLSQ_CostFunction
    {
        ConstraintCostFunction (std::vector<int> const& usedParamTypes, 
                                vector<Vector2d> const& Sigma,
                                Matrix<int> const& correspondingParams)
        : NLSQ_CostFunction (usedParamTypes, correspondingParams, 1),
                            _Sigma(Sigma)
        {
        }

        virtual void evalResidual (int const k, Vector<double> &e) const
        {
            double s = _Sigma[k][0];
            e[0] = s;
        }

        virtual void fillJacobian(int const whichParam, int const paramIx, int const k, Matrix<double>&Jdst) const
        {
            makeZeroMatrix(Jdst);
            Jdst[0][0] = 1.0;
        }

        protected:
            vector<Vector2d> const& _Sigma;
    };

} // end namespace Robust_LSQ

#endif

