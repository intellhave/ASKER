#include "bundle_large_common.h"
#include "robust/bundle_filter_scale.h"
#include "Base/v3d_timer.h"
#include <fstream>

using namespace std;
using namespace V3D;
namespace
{   
    using namespace Robust_LSQ;
    typedef Psi_SmoothTrunc Psi_Robust;
    typedef Psi_Quadratic Psi_Constraints;
    typedef Robust_Filter_MOO_Optimizer Optimizer_Base; 
    
    //**********************************************************************
    #define CAMERA_PARAM_TYPE 0
    #define POINT_PARAM_TYPE 1
    #define SCALE_PARAM_TYPE 2

    struct SparseMetricBundleOptimizer;
    int current_iteration = 0;
    int sigma_init = 5.0;

//**********************************************************************
    struct BundleCostFunction : public NLSQ_CostFunction, public BundleCost_Base
    {
        BundleCostFunction(int const mode, std::vector<int> const &usedParamTypes,
                            double const inlierThreshold,
                            vector<CameraMatrix> const &cams,
                            vector<SimpleDistortionFunction> const &distortions,
                            vector<Vector3d> const &Xs,
                            vector<Vector2d> const &measurements,
                            Matrix<int> const &correspondingParams,
                            vector<Vector2d> const &Sigma
                           )
            : NLSQ_CostFunction(usedParamTypes, correspondingParams, 2),
            BundleCost_Base(mode, inlierThreshold, cams, distortions, Xs, measurements, correspondingParams), 
            _Sigma(Sigma)
        {
        }

        double scale(double const s) const        
        {
            return 1.0/(1 + s*s);            
        }

        double scale_deriv(double const s) const        
        {            
            return -2*s/(1 + s*s)/(1 + s*s);            
        }

        virtual void preIterationCallback()
        {
            this->precompute_residuals();
            this->precompute_bundle_derivatives();
            ++current_iteration;
        }

        virtual void initializeResiduals() { } //this->precompute_residuals(); }

        virtual void evalOrgResidual(int const k, Vector<double> &e) const
        {
            Vector2d const r = _residuals[k];
            e[0] = r[0]; e[1] = r[1];            
        }

        virtual void evalResidual(int const k, Vector<double> &e) const
        {            
            Vector2d const r = _residuals[k];  e[0] = r[0];  e[1] = r[1];            
            //Scale the residual by Sigma
            double s = _Sigma[k][0];  
            V3D::scaleVectorIP(scale(s), e);
        }

        virtual bool isInlier(int const i)
        {
            Vector2d r = _residuals[i];
            double rnorm = norm_L2(r);
            return (rnorm <= _inlierThreshold);
        }

        virtual void fillJacobian(int const whichParam, int const paramIx, int const k, Matrix<double> &Jdst) const
        {
            //Prepare the scale
            double s = _Sigma[k][0];  
            Vector2d r = _residuals[k];
            switch (whichParam)
            {
                case CAMERA_PARAM_TYPE:
                    this->copy_camera_Jacobian(k, Jdst);
                    V3D::scaleMatrixIP(scale(s), Jdst);                                    
                    break;
                case POINT_PARAM_TYPE:
                    copyMatrix(_dp_dX[k], Jdst);
                    V3D::scaleMatrixIP(scale(s), Jdst);
                    break;
                case SCALE_PARAM_TYPE:                    
                    Jdst[0][0] = r[0]*scale_deriv(s);         
                    Jdst[1][0] = r[1]*scale_deriv(s);
                    break;   
                default:
                    assert(false);
            } // end switch
        }     // end fillJacobian()
        
        protected:
            vector<Vector2d> const &_Sigma;
    };        // end struct BundleCostFunction

    //**********************************************************************

    struct SparseMetricBundleOptimizer : public Optimizer_Base
    {
        typedef Optimizer_Base Base;

        SparseMetricBundleOptimizer(int const mode, NLSQ_ParamDesc const &paramDesc,
                                    std::vector<NLSQ_CostFunction *> const &costFunctions,
                                    std::vector<Robust_NLSQ_CostFunction_Base *> const &robustCostFunctions,
                                    vector<CameraMatrix> &cams,
                                    vector<SimpleDistortionFunction> &distortions,
                                    vector<Vector3d> &Xs,
                                    vector<Vector2d> &Sigma,
                                    std::ofstream &log_file
                                    )
    #if defined(USE_LIFTING)
            : Base(paramDesc, costFunctions, std::vector<Lifted_Kernel>(1, Lifted_Kernel(inlier_threshold)), std::vector<double>(1, 1.0)),
    #else
            : Base(paramDesc, costFunctions, robustCostFunctions, Sigma, log_file),
    #endif
            _mode(mode),
            _cams(cams), _distortions(distortions), _Xs(Xs),
            _savedTranslations(cams.size()), _savedRotations(cams.size()), _savedFocalLengths(cams.size()), _savedDistortions(cams.size()),
            _savedXs(Xs.size()), _cachedParamLength(0.0)
            // _Sigma(Sigma)
        {
            //Initialize Sigma
            for (int i = 0; i < _Sigma.size(); ++i)
                _Sigma[i][0] = sigma_init;


            // Since we assume that BA does not alter the inputs too much,
            // we compute the overall length of the parameter vector in advance
            // and return that value as the result of getParameterLength().
            for (int i = 0; i < _cams.size(); ++i)
            {
                _cachedParamLength += sqrNorm_L2(_cams[i].getTranslation());
                _cachedParamLength += 3.0; // Assume eye(3) for R.
            }

            if (mode >= FULL_BUNDLE_FOCAL_LENGTH)
                for (int i = 0; i < _cams.size(); ++i)
                {
                    double const f = _cams[i].getFocalLength();
                    _cachedParamLength += f * f;
                } // end for (i)

            for (int j = 0; j < _Xs.size(); ++j)
                _cachedParamLength += sqrNorm_L2(_Xs[j]);

            _cachedParamLength = sqrt(_cachedParamLength);
        }

        virtual double getParameterLength() const
        {
            return _cachedParamLength;
        }

        virtual void updateParameters(int const paramType, VectorArrayAdapter<double> const &delta)
        {
            switch (paramType)
            {
            case CAMERA_PARAM_TYPE:
            {
                Vector3d T, omega;
                Matrix3x3d R0, dR;

                for (int i = 0; i < _cams.size(); ++i)
                {
                    T = _cams[i].getTranslation();
                    T[0] += delta[i][0];
                    T[1] += delta[i][1];
                    T[2] += delta[i][2];
                    _cams[i].setTranslation(T);

                    if (_mode >= FULL_BUNDLE_METRIC)
                    {
                        // Create incremental rotation using Rodriguez formula.
                        R0 = _cams[i].getRotation();
                        omega[0] = delta[i][3];
                        omega[1] = delta[i][4];
                        omega[2] = delta[i][5];
                        createRotationMatrixRodrigues(omega, dR);
                        _cams[i].setRotation(project_to_SO3(dR * R0));
                    }

                    switch (_mode)
                    {
                    case FULL_BUNDLE_RADIAL:
                    {
                        _distortions[i].k1 += delta[i][7];
                        _distortions[i].k2 += delta[i][8];
                    }
                    case FULL_BUNDLE_FOCAL_LENGTH:
                    {
                        Matrix3x3d K = _cams[i].getIntrinsic();
                        K[0][0] += delta[i][6];
                        K[1][1] += delta[i][6];
                        _cams[i].setIntrinsic(K);
                    }
                    } // end switch (_mode)
                }
                break;
            }
            case POINT_PARAM_TYPE:
            {
                for (int j = 0; j < _Xs.size(); ++j)
                {
                    _Xs[j][0] += delta[j][0];
                    _Xs[j][1] += delta[j][1];
                    _Xs[j][2] += delta[j][2];
                }
                break;
            }
            case SCALE_PARAM_TYPE:
            {
                for (int i = 0; i < _Sigma.size(); ++i)
                {
                    _Sigma[i][0] += delta[i][0];
                }
                break;
            }
            default:
                assert(false);
            } // end switch (paramType)
        }     // end updateParametersA()


        virtual void saveAllParameters() { throw("saveAllParameters()"); }
        virtual void restoreAllParameters() { throw("restoreAllParameters()"); }

        // We assume metric/no rotations BA and no update of the rotation matrices (linearized BA)
        // FIXED: we can handle metric BA now
        virtual void copyToAllParameters(double *dst)
        {
            int pos = 0;
            for (int i = 0; i < _cams.size(); ++i)
            {
                Vector3d const T = _cams[i].getTranslation();
                std::copy_n(T.begin(), 3, dst + pos);
                pos += 3;
            }
            for (int j = 0; j < _Xs.size(); ++j)
            {
                std::copy_n(_Xs[j].begin(), 3, dst + pos);
                pos += 3;
            }

            if (_mode >= FULL_BUNDLE_METRIC)
            {
                Vector3d omega;
                for (int i = 0; i < _cams.size(); ++i)
                {
                    createRodriguesParamFromRotationMatrix(_cams[i].getRotation(), omega);
                    std::copy_n(omega.begin(), 3, dst + pos);
                    pos += 3;                    
                }
            } // end if

            for (int i = 0; i < _Sigma.size(); ++i)
            {
                *(dst+pos) = _Sigma[i][0];
                pos++;    
            }
        }

        virtual void copyFromAllParameters(double const *src)
        {
            int pos = 0;
            Vector3d T;
            for (int i = 0; i < _cams.size(); ++i)
            {
                std::copy_n(src + pos, 3, T.begin());
                pos += 3;
                _cams[i].setTranslation(T);
            }
            for (int j = 0; j < _Xs.size(); ++j)
            {
                std::copy_n(src + pos, 3, _Xs[j].begin());
                pos += 3;
            }

            if (_mode >= FULL_BUNDLE_METRIC)
            {
                Vector3d omega;
                Matrix3x3d R;
                for (int i = 0; i < _cams.size(); ++i)
                {   
                    std::copy_n(src + pos, 3, omega.begin());
                    pos += 3;
                    createRotationMatrixRodrigues(omega, R);
                    _cams[i].setRotation(R);
                }
            } // end if

            for (int i = 0; i < _Sigma.size(); ++i)
            {
                _Sigma[i][0] = *(src+pos);
                pos++;
            }
        }

    protected:
        int const _mode;

        vector<CameraMatrix> &_cams;
        vector<Vector3d> &_Xs;
        vector<SimpleDistortionFunction> &_distortions;

        vector<Vector3d> _savedTranslations;
        vector<Matrix3x3d> _savedRotations;
        vector<Vector3d> _savedXs;

        vector<double> _savedFocalLengths;
        vector<SimpleDistortionFunction> _savedDistortions;

        double _cachedParamLength;
        // vector<Vector2d> &_Sigma;
    }; // end struct SparseMetricBundleOptimizer

    //**********************************************************************

    double E_init_linearized = 0, E_final_linearized = 0;

    int adjustStructureAndMotion(
                                    int const mode,
                                    vector<CameraMatrix> &cams,
                                    vector<SimpleDistortionFunction> &distortions,
                                    vector<Vector3d> &Xs,
                                    vector<Vector2d> const &measurements2d,
                                    vector<int> const &correspondingView,
                                    vector<int> const &correspondingPoint,
                                    double inlierThreshold,
                                    vector<Vector2d> &Sigma
                                )                                
    {
        NLSQ_ParamDesc paramDesc;
        paramDesc.nParamTypes = 3; // Cameras; points; Scale
        paramDesc.dimension[CAMERA_PARAM_TYPE] = cameraParamDimensionFromMode(mode);
        paramDesc.dimension[POINT_PARAM_TYPE] = 3;
        paramDesc.dimension[SCALE_PARAM_TYPE] = 1;

        paramDesc.count[CAMERA_PARAM_TYPE] = cams.size();
        paramDesc.count[POINT_PARAM_TYPE] = Xs.size();
        paramDesc.count[SCALE_PARAM_TYPE] = measurements2d.size();

        vector<int> usedParamTypes;
        usedParamTypes.push_back(CAMERA_PARAM_TYPE);
        usedParamTypes.push_back(POINT_PARAM_TYPE);
        usedParamTypes.push_back(SCALE_PARAM_TYPE);
        
        Matrix<int> correspondingParams(measurements2d.size(), paramDesc.nParamTypes);
        for (int k = 0; k < correspondingParams.num_rows(); ++k)
        {
            correspondingParams[k][0] = correspondingView[k];
            correspondingParams[k][1] = correspondingPoint[k];
            correspondingParams[k][2] = k;
        }

        BundleCostFunction costFun(mode, usedParamTypes, inlierThreshold, 
                                   cams, distortions, Xs, measurements2d, 
                                   correspondingParams, Sigma);

        
        //THe constraint cost function
        vector<int> constraintUsedParamTypes;        
        constraintUsedParamTypes.push_back(SCALE_PARAM_TYPE);

        Matrix<int> constraintCorrespondingParams(measurements2d.size(), 1);
        for (int k = 0; k < constraintCorrespondingParams.num_rows(); ++k)
        {
            constraintCorrespondingParams[k][0] = k;
        }
        ConstraintCostFunction constraintCost(constraintUsedParamTypes, Sigma, constraintCorrespondingParams);

        vector<NLSQ_CostFunction *> costFunctions;
        vector<Robust_NLSQ_CostFunction_Base *> robustCostFunctions;

        //Push the cost functions;
        costFunctions.push_back(&costFun);
        costFunctions.push_back(&constraintCost);

        //Prepare the robust kernels
        double const tau_data = inlierThreshold; // / sqrt(2*Psi_Robust::fun(1000.0));
        cout << "tau_data = " << tau_data << endl;        
        Robust_NLSQ_CostFunction<Psi_Robust> robustCostFun(costFun, tau_data, 1.0);
        Robust_NLSQ_CostFunction<Psi_Constraints> robustConstraintCost(constraintCost, 1.0,1.0 );
        robustCostFunctions.push_back(&robustCostFun);
        robustCostFunctions.push_back(&robustConstraintCost);
    
        std::ofstream log_file("log_filter.txt");
        SparseMetricBundleOptimizer opt(mode, paramDesc, costFunctions, robustCostFunctions, cams, distortions, Xs, Sigma, log_file);
        opt.updateThreshold = 1e-12;
        //opt.updateThreshold = 1e-20;
        opt.maxIterations = 1000;
        opt.tau = 1e-3;
        //opt.tau = 1e-6;   
        
    #if defined(USE_LINEARIZED_BUNDLE)
        double const avg_focal_length = AVG_FOCAL_LENGTH;
        E_init_linearized = evalCurrentObjective(avg_focal_length, inlier_threshold, costFun);
    #endif

        Timer t("BA");
        t.start();
        opt.minimize();
        t.stop();
        cout << " Total Time = " << t.getTime() << " iters = " << opt.currentIteration << endl;
        cout << "Time per iteration: " << t.getTime() / opt.currentIteration << endl;

        log_file.close();



    #if defined(USE_LINEARIZED_BUNDLE)
        E_final_linearized = evalCurrentObjective(avg_focal_length, inlier_threshold, costFun);
    #endif

        //params.lambda = opt.lambda;
        return opt.status;
    }

    } // namespace

    //**********************************************************************

    int main(int argc, char *argv[]) //369
    {
        if (argc != 2)
        {
            cerr << "Usage: " << argv[0] << " <sparse reconstruction file>" << endl;
            return -1;
        }

        ifstream is(argv[1]);
        if (!is)
        {
            cerr << "Cannot open " << argv[1] << endl;
            return -2;
        }

        double const avg_focal_length = AVG_FOCAL_LENGTH;
        V3D::optimizerVerbosenessLevel = 1;

        cout.precision(10);

        int N, M, K;
        is >> N >> M >> K;
        cout << "N (cams) = " << N << " M (points) = " << M << " K (measurements) = " << K << endl;

        cout << "Reading image measurements..." << endl;
        vector<Vector2d> measurements(K);
        vector<int> correspondingView(K, -1);
        vector<int> correspondingPoint(K, -1);
        for (int k = 0; k < K; ++k)
        {
            is >> correspondingView[k];
            is >> correspondingPoint[k];
            is >> measurements[k][0] >> measurements[k][1];
            measurements[k][0] /= avg_focal_length;
            measurements[k][1] /= avg_focal_length;
        } // end for (k)
        cout << "Done." << endl;

        cout << "Reading cameras..." << endl;
        vector<CameraMatrix> cams(N);
        vector<SimpleDistortionFunction> distortions(N);
        for (int i = 0; i < N; ++i)
        {
            Vector3d om, T;
            double f, k1, k2;
            is >> om[0] >> om[1] >> om[2];
            is >> T[0] >> T[1] >> T[2];
            is >> f >> k1 >> k2;

            Matrix3x3d K;
            makeIdentityMatrix(K);
            K[0][0] = K[1][1] = -f / avg_focal_length;
            cams[i].setIntrinsic(K);
            cams[i].setTranslation(T);

            Matrix3x3d R;
            createRotationMatrixRodrigues(om, R);
            cams[i].setRotation(R);

            double const f2 = f * f;
            distortions[i].k1 = k1 * f2;
            distortions[i].k2 = k2 * f2 * f2;

            //cout << "k1 = " << k1 << " k2 = " << k2 << endl;
        } // end for (i)
        cout << "Done." << endl;

        cout << "Reading 3D point..." << endl;
        vector<Vector3d> Xs(M);
        for (int j = 0; j < M; ++j)
            is >> Xs[j][0] >> Xs[j][1] >> Xs[j][2];
        cout << "Done." << endl;

        corrupt_input_data(cams, Xs);

        double init_ratio = showErrorStatistics(avg_focal_length, inlier_threshold, cams, distortions, Xs, measurements, correspondingView, correspondingPoint);
        double E_init = showObjective(avg_focal_length, inlier_threshold, cams, distortions, Xs, measurements, correspondingView, correspondingPoint);

        vector<Vector2d> Sigma(K);
        
        adjustStructureAndMotion(bundle_mode, cams, distortions, Xs, 
                                 measurements, correspondingView, correspondingPoint,
                                 inlier_threshold / avg_focal_length,
                                 Sigma);

    #if defined(USE_LINEARIZED_BUNDLE)
        E_init = E_init_linearized;
    #endif
        
        std::string file_name = "structure_filter.txt";
        ofstream str_file(file_name.c_str());
        for (int i = 0; i < M; ++i)
            str_file << Xs[i][0] << " " << Xs[i][1] << " " << Xs[i][2] << endl;
        str_file.close();


        cout << "inlier_threshold = " << inlier_threshold << endl;
        double final_ratio = showErrorStatistics(avg_focal_length, inlier_threshold, cams, distortions, Xs, measurements, correspondingView, correspondingPoint);
        //showErrorStatistics(KMat, cams, Xs, measurements, correspondingView, correspondingPoint);
    #if defined(USE_LINEARIZED_BUNDLE)
        double const E_final = E_final_linearized;
    #else
        double const E_final = showObjective(avg_focal_length, inlier_threshold, cams, distortions, Xs, measurements, correspondingView, correspondingPoint);
    #endif
        //cout << "E_init = " << E_init << " E_final = " << E_final << " initial ratio = " << init_ratio << " final ratio = " << final_ratio << endl;
        char line[1000];
        sprintf(line, "E_init = %12.3f  E_final = %12.3f  initial ratio = %6f  final ratio = %6f", E_init, E_final, init_ratio, final_ratio);
        cout << line << endl;

        return 0;
} //end namespace
