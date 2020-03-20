// -*- C++ -*-
#ifndef BUNDLE_LARGE_COMMON_H
#define BUNDLE_LARGE_COMMON_H

#include "Math/v3d_linear_tnt.h"
#include "Geometry/v3d_cameramatrix.h"

#include <random>

// #define USE_TUKEYS_BIWEIGHT
// #define USE_LEAST_SQUARES
// #define USE_CAUCHY
// #define USE_GEMAN_MCCLURE
// #define MY_GEMAN_MCCLURE
// #define MY_CAUCHY
// #define USE_WELSCH

// #define USE_LINEARIZED_BUNDLE

namespace V3D
{

   inline Matrix3x3d
   project_to_SO3(Matrix3x3d const R)
   {
      Matrix<double> RR(3, 3);
      copyMatrix(R, RR);
      SVD<double> svd(RR);
      Matrix3x3d res;
      multiply_A_Bt(svd.getU(), svd.getV(), res);
      return res;
   }

   struct SimpleDistortionFunction
   {
         double k1, k2;

         SimpleDistortionFunction()
            : k1(0), k2(0)
         { }

         Vector2d operator()(Vector2d const& xu) const
         {
            double const r2 = xu[0]*xu[0] + xu[1]*xu[1];
            double const r4 = r2*r2;
            double const kr = 1 + k1*r2 + k2*r4;

            Vector2d xd;
            xd[0] = kr * xu[0];
            xd[1] = kr * xu[1];
            return xd;
         }

         Matrix2x2d derivativeWrtRadialParameters(Vector2d const& xu) const
         {
            double const r2 = xu[0]*xu[0] + xu[1]*xu[1];
            double const r4 = r2*r2;

            Matrix2x2d deriv;

            deriv[0][0] = xu[0] * r2; // d xd/d k1
            deriv[0][1] = xu[0] * r4; // d xd/d k2
            deriv[1][0] = xu[1] * r2; // d yd/d k1
            deriv[1][1] = xu[1] * r4; // d yd/d k2
            return deriv;
         }

         Matrix2x2d derivativeWrtUndistortedPoint(Vector2d const& xu) const
         {
            double const r2 = xu[0]*xu[0] + xu[1]*xu[1];
            double const r4 = r2*r2;
            double const kr = 1 + k1*r2 + k2*r4;
            double const dkr = 2*k1 + 4*k2*r2;

            Matrix2x2d deriv;
            deriv[0][0] = kr + xu[0] * xu[0] * dkr; // d xd/d xu
            deriv[0][1] =      xu[0] * xu[1] * dkr; // d xd/d yu
            deriv[1][0] = deriv[0][1];              // d yd/d xu
            deriv[1][1] = kr + xu[1] * xu[1] * dkr; // d yd/d yu
            return deriv;
         }
   }; // end struct SimpleDistortionFunction

//**********************************************************************

   inline double sqr(double const x) { return x*x; }
   inline double sgn(double const x) { return (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0); }
   double const eps_kappa = 1e-3, eps2_kappa = eps_kappa*eps_kappa;

#if defined(USE_TUKEYS_BIWEIGHT)
   constexpr double PSI_CVX_X = sqrt(0.333333333);
   inline double psi(double const tau2, double const r2)
   {
      double const r4 = r2*r2, tau4 = tau2*tau2;
      return (r2 < tau2) ? r2*(3.0 - 3*r2/tau2 + r4/tau4)/6.0f : tau2/6.0;
   }
   inline double psi_weight(double const tau2, double const r2)
   {
      return sqr(std::max(0.0, 1.0 - r2/tau2));
   }
   inline double psi_hat(double const tau2, double const r2, double const w2)
   {
      double const w = sqrt(w2);
      return w2*r2 + tau2/3.0*sqr(w-1)*(2*w+1);
   }

   inline double kappa(double const tau, double const w2)
   {
      double const f = sqrt(1.0/3);
      double const w = sqrt(w2);
      return f * tau * (w - 1)*sqrt(2*w+1);
   }
   inline double kappa_w(double const tau, double const w)
   {
      double const f = sqrt(1.0/3);
      return f * tau * (w - 1)*sqrt(2*w+1);
   }
   inline double dkappa_dw2(double const tau, double const w2)
   {
      double const f = sqrt(3.0)/2;
      double const w = sqrt(w2);
      return f * tau / sqrt(2*w+1);
   }
   inline double dkappa_dw(double const tau, double const w) { return 2.0*w*dkappa_dw2(tau, w*w); }
#elif defined(USE_CAUCHY)
   constexpr double PSI_CVX_X = sqrt(0.5);
   inline double psi(double const tau2, double const r2)                      { return tau2*log1p(r2/tau2)/2.0; }
   inline double psi_weight(double const tau2, double const r2)               { return 1.0/(1.0 + r2/tau2); }
   inline double psi_hat(double const tau2, double const r2, double const w2) { return w2*r2 + tau2*(w2 - log(w2) - 1); }
# if 1
   inline double kappa(double const tau, double const w2)                     { return tau*sqrt(w2 - log(w2) - 1.0); }
   inline double dkappa_dw2(double const tau, double const w2)                { double denom = 2.0*sqrt(w2 - log(w2) - 1.0 + eps2_kappa); return tau*(1.0 - 1.0/w2)/denom; }
   inline double dkappa_dw(double const tau, double const w)                  { double denom = sqrt(w*w - log(w*w) - 1.0 + eps2_kappa); return tau*(w - 1.0/w)/denom; }
# else
   inline double kappa(double const tau, double const w2)                     { return sgn(1-w2) * tau*sqrt(w2 - log(w2) - 1.0); }
   inline double dkappa_dw2(double const tau, double const w2)                { double denom = 2.0*sqrt(w2 - log(w2) - 1.0 + eps2_kappa); return sgn(1-w2) * tau*(1.0 - 1.0/w2)/denom; }
   inline double dkappa_dw(double const tau, double const w)                  { double denom = sqrt(w*w - log(w*w) - 1.0 + eps2_kappa); return sgn(1-w) * tau*(w - 1.0/w)/denom; }
# endif
#elif defined(USE_GEMAN_MCCLURE)
   inline double psi(double const tau2, double const r2)                      { return tau2*r2/(tau2 + r2)/2.0; }
   inline double psi_weight(double const tau2, double const r2)               { return tau2*tau2/sqr(tau2 + r2); }
   inline double psi_hat(double const tau2, double const r2, double const w2) { return w2*r2 + tau2*sqr(sqrt(w2) - 1.0); }
   inline double kappa(double const tau, double const w2)                     { return tau*(sqrt(w2) - 1.0); }
   inline double kappa_w(double const tau, double const w)                    { return tau*(w - 1.0); }
   inline double dkappa_dw2(double const tau, double const w2)                { return tau/2.0/sqrt(w2); }
   inline double dkappa_dw(double const tau, double const w)                  { return tau; }
#elif defined(MY_GEMAN_MCCLURE)   
   constexpr double PSI_CVX_X = 1.0;
   inline double psi(double const tau, double const r)                        { return r*r/(tau*tau + r*r); }
   inline double psi_deriv(double const tau, double const r)                  { return 2*r*tau*tau/sqr(r*r+tau*tau); }
   inline double psi_deriv2(double const tau, double const r)                 { return 2*tau*tau*(r*r+tau*tau)*(tau*tau-3*r*r)/sqr(sqr(r*r+tau*tau)); }
   inline double psi_weight(double const tau2, double const r2)               { return tau2*tau2/sqr(tau2 + r2); }
#elif defined(MY_CAUCHY)   
   constexpr double PSI_CVX_X = 1.0;
   inline double psi(double const tau, double const r)                        { return 0.5*tau*tau*log1p(r*r/tau/tau); }
   inline double psi_deriv(double const tau, double const r)                  { return r*1.0/(1.0+r*r/tau/tau); }
   inline double psi_deriv2(double const tau, double const r)                 { return 2*tau*tau*(r*r+tau*tau)*(tau*tau-3*r*r)/sqr(sqr(r*r+tau*tau)); }
   inline double psi_weight(double const tau2, double const r2)               { return tau2*tau2/sqr(tau2 + r2); }
   // inline double psi(double const tau, double const r)                        { return 0.5*tau*tau*log1p(r*r/tau*tau); }
   // inline double psi_deriv(double const tau, double const r)                  { return tau*tau*r/(1+r*r/tau/tau); }
   // inline double psi_deriv2(double const tau, double const r)                 { return 2*tau*tau*(r*r+tau*tau)*(tau*tau-3*r*r)/sqr(sqr(r*r+tau*tau)); }
   // inline double psi_weight(double const tau2, double const r2)               { return tau2*tau2/sqr(tau2 + r2); }
#elif defined(USE_WELSCH)
   constexpr double PSI_CVX_X = sqrt(0.5);

   inline double psi(double const tau2, double const r2)                      { return tau2*(1.0 - exp(-r2/tau2))/2.0; }
   inline double psi_weight(double const tau2, double const r2)               { return exp(-r2/tau2); }
   inline double psi_hat(double const tau2, double const r2, double const w2) { return w2*r2 + tau2*(1.0 + w2*log(w2) - w2); }
# if 1
   inline double kappa(double const tau, double const w2)
   {
      if (w2 < 1e-8) return tau;
      if (fabs(w2 - 1.0) < 1e-8) return 0.0;
      return tau*sqrt(1.0 + w2*(log(w2) - 1.0)) * sgn(1.0 - w2);
   }
   inline double kappa_w(double const tau, double const w) { return kappa(tau, w*w); }
   inline double dkappa_dw2(double const tau, double const w2)
   {
      if (w2 < 1e-8) return 10.0*tau;

      if (fabs(w2 - 1.0) > 1e-6)
      {
         double const logW = log(w2);
         double const res = tau/2.0*logW/sqrt(1.0 + w2*logW - w2) * sgn(1.0 - w2);
         //if (isnan(res)) cout << "dkappa_dw2(): w2 = " << w2 << " logW = " << logW << " sgn(1-w2) = " << sgn(1.0 - w2) << endl;
         return res;
      }
      return -tau * 0.70710678118;
   }
   inline double dkappa_dw(double const tau, double const w)
   {
      double const w2 = w*w;
      if (w2 < 1e-8) return 0.0;
      return 2.0*w*dkappa_dw2(tau, w2);
   }
# elif 0
   inline double kappa(double const tau, double const w2)                     { return tau*sqrt(1.0 + w2*(log(w2) - 1.0) + eps2_kappa); }
   inline double dkappa_dw2(double const tau, double const w2)                { double logW = log(w2); return tau/2.0*logW/sqrt(1.0 + w2*logW - w2 + eps2_kappa); }
   inline double dkappa_dw(double const tau, double const w)                  { return 2.0*w*dkappa_dw2(tau, w*w); }
# else
   inline double kappa(double const tau, double const w2)                     { return tau*sqrt(1.0 + w2*(log(w2) - 1.0) + eps2_kappa) * sgn(1.0 - w2); }
   inline double dkappa_dw2(double const tau, double const w2)                { double logW = log(w2); return tau/2.0*logW/sqrt(1.0 + w2*logW - w2 + eps2_kappa) * sgn(1.0 - w2); }
   inline double dkappa_dw(double const tau, double const w)                  { return 2.0*w*dkappa_dw2(tau, w*w); }
# endif
#elif defined(USE_LEAST_SQUARES)
   constexpr double PSI_CVX_X = (0.333333333);
   inline double psi(double const tau, double const r)                        { return r*r; }
   inline double psi_deriv(double const tau, double const r)                  { return 2*r; }
   inline double psi_deriv2(double const tau, double const r)                  { return 2; }
   inline double psi_weight(double const tau, double const r)                  { return 1; }
#else
   // Smooth truncated quadratic
   constexpr double PSI_CVX_X = sqrt(0.333333333);
   inline double psi(double const tau2, double const r2)                      { return (r2 < tau2) ? r2*(2.0 - r2/tau2)/4.0f : tau2/4; }
   inline double psi_weight(double const tau2, double const r2)               { return std::max(0.0, 1.0 - r2/tau2); }
   inline double psi2_weight(double const tau2, double const r2)              { return (r2 < tau2) ? -1.0/tau2 : 0; }
   inline double psi_hat(double const tau2, double const r2, double const w2) { return w2*r2 + tau2/2.0*(w2-1)*(w2-1); }
   inline double kappa(double const tau, double const w2)                     { return 0.70710678118*tau*(w2 - 1); }
   inline double kappa_w(double const tau, double const w)                    { return 0.70710678118*tau*(w*w - 1); }
   inline double dkappa_dw2(double const tau, double const w2)                { return 0.70710678118*tau; }
   inline double dkappa_dw(double const tau, double const w)                  { return 2.0*0.70710678118*tau*w; }
#endif

   inline double
   showErrorStatistics(double const avg_focal_length, double const inlierThreshold,
                       vector<CameraMatrix> const& cams,
                       vector<SimpleDistortionFunction> const& distortions,
                       vector<Vector3d> const& Xs,
                       vector<Vector2d> const& measurements,
                       vector<int> const& correspondingView,
                       vector<int> const& correspondingPoint)
   {
      int const nBins = 51;
      vector<int> histogram_x(nBins, 0), histogram_y(nBins, 0);

      int const K = measurements.size();

      double const inlierThreshold1 = PSI_CVX_X * inlierThreshold;

      int nInliers = 0;

      double meanReprojectionError = 0.0, inlierReprojectionError = 0.0;
      for (int k = 0; k < K; ++k)
      {
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];
         Vector2d p = cams[i].projectPoint(distortions[i], Xs[j]);

         double const reprojectionError = avg_focal_length * norm_L2(p - measurements[k]);

         double const error_x = avg_focal_length * (p[0] - measurements[k][0]) / inlierThreshold;
         double const error_y = avg_focal_length * (p[1] - measurements[k][1]) / inlierThreshold;
         int const bin_x = std::max(0, std::min(nBins-1, int(nBins/2 + 10*error_x + 0.5)));
         int const bin_y = std::max(0, std::min(nBins-1, int(nBins/2 + 10*error_y + 0.5)));
         histogram_x[bin_x] += 1; histogram_y[bin_y] += 1;

         meanReprojectionError += reprojectionError;
         if (reprojectionError <= inlierThreshold1)
         {
            ++nInliers;
            inlierReprojectionError += reprojectionError;
         }
      }
      cout << "mean reprojection error = " << meanReprojectionError/K << endl;
      cout << "inlier mean reprojection error = " << inlierReprojectionError/nInliers << " " << nInliers << " / " << K << " inliers." << endl;
      // cout << "hist_x = "; displayVector(histogram_x);
      // cout << "hist_y = "; displayVector(histogram_y);
      return double(nInliers) / K;
   }

   inline double
   showObjective(double const avg_focal_length, double const inlierThreshold,
                 vector<CameraMatrix> const& cams,
                 vector<SimpleDistortionFunction> const& distortions,
                 vector<Vector3d> const& Xs,
                 vector<Vector2d> const& measurements,
                 vector<int> const& correspondingView,
                 vector<int> const& correspondingPoint)
   {
      int const K = measurements.size();

      double const tau2 = inlierThreshold*inlierThreshold;
      double const avg_focal_length2 = avg_focal_length*avg_focal_length;

      double obj = 0.0;
      for (int k = 0; k < K; ++k)
      {
         int const i = correspondingView[k];
         int const j = correspondingPoint[k];
         Vector2d p = cams[i].projectPoint(distortions[i], Xs[j]);

         double const r2 = avg_focal_length2 * sqrNorm_L2(p - measurements[k]);
         obj += psi(tau2, r2);
      }
      //cout << "true objective = " << obj << endl;
      return obj;
   }

   inline void
   corrupt_input_data(vector<CameraMatrix> &cams, vector<Vector3d> &Xs)
   {
      return;

      int const N = cams.size(), M = Xs.size();

      if (1)
      {
         //double const corruption_level = 0.05;
         double const corruption_level = 0.001;

         Vector3d centerX, centerC; makeZeroVector(centerX); makeZeroVector(centerC);
         for (int j = 0; j < M; ++j) addVectorsIP(Xs[j], centerX);
         for (int i = 0; i < N; ++i) addVectorsIP(cams[i].cameraCenter(), centerC);
         scaleVectorIP(1.0 / M, centerX); scaleVectorIP(1.0 / N, centerC);

         vector<double> dists(M, 0.0);

         for (int j = 0; j < M; ++j) dists[j] = norm_L2(Xs[j]);
         std::sort(dists.begin(), dists.end());

         double const thX = dists[M/2];
         makeZeroVector(centerX); int nPoints = 0;
         for (int j = 0; j < M; ++j)
            if (norm_L2(Xs[j]) <= thX)
            {
               addVectorsIP(Xs[j], centerX);
               ++nPoints;
            }
         scaleVectorIP(1.0/nPoints, centerX);

         for (int j = 0; j < M; ++j) dists[j] = distance_L2(Xs[j], centerX);
         std::sort(dists.begin(), dists.end());

         double const scale = dists[M/2];

         //cout << "scale = " << scale << " dists[0] = " << dists[0] << " dists[end] = " << dists.back() << " centerX = "; displayVector(centerX);

         std::default_random_engine rng;
         std::normal_distribution<double> dist(0.0, corruption_level * scale);

         for (int j = 0; j < M; ++j)
         {
            Xs[j][0] += dist(rng);
            Xs[j][1] += dist(rng);
            Xs[j][2] += dist(rng);
         }

         //for (int i = 0; i < N; ++i) displayVector(cams[i].cameraCenter());
         
      }
      else if (1)
      {
         std::default_random_engine rng;
         std::normal_distribution<double> dist(0.0, 1.0);

         for (int j = 0; j < M; ++j)
         {
            Xs[j][0] = dist(rng); Xs[j][1] = dist(rng); Xs[j][2] = dist(rng);
         }
      }
   } // end corrupt_input_data()

//**********************************************************************

   //double const AVG_FOCAL_LENGTH = 1000.0;
   double const AVG_FOCAL_LENGTH = 1.0;

   enum
   {
      FULL_BUNDLE_NO_ROTATIONS = -1,
      FULL_BUNDLE_METRIC = 0,
      FULL_BUNDLE_FOCAL_LENGTH = 1, // f
      FULL_BUNDLE_RADIAL = 2,       // f, k1, k2
   };

   typedef InlineMatrix<double, 3, 6> Matrix3x6d;
   typedef InlineMatrix<double, 2, 6> Matrix2x6d;

   static int cameraParamDimensionFromMode(int mode)
   {
      switch (mode)
      {
         case FULL_BUNDLE_NO_ROTATIONS: return 3;
         case FULL_BUNDLE_METRIC:       return 6;
         case FULL_BUNDLE_FOCAL_LENGTH: return 7;
         case FULL_BUNDLE_RADIAL:       return 9;
      }
      return 0;
   }

   struct BundleCost_Base
   {
         BundleCost_Base(int const mode, double const inlierThreshold,
                         vector<CameraMatrix> const& cams,
                         vector<SimpleDistortionFunction> const& distortions,
                         vector<Vector3d > const& Xs,
                         vector<Vector2d > const& measurements,
                         Matrix<int> const& correspondingParams)
            : _mode(mode), _cams(cams), _distortions(distortions), _Xs(Xs), _measurements(measurements), _correspondingParams_base(correspondingParams),
              _inlierThreshold(inlierThreshold), _sqrInlierThreshold(inlierThreshold*inlierThreshold),
              _dp_dk1k2(measurements.size()), _dp_df(measurements.size()), _dp_dRT(measurements.size()), _dp_dX(measurements.size()), _residuals(measurements.size())
#if defined(USE_LINEARIZED_BUNDLE)
            , _cams0(cams), _distortions0(distortions), _Xs0(Xs)
#endif
         {
#if defined(USE_LINEARIZED_BUNDLE)
            int const K = _measurements.size();
            _residuals0.resize(K);
            for (int k = 0; k < K; ++k)
            {
               int const view  = _correspondingParams_base[k][0];
               int const point = _correspondingParams_base[k][1];

               Vector2d const q = this->projectPoint(_Xs[point], view);
               _residuals0[k] = q - _measurements[k];
            } // end for (k)

            this->precompute_bundle_derivatives(true);
#endif
         }

#if defined(USE_LINEARIZED_BUNDLE)
         vector<Vector2d> const& residuals0() const { return _residuals0; }
#endif

      protected:
         void poseDerivatives(int i, int j, Vector3d& XX, Matrix3x6d& d_dRT, Matrix3x3d& d_dX) const
         {
            XX = _cams[i].transformPointIntoCameraSpace(_Xs[j]);

            // See Frank Dellaerts bundle adjustment tutorial.
            // d(dR * R0 * X + t)/d omega = -[R0 * X]_x
            Matrix3x3d J;
            makeCrossProductMatrix(XX - _cams[i].getTranslation(), J);
            scaleMatrixIP(-1.0, J);

            // Now the transformation from world coords into camera space is xx = Rx + T
            // Hence the derivative of x wrt. T is just the identity matrix.
            makeIdentityMatrix(d_dRT);
            copyMatrixSlice(J, 0, 0, 3, 3, d_dRT, 0, 3);

            // The derivative of Rx+T wrt x is just R.
            copyMatrix(_cams[i].getRotation(), d_dX);
         } // end poseDerivatives()

         void precompute_residuals()
         {
            int const K = _measurements.size();
            for (int k = 0; k < K; ++k)
            {
               int const view  = _correspondingParams_base[k][0];
               int const point = _correspondingParams_base[k][1];
#if !defined(USE_LINEARIZED_BUNDLE)
               Vector2d const q = this->projectPoint(_Xs[point], view);
               _residuals[k] = q - _measurements[k];
#else
               Vector3d const delta_X = _Xs[point] - _Xs0[point];
               Vector3d const delta_T = _cams[view].getTranslation() - _cams0[view].getTranslation();
               InlineVector<double, 6> delta_RT; makeZeroVector(delta_RT);
               delta_RT[0] = delta_T[0]; delta_RT[1] = delta_T[1]; delta_RT[2] = delta_T[2];

               Vector2d r_X, r_RT;
               multiply_A_v(_dp_dX[k], delta_X, r_X);
               multiply_A_v(_dp_dRT[k], delta_RT, r_RT);
               _residuals[k] = _residuals0[k] + r_X + r_RT;
#endif
            } // end for (k)
         } // end precompute_residuals()

#if !defined(USE_LINEARIZED_BUNDLE)
         static constexpr bool precompute_default_arg = true;
#else
         static constexpr bool precompute_default_arg = false;
#endif

         void precompute_bundle_derivatives(bool precompute = precompute_default_arg)
         {
            if (!precompute) return;

            int const K = _measurements.size();
            for (int k = 0; k < K; ++k)
            {
               int const view  = _correspondingParams_base[k][0];
               int const point = _correspondingParams_base[k][1];

               Vector3d XX;
               Matrix3x6d dXX_dRT;
               Matrix3x3d dXX_dX;
               this->poseDerivatives(view, point, XX, dXX_dRT, dXX_dX);

               Vector2d xu; // undistorted image point
               xu[0] = XX[0] / XX[2];
               xu[1] = XX[1] / XX[2];

               Vector2d const xd = _distortions[view](xu); // distorted image point

               double const focalLength = _cams[view].getFocalLength();

               Matrix2x2d dp_dxd;
               dp_dxd[0][0] = focalLength; dp_dxd[0][1] = 0;
               dp_dxd[1][0] = 0;           dp_dxd[1][1] = focalLength;

               Matrix2x3d dxu_dXX;
               dxu_dXX[0][0] = 1.0f / XX[2]; dxu_dXX[0][1] = 0;            dxu_dXX[0][2] = -XX[0]/(XX[2]*XX[2]);
               dxu_dXX[1][0] = 0;            dxu_dXX[1][1] = 1.0f / XX[2]; dxu_dXX[1][2] = -XX[1]/(XX[2]*XX[2]);

               Matrix2x2d dxd_dxu = _distortions[view].derivativeWrtUndistortedPoint(xu);
               Matrix2x2d dp_dxu = dp_dxd * dxd_dxu;
               Matrix2x3d dp_dXX = dp_dxu * dxu_dXX;

               switch (_mode)
               {
                  case FULL_BUNDLE_RADIAL:
                  {
                     Matrix2x2d dxd_dk1k2 = _distortions[view].derivativeWrtRadialParameters(xu);
                     _dp_dk1k2[k] = dp_dxd * dxd_dk1k2;
                     // No break here!
                  }
                  case FULL_BUNDLE_FOCAL_LENGTH:
                  {
                     _dp_df[k] = xd;
                  }
                  case FULL_BUNDLE_METRIC:
                  case FULL_BUNDLE_NO_ROTATIONS:
                  {
                     Matrix2x6d& dp_dRT = _dp_dRT[k];
                     multiply_A_B(dp_dXX, dXX_dRT, dp_dRT);
                  }
               } // end switch

               // Jacobian w.r.t. 3d points
               multiply_A_B(dp_dXX, dXX_dX, _dp_dX[k]);
            } // end for (k)
         } // end precompute_bundle_derivatives()

         void copy_camera_Jacobian(int const k, Matrix<double>& Jdst) const
         {
            switch (_mode)
            {
               case FULL_BUNDLE_RADIAL:
               {
                  copyMatrixSlice(_dp_dk1k2[k], 0, 0, 2, 2, Jdst, 0, 7);
                  // No break here!
               }
               case FULL_BUNDLE_FOCAL_LENGTH:
               {
                  Jdst[0][6] = _dp_df[k][0];
                  Jdst[1][6] = _dp_df[k][1];
                  // No break here!
               }
               case FULL_BUNDLE_METRIC:
               {
                  copyMatrixSlice(_dp_dRT[k], 0, 0, 2, 6, Jdst, 0, 0);
                  break;
               }
               case FULL_BUNDLE_NO_ROTATIONS:
               {
                  copyMatrixSlice(_dp_dRT[k], 0, 0, 2, 3, Jdst, 0, 0);
               }
            } // end switch
         } // end copy_camera_Jacobian()

         int const _mode;

         vector<CameraMatrix>             const& _cams;
         vector<SimpleDistortionFunction> const& _distortions;
         vector<Vector3d>                 const& _Xs;

         double const _inlierThreshold, _sqrInlierThreshold;

         vector<Vector2d> const& _measurements;
         Matrix<int>      const& _correspondingParams_base;

         vector<Matrix2x2d> _dp_dk1k2;
         vector<Vector2d>   _dp_df;
         vector<Matrix2x6d> _dp_dRT;
         vector<Matrix2x3d> _dp_dX;
         vector<Matrix2x2d> _dpsi_dp;
         vector<Vector2d>   _residuals;
#if defined(USE_LINEARIZED_BUNDLE)
         vector<CameraMatrix>             const _cams0;
         vector<SimpleDistortionFunction> const _distortions0;
         vector<Vector3d>                 const _Xs0;
         vector<Vector2d>                       _residuals0;
#endif

      private:
         Vector2d projectPoint(Vector3d const& X, int i) const
         {
#if 0
            return _cams[i].projectPoint(_distortions[i], X);
#else
            Vector3d const XX = _cams[i].transformPointIntoCameraSpace(X);
            Vector2d const xu(XX[0] / XX[2],  XX[1] / XX[2]);
            Vector2d const xd = _distortions[i](xu);
            return _cams[i].getFocalLength() * xd;
#endif
         }

         friend double evalCurrentObjective(double const avg_focal_length, double const inlierThreshold, BundleCost_Base &cost);
   }; // end struct BundleCost_Base

   inline double evalCurrentObjective(double const avg_focal_length, double const inlierThreshold, BundleCost_Base &cost)
   {
      double const tau2 = sqr(inlierThreshold), avg_focal_length2 = sqr(avg_focal_length);
      cost.precompute_residuals();
      auto const& residuals = cost._residuals;
      int const K = residuals.size();

      double err = 0.0;
      for (int k = 0; k < K; ++k)
      {
         double const r2 = avg_focal_length2 * sqrNorm_L2(residuals[k]);
         err += psi(tau2, r2);
      }
      return err;
   } // end evalCurrentObjective()

//**********************************************************************

   // int const bundle_mode = FULL_BUNDLE_NO_ROTATIONS;
   int const bundle_mode = FULL_BUNDLE_METRIC;
   // int const bundle_mode = FULL_BUNDLE_FOCAL_LENGTH;
   // int const bundle_mode = FULL_BUNDLE_RADIAL;

   double const inlier_threshold = 1.0;
   //double const inlier_threshold = 1.0;
   // double const inlier_threshold = 0.5;

   int const nBundleIterations = 100;

} // end namespace V3D

#endif
