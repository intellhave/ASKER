// -*- C++ -*-
#ifndef ROBUST_COMMON_H
#define ROBUST_COMMON_H

#include "Math/v3d_linear.h"

#include <vector>
#include <random>

//#define USE_TUKEYS_BIWEIGHT
//#define USE_CAUCHY
//#define USE_GEMAN_MCCLURE
#define USE_WELSCH

namespace V3D
{
   using namespace std;

   inline double sqr(double const x) { return x*x; }
   inline double sgn(double const x) { return (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0); }

   double const eps_kappa = 1e-3, eps2_kappa = eps_kappa*eps_kappa;

#if defined(USE_TUKEYS_BIWEIGHT)
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
   inline double dkappa_dw2(double const tau, double const w2)
   {
      double const f = sqrt(3.0)/2;
      double const w = sqrt(w2);
      return f * tau / sqrt(2*w+1);
   }
   inline double dkappa_dw(double const tau, double const w) { return 2.0*w*dkappa_dw2(tau, w*w); }
#elif defined(USE_CAUCHY)
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
#elif defined(USE_WELSCH)
   inline double psi(double const tau2, double const r2)                      { return tau2*(1.0 - exp(-r2/tau2))/2.0; }
   inline double psi_weight(double const tau2, double const r2)               { return exp(-r2/tau2); }
   inline double psi_hat(double const tau2, double const r2, double const w2) { return w2*r2 + tau2*(1.0 + w2*log(w2) - w2); }
# if 1
   inline double kappa(double const tau, double const w2)
   {
      if (w2 == 0) return tau;
      return tau*sqrt(1.0 + w2*(log(w2) - 1.0)) * sgn(1.0 - w2);
   }
   inline double kappa_w(double const tau, double const w) { return kappa(tau, w*w); }
   inline double dkappa_dw2(double const tau, double const w2)
   {
      double const W = std::max(1e-8, w2);
      if (fabs(W - 1.0) > 1e-4)
      {
         double const logW = log(W);
         double const res = tau/2.0*logW/sqrt(1.0 + W*logW - W) * sgn(1.0 - W);
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
#else
   // Smooth truncated quadratic
   inline double psi(double const tau2, double const r2)                      { return (r2 < tau2) ? r2*(2.0 - r2/tau2)/4.0f : tau2/4; }
   inline double psi_weight(double const tau2, double const r2)               { return std::max(0.0, 1.0 - r2/tau2); }
   inline double psi_hat(double const tau2, double const r2, double const w2) { return w2*r2 + tau2/2.0*(w2-1)*(w2-1); }
   inline double kappa(double const tau, double const w2)                     { return 0.70710678118*tau*(w2 - 1); }
   inline double dkappa_dw2(double const tau, double const w2)                { return 0.70710678118*tau; }
   inline double dkappa_dw(double const tau, double const w)                  { return 2.0*0.70710678118*tau*w; }
#endif

#if 0
   double const weight_multiplier = 1.0;
   inline double const omega(double u)     { return weight_multiplier*u; }
   inline double const domega(double u)    { return weight_multiplier; }
   inline double const omega_inv(double w) { return w/weight_multiplier; }
#elif 0
   inline double const omega(double u)     { return 0.5*(u + sqrt(1.0 + u*u)); }
   inline double const domega(double u)    { return 0.5 + 0.5*u/sqrt(1.0 + u*u); }
   inline double const omega_inv(double w) { return w - 0.25/w; }
#else
   double const weight_muliplier = 0.5;
   inline double const omega(double u)     { return std::exp(u * weight_muliplier); }
   inline double const domega(double u)    { return weight_muliplier * std::exp(u * weight_muliplier); }
   inline double const omega_inv(double w) { return std::log(w) / weight_muliplier; }
#endif
   const double init_weight = omega_inv(1.0);
   //const double init_weight = omega_inv(0.1);
   //const double init_weight = omega_inv(0.5);

   int const nIterations = 50;

   int const Dim = 3, N = 1000, N_inliers = 0.4 * N;
   //int const Dim = 3, N = 1000, N_inliers = 0.6 * N;
   double const lower_bound_data = -20, upper_bound_data = 20;

   typedef InlineVector<double, Dim> point_t;

   const double psi_tau = 1.0, sqr_psi_tau = sqr(psi_tau);
   //const double psi_tau = 2.0, sqr_psi_tau = sqr(psi_tau);

} // end namespace V3D

#endif
