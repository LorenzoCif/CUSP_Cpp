#include <RcppArmadillo.h>
#include <RcppGSL.h>
#include <mvt.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include<omp.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppGSL)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]
#include <progress.hpp>
#include <progress_bar.hpp>
// [[Rcpp::depends(RcppProgress)]]

using namespace arma;
using namespace std;

static double const log2pi = std::log(2.0 * M_PI);

/*============================================================================
 * compute the densiti of t multivariate density distribution with diagonal 
 * covariance
 *
 * args:
 * - nu:            degrees of freedom
 * - diag_psi_iw:   diagonal variance

 * double function
 =============================================================================*/

inline double log_t_density_diag(const double nu,  const double diag_psi_iw, const vec &x){
  int k=x.n_elem;
  
  double adjustment_factor = diag_psi_iw;
  
  double det_sig_half= k* log(adjustment_factor)/2;
  
  double quad_form= dot(x,x)/  (adjustment_factor*nu); 

  double density =gsl_sf_lnpoch(nu/2 , ((double)k)/2)-  //lgamma((nu+k)/2 ) -lgamma(nu/2) - 
    (k* log((datum::pi)*nu) + (nu+k)*log1p(quad_form ) )/2 -det_sig_half;
  return density;
}

// =============================================================================
//' @export
//' @name CUSP
//' @title C++ function to estimate the factrial model with CUSP prior
//' via Gibbs sampling
//' @keywords internal
//'
//' @param y a matrix of observations n x p
//' @param n_iter number of iteration
//' @param n_burn number of burn-in iterations
//' @param start_adapt iteration to start the adaption of the latent space dimension
//' @param prior_par list of prior parameters:
//'   * a_sigma shape of sigma
//'   * b_sigma rate of sigma
//    * a_theta shape of theta CUSP
//'   * b_theta rate of theta CUSP
//'   * alpha concetration od DP for CUSP prior
//'   * theta_inf variance of spike in CUSP prior
//' @param start list of initial values of parameters:
//'   * H initial number of active factors
//'   * eta matrix of latent factors
//'   * sigma vector of the diagonal variances of data
//'   
//' @return list:
//'   * n_iter number of iteration
//'   * n_burn number of burn-in iterations
//'   * H_star number of active factors
//'   * Lambda matrix of loadings
//'   * sigma diagonal variances
//

// [[Rcpp::export]]
Rcpp::List CUSP(mat y, uword n_iter, uword n_burn, uword start_adapt, Rcpp::List prior_par, Rcpp::List start){

  Progress progress(n_iter, true);
  
  double a_sigma = prior_par["a_sigma"];
  double b_sigma = prior_par["b_sigma"];
  double a_theta = prior_par["a_theta"];
  double b_theta = prior_par["b_theta"];
  double alpha = prior_par["alpha"];
  double theta_inf = prior_par["theta_inf"];
  
  double alpha_0 = -1;
  double alpha_1 = -0.0005;
  
  uword p = y.n_cols;
  uword n = y.n_rows;
  uword H = start["H"];
  
  // initialization
  mat Lambda(p, H);            // p x H 
  mat eta = start["eta"];      // n x H
  vec sigma = start["sigma"];  // p x 1
  vec theta = start["theta"];  // H x 1
  uvec z(H);
  vec v(H);
  vec w(H, fill::value(1));
  w = w/H;
  
  // output
  vec H_out(n_iter-n_burn);
  vec H_star_out(n_iter-n_burn);
  field<mat> Lambda_out(n_iter-n_burn);
  mat sigma_out(n_iter-n_burn, p);
  
  // temp variables
  double flag = 0;
  mat V_Lambda_temp(H, H);
  vec m_Lambda_temp(H);
  double a_sigma_temp;
  double b_sigma_temp;
  double m1_temp;
  double m2_temp;
  mat V_eta_temp(H,H);
  vec m_eta_temp(H);
  double n1_v;
  double n2_v;
  uword H_star=H;
  uvec active(H);
  double Lambda_sum_squared=0;
  mat Lambda_temp(p, H);
  mat eta_temp(n, H);
  vec theta_temp(H);
  vec w_temp(H);
  vec v_temp(H);
  mat eta2(H, H);
  
  // arma_rng::set_seed(123);
  
  // START GIBBS SAMPLING
  
  for(uword it=0; it<n_iter; it++){
    
    // UPDATE LAMBDA
    
    eta2 = eta.t() * eta;
    for (uword j=0; j<p; j++){
      V_Lambda_temp = chol(diagmat(1/theta) + 1/sigma(j)*eta2);
      
      // see https://gallery.rcpp.org/articles/simulation-smoother-using-rcpparmadillo/
      // sampling from multivariate guassian via back-foward substitution 
      Lambda.row(j) = (solve(trimatu(V_Lambda_temp ), randn<vec>(H) + 
        solve(trimatl((V_Lambda_temp ).t()), 1/sigma(j) * (eta.t() * y.col(j))))).t();
    }
    
    
    // UPDATe SIGMA
    
    for (uword j=0; j<p; j++){
      m1_temp = 0;
      m2_temp = 0;
      a_sigma_temp = a_sigma + 0.5*n;
      for (uword i=0; i<n; i++){
        // m1_temp = m1_temp + Lambda(j, h) * eta(i, h);
        m1_temp = as_scalar(Lambda.row(j) * eta.row(i).as_col());
        m2_temp = m2_temp + pow(y(i,j) - m1_temp, 2);
      }
      b_sigma_temp = b_sigma + 0.5*m2_temp;

      sigma(j) = 1 / randg(distr_param(a_sigma_temp, 1/b_sigma_temp));

    }
    
    // UPDATE ETA
    
    for (uword i=0; i<n; i++){
      V_eta_temp = chol(eye(H, H) + Lambda.t() * diagmat(1/sigma) *  Lambda);
      eta.row(i) = (solve(trimatu(V_eta_temp), randn<vec>(H) + 
        solve(trimatl((V_eta_temp ).t()), Lambda.t() * diagmat(1/sigma) * (y.row(i)).t()))).t();
    }
    
    // SAMPLE Z

    Rcpp::NumericVector temp_prob(H);
    for (uword h=0; h<H; h++){
      for (uword l=0; l<H; l++){
        if (l<=h) {
          temp_prob(l) = w(l) * as_scalar(prod(normpdf(Lambda.col(h), 0, sqrt(theta_inf))));
        } else {
          temp_prob(l) = exp(log(w(l)) + log_t_density_diag( 2*a_theta, b_theta/a_theta, Lambda.col(h)));
        }
        
      }
      
      Rcpp::IntegerVector h_index = Rcpp::seq(0, H-1);
      
      if (sum(temp_prob) == 0){
        cout << "Warning: all probabilities for sampling z are equal to 0" << endl;
        temp_prob.fill(0);
        temp_prob(H-1) = 1;
      }
      
      z(h) = Rcpp::sample(h_index, 1, false, temp_prob)(0);
    }
    
    // SAMPLE v
    if (H > 1){
      for (uword l=0; l<(H-1); l++){
        n1_v = sum(z == l);
        n2_v = sum(z > l);
        v(l) = Rcpp::as<double>(Rcpp::wrap(R::rbeta(1 + n1_v, alpha + n2_v)));
      }
    }
    
    v(H-1) = 1;
    
    // UPSATE W
    w(0) = v(0);
    if (H>1){
      for (uword l=1; l<H; l++){
        w(l) = v(l) * (1-v(l-1)) * (w(l-1)) / (v(l-1));
      }
    }
    
    
    // UPDATE theta

    for (uword h=0; h<H; h++){
      if (z(h) <= h){
        theta(h) = theta_inf;
      } else{
        Lambda_sum_squared = as_scalar(trans(Lambda.col(h)) * Lambda.col(h));
        theta(h) = 1 / randg(distr_param(a_theta + 0.5*p, 1/( b_theta + 0.5*Lambda_sum_squared)));
      }
    }
    
    
    // ADAPT LATENT DIMENSION H
    if ((it > start_adapt) & (randu() < exp(alpha_0+alpha_1*it))){
      Lambda_temp = Lambda;
      eta_temp = eta;
      theta_temp = theta;
      w_temp = w;
      v_temp = v;
      
      
      uvec H_index = linspace<uvec>(0, H-1, H);
      H_star = sum(z > H_index);

      if (H_star > 0){
        active.resize(H_star);
        active = find(z > H_index);
      }
      
      
      if ((H_star < (H-1)) & (H_star > 0)){
        flag = 1;
        H = H_star + 1;
        Lambda.set_size(p,H);
        Lambda.cols(0, H-2) = Lambda_temp.cols(active);
        eta.set_size(n, H);
        eta.cols(0, H-2) = eta_temp.cols(active);
        theta.set_size(H);
        theta.subvec(0, H-2) = theta_temp(active);
        v.set_size(H);
        v.subvec(0, H-2) = v_temp(active);
        w.set_size(H);
        w.subvec(0, H-2) = w_temp(active);
        eta.tail_cols(1) = randn(n);
        theta.tail(1) = theta_inf;
        w.tail(1) = 0;
        w.tail(1) = 1 - sum(w);
        Lambda.tail_cols(1) = randn(p) * sqrt(theta_inf);
      } else {
        flag = 1;
        if (H_star>0){
          H = H + 1;
        } else{
          H = 1;
        }
        
        Lambda.resize(p,H);
        eta.resize(n, H);
        theta.resize(H);
        v.resize(H);
        w.resize(H);
        eta.tail_cols(1) = randn(n);
        theta.tail(1) = theta_inf;
        if (H > 1){
          v.tail(1) = Rcpp::as<double>(Rcpp::wrap(R::rbeta(1 , alpha)));
          w.tail(1) = v.tail(1) * (1-v(H-2)) * (w(H-2)) / (v(H-2));
        } else{
          v.tail(1) = 1;
          w.tail(1) = 1;
        }
        
        Lambda.tail_cols(1) =  randn(p) * sqrt(theta_inf);
      }
      
      if (flag==1){
        flag = 0;
        Lambda_temp.set_size(p, H);
        eta_temp.set_size(n, H);
        theta_temp.set_size(H);
        w_temp.set_size(H);
        v_temp.set_size(H);
        V_Lambda_temp.set_size(H, H);
        m_Lambda_temp.set_size(H);
        V_eta_temp.set_size(H,H);
        m_eta_temp.set_size(H);
        z.set_size(H);
      }
    }
    
    
    if ((it >= n_burn)){
      H_out(it-n_burn) = H;
      H_star_out(it-n_burn) = H_star;
      Lambda_out(it-n_burn) = Lambda;
      sigma_out.row(it-n_burn) = sigma.as_row();
    } 
   
   progress.increment();
 }
  
  
  Rcpp::List out;
  out["n_iter"] = n_iter;
  out["n_burn"] = n_burn;
  out["Lambda"] = Lambda_out;
  out["sigma"] = sigma_out;
  out["H_star"] = H_star_out;
  
  return out;
  
}
